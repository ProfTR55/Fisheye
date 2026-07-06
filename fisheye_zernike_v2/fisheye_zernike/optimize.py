"""Calibration routine for line-based fisheye models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

from .loss import LossConfig, evaluate_objective, straightness_residuals, total_residual_vector
from .model import RadialFisheyeModel, coefficient_count


@dataclass
class OptimizeConfig:
    max_iters: int = 500
    alternating_rounds: int = 4
    outlier_trim_quantile: float = 0.96
    min_points_after_trim: int = 16
    init_half_angle_deg: float = 105.0
    multi_start: int = 3
    start_half_angle_span_deg: float = 20.0
    start_center_jitter_frac: float = 0.04
    start_anisotropy_jitter: float = 0.08
    start_tangential_jitter: float = 0.02
    random_seed: int = 13
    verbose: int = 0
    use_projection_priors: bool = False
    projection_prior_families: Tuple[str, ...] = ("equidistant", "equisolid", "stereographic")
    projection_prior_half_angles_deg: Tuple[float, ...] = (95.0, 110.0, 125.0)
    enable_tangential: bool = True


@dataclass
class CalibrationResult:
    model: RadialFisheyeModel
    loss_config: LossConfig
    optimize_config: OptimizeConfig
    final_metrics: Dict[str, float]
    history: List[Dict[str, float]] = field(default_factory=list)
    line_groups_used: List[np.ndarray] = field(default_factory=list)
    debug_lines: List[Dict[str, np.ndarray]] = field(default_factory=list)
    start_summaries: List[Dict[str, object]] = field(default_factory=list)
    best_init_label: str = ""


def _build_bounds(
    family: str, image_shape: Tuple[int, int], enable_tangential: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    n = coefficient_count(family)
    tang = 0.18 if enable_tangential else 1e-6
    lower = np.array(
        [-0.15 * w, -0.15 * h, 0.05, *([-6.0] * (n - 1)), 0.65, 0.65, -tang, -tang],
        dtype=np.float64,
    )
    upper = np.array(
        [1.15 * w, 1.15 * h, 6.0, *([6.0] * (n - 1)), 1.45, 1.45, tang, tang],
        dtype=np.float64,
    )
    return lower, upper


def _projection_theta_samples(family: str, half_angle_rad: float, num_samples: int = 80) -> np.ndarray:
    r = np.linspace(0.0, 1.0, num_samples)
    family = family.lower()
    if family == "equidistant":
        return half_angle_rad * r
    if family == "equisolid":
        s = np.sin(0.5 * half_angle_rad)
        return 2.0 * np.arcsin(np.clip(r * s, -1.0, 1.0))
    if family == "stereographic":
        t = np.tan(0.5 * half_angle_rad)
        return 2.0 * np.arctan(r * t)
    if family == "orthographic":
        s = np.sin(min(half_angle_rad, 0.5 * np.pi - 1e-3))
        return np.arcsin(np.clip(r * s, -1.0, 1.0))
    raise ValueError(f"Unknown projection family: {family}")


def _fit_coeffs_to_theta(model_family: str, theta_samples: np.ndarray) -> np.ndarray:
    n = theta_samples.shape[0]
    r = np.linspace(0.0, 1.0, n)
    mask = r > 0.0
    tmp = RadialFisheyeModel.initial_from_shape((10, 10), family=model_family)
    A = tmp._basis(r[mask], 0)
    coeffs, *_ = np.linalg.lstsq(A, theta_samples[mask], rcond=None)
    return coeffs.astype(np.float64)


def _generate_initial_parameters(
    family: str,
    image_shape: Tuple[int, int],
    cfg: OptimizeConfig,
    bounds: Tuple[np.ndarray, np.ndarray],
) -> List[Tuple[np.ndarray, str]]:
    h, w = image_shape[:2]
    lower, upper = bounds
    rng = np.random.default_rng(cfg.random_seed)
    init_list: List[Tuple[np.ndarray, str]] = []
    cx0 = 0.5 * (w - 1)
    cy0 = 0.5 * (h - 1)

    def push(coeffs: np.ndarray, cx: float, cy: float, sx: float, sy: float, p1: float, p2: float, label: str) -> None:
        vec = np.array([cx, cy, *coeffs.tolist(), sx, sy, p1, p2], dtype=np.float64)
        init_list.append((np.clip(vec, lower + 1e-6, upper - 1e-6), label))

    coeffs = np.zeros(coefficient_count(family), dtype=np.float64)
    coeffs[0] = np.deg2rad(cfg.init_half_angle_deg)
    push(coeffs, cx0, cy0, 1.0, 1.0, 0.0, 0.0, "linear_baseline")

    if cfg.use_projection_priors:
        for projection in cfg.projection_prior_families:
            for half_deg in cfg.projection_prior_half_angles_deg:
                try:
                    theta = _projection_theta_samples(projection, np.deg2rad(float(half_deg)))
                    coeffs = _fit_coeffs_to_theta(family, theta)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                push(coeffs, cx0, cy0, 1.0, 1.0, 0.0, 0.0, f"{projection}_{int(round(half_deg))}deg")

    for k in range(max(0, int(cfg.multi_start) - 1)):
        coeffs = np.zeros(coefficient_count(family), dtype=np.float64)
        half_angle = cfg.init_half_angle_deg + rng.uniform(
            -cfg.start_half_angle_span_deg, cfg.start_half_angle_span_deg
        )
        coeffs[0] = np.deg2rad(float(np.clip(half_angle, 65.0, 145.0)))
        push(
            coeffs,
            cx0 + rng.normal(scale=cfg.start_center_jitter_frac * w),
            cy0 + rng.normal(scale=cfg.start_center_jitter_frac * h),
            1.0 + rng.uniform(-cfg.start_anisotropy_jitter, cfg.start_anisotropy_jitter),
            1.0 + rng.uniform(-cfg.start_anisotropy_jitter, cfg.start_anisotropy_jitter),
            rng.uniform(-cfg.start_tangential_jitter, cfg.start_tangential_jitter),
            rng.uniform(-cfg.start_tangential_jitter, cfg.start_tangential_jitter),
            f"random_{k}",
        )
    return init_list


def _trim_line_outliers(
    debug_lines: List[Dict[str, np.ndarray]], quantile: float, min_points_after_trim: int
) -> List[np.ndarray]:
    trimmed: List[np.ndarray] = []
    for item in debug_lines:
        pts = item["img_points"]
        dist = np.abs(item["rect_distances"])
        if dist.size == 0:
            continue
        keep = dist <= np.quantile(dist, quantile)
        kept = pts[keep]
        trimmed.append(kept if kept.shape[0] >= min_points_after_trim else pts)
    return trimmed


def calibrate_from_lines(
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    model_family: str = "zernike4",
    optimize_cfg: OptimizeConfig | None = None,
    loss_cfg: LossConfig | None = None,
) -> CalibrationResult:
    if not line_groups:
        raise ValueError("No line constraints were provided for calibration.")
    model_family = model_family.lower()
    optimize_cfg = optimize_cfg or OptimizeConfig()
    loss_cfg = loss_cfg or LossConfig()
    bounds = _build_bounds(model_family, image_shape, optimize_cfg.enable_tangential)
    starts = _generate_initial_parameters(model_family, image_shape, optimize_cfg, bounds)
    rounds = max(1, int(optimize_cfg.alternating_rounds))
    per_round_iters = max(30, int(np.ceil(optimize_cfg.max_iters / rounds)))
    base_lines = [np.asarray(line, dtype=np.float64).reshape(-1, 2) for line in line_groups]

    best: CalibrationResult | None = None
    best_objective = np.inf
    start_summaries: List[Dict[str, object]] = []

    for start_id, (params, label) in enumerate(starts):
        current_lines = [line.copy() for line in base_lines]
        history: List[Dict[str, float]] = []
        for outer in range(rounds):
            result = least_squares(
                total_residual_vector,
                params,
                args=(model_family, image_shape, current_lines, loss_cfg),
                method="trf",
                bounds=bounds,
                max_nfev=per_round_iters,
                verbose=max(0, optimize_cfg.verbose - 1) if start_id == 0 else 0,
                x_scale="jac",
                loss="soft_l1",
                f_scale=1.0,
            )
            params = result.x
            model = RadialFisheyeModel.from_vector(model_family, params)
            metrics = evaluate_objective(model, image_shape, current_lines, loss_cfg)
            metrics.update({"start_id": float(start_id), "outer_round": float(outer), "nfev": float(result.nfev)})
            history.append(metrics)
            _, _, debug = straightness_residuals(model, image_shape, current_lines, loss_cfg)
            if outer < rounds - 1 and debug:
                current_lines = _trim_line_outliers(
                    debug, optimize_cfg.outlier_trim_quantile, optimize_cfg.min_points_after_trim
                )

        model = RadialFisheyeModel.from_vector(model_family, params)
        final_metrics = evaluate_objective(model, image_shape, current_lines, loss_cfg)
        _, _, final_debug = straightness_residuals(model, image_shape, current_lines, loss_cfg)
        summary = {
            "start_id": float(start_id),
            "init_label": label,
            "objective": float(final_metrics["objective"]),
            "line_rmse": float(final_metrics["line_rmse"]),
            "raw_line_rmse": float(final_metrics["raw_line_rmse"]),
            "theta_edge_rad": float(model.g(np.array([1.0]))[0]),
            "cx": float(model.cx),
            "cy": float(model.cy),
            "sx": float(model.sx),
            "sy": float(model.sy),
            "p1": float(model.p1),
            "p2": float(model.p2),
        }
        start_summaries.append(summary)
        if final_metrics["objective"] < best_objective:
            best_objective = float(final_metrics["objective"])
            best = CalibrationResult(
                model=model,
                loss_config=loss_cfg,
                optimize_config=optimize_cfg,
                final_metrics=final_metrics,
                history=history,
                line_groups_used=current_lines,
                debug_lines=final_debug,
                start_summaries=start_summaries.copy(),
                best_init_label=label,
            )
    if best is None:
        raise RuntimeError("Calibration failed in all starts.")
    best.start_summaries = start_summaries
    return best
