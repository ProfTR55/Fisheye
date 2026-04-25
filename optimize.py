"""Calibration routine from line/plumb-line constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

from loss import LossConfig, evaluate_objective, straightness_residuals, total_residual_vector
from model import FisheyePoly4Model


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
    verbose: int = 2
    use_projection_priors: bool = True
    projection_prior_families: Tuple[str, ...] = (
        "equidistant",
        "equisolid",
        "stereographic",
    )
    projection_prior_half_angles_deg: Tuple[float, ...] = (95.0, 110.0, 125.0)


@dataclass
class CalibrationResult:
    model: FisheyePoly4Model
    loss_config: LossConfig
    optimize_config: OptimizeConfig
    final_metrics: Dict[str, float]
    history: List[Dict[str, float]] = field(default_factory=list)
    line_groups_used: List[np.ndarray] = field(default_factory=list)
    debug_lines: List[Dict[str, np.ndarray]] = field(default_factory=list)
    start_summaries: List[Dict[str, object]] = field(default_factory=list)
    best_init_label: str = ""


def _build_bounds(image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    lower = np.array(
        [
            -0.15 * w,
            -0.15 * h,
            0.05,
            -6.0,
            -6.0,
            -6.0,
            0.65,
            0.65,
            -0.18,
            -0.18,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            1.15 * w,
            1.15 * h,
            6.0,
            6.0,
            6.0,
            6.0,
            1.45,
            1.45,
            0.18,
            0.18,
        ],
        dtype=np.float64,
    )
    return lower, upper


def _projection_theta_samples(
    family: str, half_angle_rad: float, num_samples: int = 64
) -> np.ndarray:
    """Sample theta(r_hat) for a physical fisheye projection family on r_hat in [0, 1].

    All families are normalized so that theta(1) = half_angle_rad, matching how
    the poly4 model interprets the unit-radius edge of the image.
    """
    r = np.linspace(0.0, 1.0, num_samples)
    family = family.lower()
    if family == "equidistant":
        # theta = k * r, with k chosen so theta(1) = half_angle.
        return half_angle_rad * r
    if family == "equisolid":
        # r ∝ sin(theta/2). Invert: theta = 2 * arcsin(r * sin(half_angle/2)).
        s = np.sin(0.5 * half_angle_rad)
        return 2.0 * np.arcsin(np.clip(r * s, -1.0, 1.0))
    if family == "stereographic":
        # r ∝ tan(theta/2). Invert: theta = 2 * arctan(r * tan(half_angle/2)).
        t = np.tan(0.5 * half_angle_rad)
        return 2.0 * np.arctan(r * t)
    if family == "orthographic":
        # r ∝ sin(theta). Invert: theta = arcsin(r * sin(half_angle)).
        s = np.sin(min(half_angle_rad, 0.5 * np.pi - 1e-3))
        return np.arcsin(np.clip(r * s, -1.0, 1.0))
    raise ValueError(f"Unknown projection family: {family}")


def _fit_poly4_to_theta(theta_samples: np.ndarray) -> np.ndarray:
    """Least-squares fit of theta(r) = a1 r + a2 r^2 + a3 r^3 + a4 r^4."""
    n = theta_samples.shape[0]
    r = np.linspace(0.0, 1.0, n)
    # Drop r=0 (theta=0) to avoid trivial constraint and improve conditioning.
    mask = r > 0.0
    A = np.stack([r[mask], r[mask] ** 2, r[mask] ** 3, r[mask] ** 4], axis=1)
    b = theta_samples[mask]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs.astype(np.float64)


def _generate_initial_parameters(
    image_shape: Tuple[int, int],
    optimize_cfg: OptimizeConfig,
    bounds: Tuple[np.ndarray, np.ndarray],
) -> List[Tuple[np.ndarray, str]]:
    """Generate initial states for multi-start solving.

    Returns a list of (parameter_vector, label) pairs. The label identifies the
    initialization family for diagnostic reporting.
    """
    h, w = image_shape[:2]
    lower, upper = bounds
    cx0 = 0.5 * (w - 1)
    cy0 = 0.5 * (h - 1)
    rng = np.random.default_rng(optimize_cfg.random_seed)
    init_list: List[Tuple[np.ndarray, str]] = []

    def _push(coeffs: np.ndarray, cx: float, cy: float, sx: float, sy: float,
              p1: float, p2: float, label: str) -> None:
        vec = np.array(
            [cx, cy, *coeffs.tolist(), sx, sy, p1, p2], dtype=np.float64
        )
        vec = np.clip(vec, lower + 1e-6, upper - 1e-6)
        init_list.append((vec, label))

    # Linear (poly4 with a1=theta_max only) baseline anchor — same as legacy default.
    base_half_angle = float(optimize_cfg.init_half_angle_deg)
    base_coeffs = np.array(
        [np.deg2rad(base_half_angle), 0.0, 0.0, 0.0], dtype=np.float64
    )
    _push(base_coeffs, cx0, cy0, 1.0, 1.0, 0.0, 0.0, label="linear_baseline")

    # Physical projection priors — anchor the search near plausible fisheye shapes.
    if optimize_cfg.use_projection_priors:
        for family in optimize_cfg.projection_prior_families:
            for half_deg in optimize_cfg.projection_prior_half_angles_deg:
                half_rad = np.deg2rad(float(half_deg))
                try:
                    theta = _projection_theta_samples(family, half_rad)
                    coeffs = _fit_poly4_to_theta(theta)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                _push(
                    coeffs, cx0, cy0, 1.0, 1.0, 0.0, 0.0,
                    label=f"{family}_{int(round(half_deg))}deg",
                )

    # Randomized starts for additional diversity.
    n_random = max(0, int(optimize_cfg.multi_start) - 1)
    for k in range(n_random):
        half_angle = optimize_cfg.init_half_angle_deg + rng.uniform(
            -optimize_cfg.start_half_angle_span_deg,
            optimize_cfg.start_half_angle_span_deg,
        )
        half_angle = float(np.clip(half_angle, 65.0, 145.0))
        coeffs = np.array([np.deg2rad(half_angle), 0.0, 0.0, 0.0], dtype=np.float64)
        cx = cx0 + rng.normal(scale=optimize_cfg.start_center_jitter_frac * w)
        cy = cy0 + rng.normal(scale=optimize_cfg.start_center_jitter_frac * h)
        sx = 1.0 + rng.uniform(
            -optimize_cfg.start_anisotropy_jitter, optimize_cfg.start_anisotropy_jitter
        )
        sy = 1.0 + rng.uniform(
            -optimize_cfg.start_anisotropy_jitter, optimize_cfg.start_anisotropy_jitter
        )
        p1 = rng.uniform(
            -optimize_cfg.start_tangential_jitter, optimize_cfg.start_tangential_jitter
        )
        p2 = rng.uniform(
            -optimize_cfg.start_tangential_jitter, optimize_cfg.start_tangential_jitter
        )
        _push(coeffs, cx, cy, sx, sy, p1, p2, label=f"random_{k}")

    return init_list


def _trim_line_outliers(
    debug_lines: List[Dict[str, np.ndarray]],
    quantile: float,
    min_points_after_trim: int,
) -> List[np.ndarray]:
    trimmed: List[np.ndarray] = []
    for item in debug_lines:
        img_points = item["img_points"]
        dist = np.abs(item["rect_distances"])
        if dist.size == 0:
            continue
        thr = np.quantile(dist, quantile)
        keep = dist <= thr
        pts = img_points[keep]
        if pts.shape[0] < min_points_after_trim:
            pts = img_points
        trimmed.append(pts)
    return trimmed


def calibrate_from_lines(
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    optimize_cfg: OptimizeConfig | None = None,
    loss_cfg: LossConfig | None = None,
) -> CalibrationResult:
    if not line_groups:
        raise ValueError("No line constraints were provided for calibration.")

    optimize_cfg = optimize_cfg or OptimizeConfig()
    loss_cfg = loss_cfg or LossConfig()
    bounds = _build_bounds(image_shape)
    init_params_list = _generate_initial_parameters(
        image_shape=image_shape, optimize_cfg=optimize_cfg, bounds=bounds
    )
    rounds = max(1, int(optimize_cfg.alternating_rounds))
    per_round_iters = max(30, int(np.ceil(optimize_cfg.max_iters / rounds)))
    base_lines = [np.asarray(line, dtype=np.float64).reshape(-1, 2) for line in line_groups]

    best_result: CalibrationResult | None = None
    best_objective = np.inf
    start_summaries: List[Dict[str, float]] = []

    for start_id, (init_params, init_label) in enumerate(init_params_list):
        params = init_params.copy()
        history: List[Dict[str, float]] = []
        current_lines = [line.copy() for line in base_lines]
        for outer in range(rounds):
            result = least_squares(
                fun=total_residual_vector,
                x0=params,
                args=(image_shape, current_lines, loss_cfg),
                method="trf",
                bounds=bounds,
                max_nfev=per_round_iters,
                verbose=max(0, optimize_cfg.verbose - 1) if start_id == 0 else 0,
                x_scale="jac",
                loss="soft_l1",
                f_scale=1.0,
            )
            params = result.x
            model = FisheyePoly4Model.from_vector(params)
            metrics = evaluate_objective(model, image_shape, current_lines, loss_cfg)
            metrics.update(
                {
                    "start_id": float(start_id),
                    "outer_round": float(outer),
                    "nfev": float(result.nfev),
                    "status": float(result.status),
                }
            )
            history.append(metrics)

            _, _, debug_lines = straightness_residuals(model, image_shape, current_lines, loss_cfg)
            if outer < rounds - 1 and debug_lines:
                current_lines = _trim_line_outliers(
                    debug_lines=debug_lines,
                    quantile=optimize_cfg.outlier_trim_quantile,
                    min_points_after_trim=optimize_cfg.min_points_after_trim,
                )

        final_model = FisheyePoly4Model.from_vector(params)
        final_metrics = evaluate_objective(final_model, image_shape, current_lines, loss_cfg)
        _, _, final_debug_lines = straightness_residuals(
            final_model, image_shape, current_lines, loss_cfg
        )
        start_summary = {
            "start_id": float(start_id),
            "init_label": init_label,
            "objective": float(final_metrics["objective"]),
            "line_rmse": float(final_metrics["line_rmse"]),
            "total_rmse": float(final_metrics["total_rmse"]),
            "theta_edge_rad": float(final_model.g(np.array([1.0]))[0]),
            "cx": float(final_model.cx),
            "cy": float(final_model.cy),
            "sx": float(final_model.sx),
            "sy": float(final_model.sy),
            "p1": float(final_model.p1),
            "p2": float(final_model.p2),
        }
        start_summaries.append(start_summary)
        if final_metrics["objective"] < best_objective:
            best_objective = float(final_metrics["objective"])
            best_result = CalibrationResult(
                model=final_model,
                loss_config=loss_cfg,
                optimize_config=optimize_cfg,
                final_metrics=final_metrics,
                history=history,
                line_groups_used=current_lines,
                debug_lines=final_debug_lines,
                start_summaries=start_summaries.copy(),
                best_init_label=init_label,
            )

    if best_result is None:
        raise RuntimeError("Calibration failed in all starts.")
    best_result.start_summaries = start_summaries
    return best_result
