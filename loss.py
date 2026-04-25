"""Straight-line calibration loss and regularization terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from model import FisheyePoly4Model


@dataclass
class LossConfig:
    min_points_per_line: int = 8
    min_valid_points_per_line: int = 6
    monotonic_min_slope: float = 0.02
    lambda_mono: float = 40.0
    lambda_smooth: float = 0.35
    lambda_center: float = 8.0
    lambda_theta_bounds: float = 20.0
    lambda_coeff_l2: float = 0.08
    lambda_anisotropy: float = 3.0
    lambda_tangential: float = 4.0
    min_theta_at_edge: float = np.deg2rad(55.0)
    max_theta_at_edge: float = np.deg2rad(150.0)
    regularizer_grid_size: int = 64
    invalid_point_penalty: float = 2.5
    edge_weight_alpha: float = 1.6
    edge_weight_power: float = 2.0
    edge_weight_max: float = 4.0
    line_span_weight_alpha: float = 0.8
    line_span_weight_power: float = 1.0


def fit_2d_line_pca(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 2D line via PCA and return (center, direction, signed distances)."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    center = np.mean(pts, axis=0)
    centered = pts - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    signed_dist = centered @ normal
    return center, direction, signed_dist


def straightness_residuals(
    model: FisheyePoly4Model,
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    cfg: LossConfig,
) -> Tuple[np.ndarray, Dict[str, float], List[Dict[str, np.ndarray]]]:
    """Compute point-to-best-line residuals in rectified space."""
    residuals: List[np.ndarray] = []
    accepted_lines = 0
    accepted_points = 0
    debug_lines: List[Dict[str, np.ndarray]] = []

    static_kept_lines = 0
    static_kept_points = 0
    for line_pts in line_groups:
        pts = np.asarray(line_pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < cfg.min_points_per_line:
            continue
        static_kept_lines += 1
        static_kept_points += pts.shape[0]

        rho_hat = model.normalized_radius(pts, image_shape=image_shape)
        point_weight = 1.0 + cfg.edge_weight_alpha * (rho_hat**cfg.edge_weight_power)
        point_weight = np.clip(point_weight, 1.0, cfg.edge_weight_max)
        radial_span = float(np.max(rho_hat) - np.min(rho_hat))
        line_weight = 1.0 + cfg.line_span_weight_alpha * (
            radial_span**cfg.line_span_weight_power
        )
        total_weight = point_weight * line_weight

        rect_pts, valid = model.image_points_to_rectified(pts, image_shape=image_shape)
        rect_valid = rect_pts[valid]

        if rect_valid.shape[0] >= cfg.min_valid_points_per_line:
            center, direction, _ = fit_2d_line_pca(rect_valid)
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            signed_dist_all = (rect_pts - center) @ normal
            signed_dist_all[~valid] = cfg.invalid_point_penalty
            weighted = np.sqrt(total_weight) * signed_dist_all
            residuals.append(weighted)
            accepted_lines += 1
            accepted_points += rect_valid.shape[0]
            debug_lines.append(
                {
                    "img_points": pts[valid],
                    "rect_points": rect_valid,
                    "rect_center": center,
                    "rect_direction": direction,
                    "rect_distances": signed_dist_all[valid],
                    "weights": total_weight[valid],
                }
            )
        else:
            # Keep residual length constant for least-squares stability.
            residuals.append(
                np.sqrt(total_weight)
                * np.full(pts.shape[0], cfg.invalid_point_penalty, dtype=np.float64)
            )

    if not residuals:
        # Force optimizer away from degenerate/no-line states.
        residual = np.array([50.0], dtype=np.float64)
    else:
        residual = np.concatenate(residuals, axis=0)

    metrics = {
        "accepted_lines": float(accepted_lines),
        "accepted_points": float(accepted_points),
        "static_kept_lines": float(static_kept_lines),
        "static_kept_points": float(static_kept_points),
        "straight_rmse": float(np.sqrt(np.mean(residual**2))),
    }
    return residual, metrics, debug_lines


def regularization_residuals(
    model: FisheyePoly4Model,
    image_shape: Tuple[int, int],
    cfg: LossConfig,
) -> np.ndarray:
    h, w = image_shape[:2]
    r = np.linspace(0.0, 1.0, cfg.regularizer_grid_size)

    g_prime = model.g_prime(r)
    mono_violation = np.maximum(0.0, cfg.monotonic_min_slope - g_prime)
    mono_res = np.sqrt(cfg.lambda_mono) * mono_violation

    g_second = model.g_second(r)
    smooth_res = np.sqrt(cfg.lambda_smooth) * g_second

    cx0 = 0.5 * (w - 1)
    cy0 = 0.5 * (h - 1)
    center_res = np.array(
        [
            np.sqrt(cfg.lambda_center) * (model.cx - cx0) / max(w, 1),
            np.sqrt(cfg.lambda_center) * (model.cy - cy0) / max(h, 1),
        ],
        dtype=np.float64,
    )

    theta_edge = model.g(np.array([1.0], dtype=np.float64))[0]
    theta_low = np.sqrt(cfg.lambda_theta_bounds) * max(0.0, cfg.min_theta_at_edge - theta_edge)
    theta_high = np.sqrt(cfg.lambda_theta_bounds) * max(0.0, theta_edge - cfg.max_theta_at_edge)
    theta_res = np.array([theta_low, theta_high], dtype=np.float64)

    coeff_res = np.sqrt(cfg.lambda_coeff_l2) * model.coeffs
    anisotropy_res = np.sqrt(cfg.lambda_anisotropy) * np.array(
        [model.sx - 1.0, model.sy - 1.0], dtype=np.float64
    )
    tangential_res = np.sqrt(cfg.lambda_tangential) * np.array(
        [model.p1, model.p2], dtype=np.float64
    )

    return np.concatenate(
        [
            mono_res,
            smooth_res,
            center_res,
            theta_res,
            coeff_res,
            anisotropy_res,
            tangential_res,
        ]
    )


def total_residual_vector(
    param_vector: np.ndarray,
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    cfg: LossConfig,
) -> np.ndarray:
    model = FisheyePoly4Model.from_vector(param_vector)
    line_res, _, _ = straightness_residuals(
        model=model, image_shape=image_shape, line_groups=line_groups, cfg=cfg
    )
    reg_res = regularization_residuals(model=model, image_shape=image_shape, cfg=cfg)
    return np.concatenate([line_res, reg_res], axis=0)


def evaluate_objective(
    model: FisheyePoly4Model,
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    cfg: LossConfig,
) -> Dict[str, float]:
    line_res, metrics, _ = straightness_residuals(
        model=model, image_shape=image_shape, line_groups=line_groups, cfg=cfg
    )
    reg_res = regularization_residuals(model=model, image_shape=image_shape, cfg=cfg)
    total = np.concatenate([line_res, reg_res], axis=0)
    metrics.update(
        {
            "line_rmse": float(np.sqrt(np.mean(line_res**2))),
            "reg_rmse": float(np.sqrt(np.mean(reg_res**2))),
            "total_rmse": float(np.sqrt(np.mean(total**2))),
            "objective": float(np.mean(total**2)),
        }
    )
    return metrics
