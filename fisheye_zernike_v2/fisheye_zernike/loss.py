"""Straight-line loss and projection regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .model import RadialFisheyeModel


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
    lambda_high_order: float = 0.05
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
    spatial_balance_grid_size: int = 4
    spatial_balance_strength: float = 0.0


def fit_2d_line_pca(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    center = np.mean(pts, axis=0)
    centered = pts - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    return center, direction, centered @ normal


def _spatial_balance_weights(
    line_groups: List[np.ndarray], image_shape: Tuple[int, int], cfg: LossConfig
) -> List[float]:
    grid = max(1, int(cfg.spatial_balance_grid_size))
    strength = max(0.0, float(cfg.spatial_balance_strength))
    if grid <= 1 or strength <= 0.0:
        return [1.0 for _ in line_groups]
    h, w = image_shape[:2]
    cells: List[Tuple[int, int] | None] = []
    counts: Dict[Tuple[int, int], int] = {}
    for line in line_groups:
        pts = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < cfg.min_points_per_line:
            cells.append(None)
            continue
        center = np.mean(pts, axis=0)
        cell = (
            int(np.clip(np.floor(center[0] / max(w, 1) * grid), 0, grid - 1)),
            int(np.clip(np.floor(center[1] / max(h, 1) * grid), 0, grid - 1)),
        )
        cells.append(cell)
        counts[cell] = counts.get(cell, 0) + 1
    return [1.0 if cell is None else float(counts[cell]) ** (-strength) for cell in cells]


def straightness_residuals(
    model: RadialFisheyeModel,
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    cfg: LossConfig,
) -> Tuple[np.ndarray, Dict[str, float], List[Dict[str, np.ndarray]]]:
    residuals: List[np.ndarray] = []
    raw_distances: List[np.ndarray] = []
    debug_lines: List[Dict[str, np.ndarray]] = []
    accepted_lines = 0
    accepted_points = 0
    static_kept_lines = 0
    static_kept_points = 0
    spatial_weights = _spatial_balance_weights(line_groups, image_shape, cfg)

    for line_idx, line_pts in enumerate(line_groups):
        pts = np.asarray(line_pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < cfg.min_points_per_line:
            continue
        static_kept_lines += 1
        static_kept_points += pts.shape[0]

        rho_hat = model.normalized_radius(pts, image_shape)
        point_weight = 1.0 + cfg.edge_weight_alpha * (rho_hat**cfg.edge_weight_power)
        point_weight = np.clip(point_weight, 1.0, cfg.edge_weight_max)
        radial_span = float(np.max(rho_hat) - np.min(rho_hat))
        line_weight = 1.0 + cfg.line_span_weight_alpha * radial_span
        line_weight *= spatial_weights[line_idx]
        total_weight = point_weight * line_weight

        rect_pts, valid = model.image_points_to_rectified(pts, image_shape)
        rect_valid = rect_pts[valid]
        if rect_valid.shape[0] >= cfg.min_valid_points_per_line:
            center, direction, _ = fit_2d_line_pca(rect_valid)
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            signed = (rect_pts - center) @ normal
            signed[~valid] = cfg.invalid_point_penalty
            residuals.append(np.sqrt(total_weight) * signed)
            raw_distances.append(signed[valid])
            accepted_lines += 1
            accepted_points += rect_valid.shape[0]
            debug_lines.append(
                {
                    "img_points": pts[valid],
                    "rect_points": rect_valid,
                    "rect_center": center,
                    "rect_direction": direction,
                    "rect_distances": signed[valid],
                    "weights": total_weight[valid],
                }
            )
        else:
            residuals.append(
                np.sqrt(total_weight)
                * np.full(pts.shape[0], cfg.invalid_point_penalty, dtype=np.float64)
            )

    residual = np.concatenate(residuals, axis=0) if residuals else np.array([50.0])
    raw_concat = np.concatenate(raw_distances, axis=0) if raw_distances else residual
    metrics = {
        "accepted_lines": float(accepted_lines),
        "accepted_points": float(accepted_points),
        "static_kept_lines": float(static_kept_lines),
        "static_kept_points": float(static_kept_points),
        "straight_rmse": float(np.sqrt(np.mean(residual**2))),
        "raw_line_rmse": float(np.sqrt(np.mean(raw_concat**2))),
        "raw_line_mae": float(np.mean(np.abs(raw_concat))),
        "raw_line_p95": float(np.percentile(np.abs(raw_concat), 95)),
    }
    return residual, metrics, debug_lines


def regularization_residuals(
    model: RadialFisheyeModel,
    image_shape: Tuple[int, int],
    cfg: LossConfig,
) -> np.ndarray:
    h, w = image_shape[:2]
    r = np.linspace(0.0, 1.0, cfg.regularizer_grid_size)

    mono_res = np.sqrt(cfg.lambda_mono) * np.maximum(
        0.0, cfg.monotonic_min_slope - model.g_prime(r)
    )
    smooth_res = np.sqrt(cfg.lambda_smooth) * model.g_second(r)

    cx0 = 0.5 * (w - 1)
    cy0 = 0.5 * (h - 1)
    center_res = np.array(
        [
            np.sqrt(cfg.lambda_center) * (model.cx - cx0) / max(w, 1),
            np.sqrt(cfg.lambda_center) * (model.cy - cy0) / max(h, 1),
        ],
        dtype=np.float64,
    )

    theta_edge = float(model.g(np.array([1.0]))[0])
    theta_res = np.array(
        [
            np.sqrt(cfg.lambda_theta_bounds) * max(0.0, cfg.min_theta_at_edge - theta_edge),
            np.sqrt(cfg.lambda_theta_bounds) * max(0.0, theta_edge - cfg.max_theta_at_edge),
        ],
        dtype=np.float64,
    )
    coeff_res = np.sqrt(cfg.lambda_coeff_l2) * model.coeffs
    high_order_res = np.sqrt(cfg.lambda_high_order) * model.coeff_regularization_weights() * model.coeffs
    anisotropy_res = np.sqrt(cfg.lambda_anisotropy) * np.array([model.sx - 1.0, model.sy - 1.0])
    tangential_res = np.sqrt(cfg.lambda_tangential) * np.array([model.p1, model.p2])
    return np.concatenate(
        [
            mono_res,
            smooth_res,
            center_res,
            theta_res,
            coeff_res,
            high_order_res,
            anisotropy_res,
            tangential_res,
        ]
    )


def total_residual_vector(
    param_vector: np.ndarray,
    family: str,
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    cfg: LossConfig,
) -> np.ndarray:
    model = RadialFisheyeModel.from_vector(family, param_vector)
    line_res, _, _ = straightness_residuals(model, image_shape, line_groups, cfg)
    reg_res = regularization_residuals(model, image_shape, cfg)
    return np.concatenate([line_res, reg_res], axis=0)


def evaluate_objective(
    model: RadialFisheyeModel,
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    cfg: LossConfig,
) -> Dict[str, float]:
    line_res, metrics, _ = straightness_residuals(model, image_shape, line_groups, cfg)
    reg_res = regularization_residuals(model, image_shape, cfg)
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
