"""Diagnostics and compact confidence reporting."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .model import RadialFisheyeModel


def _line_coverage(line_groups: List[np.ndarray], image_shape: Tuple[int, int], grid: int = 4) -> float:
    if not line_groups:
        return 0.0
    h, w = image_shape[:2]
    cells = set()
    for line in line_groups:
        pts = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            continue
        center = np.mean(pts, axis=0)
        gx = int(np.clip(np.floor(center[0] / max(w, 1) * grid), 0, grid - 1))
        gy = int(np.clip(np.floor(center[1] / max(h, 1) * grid), 0, grid - 1))
        cells.add((gx, gy))
    return float(len(cells) / max(1, grid * grid))


def _max_radial_span(model: RadialFisheyeModel, line_groups: List[np.ndarray], image_shape: Tuple[int, int]) -> float:
    spans = []
    for line in line_groups:
        pts = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        rho = model.normalized_radius(pts, image_shape)
        spans.append(float(np.max(rho) - np.min(rho)))
    return max(spans) if spans else 0.0


def confidence_report(
    model: RadialFisheyeModel,
    image_shape: Tuple[int, int],
    line_groups: List[np.ndarray],
    train_metrics: Dict[str, float],
    validation_metrics: Dict[str, float] | None = None,
) -> Dict[str, object]:
    line_score = float(np.clip(len(line_groups) / 10.0, 0.0, 1.0))
    coverage_score = _line_coverage(line_groups, image_shape)
    radial_span = _max_radial_span(model, line_groups, image_shape)
    radial_score = float(np.clip(radial_span / 0.22, 0.0, 1.0))
    rmse = float(train_metrics.get("raw_line_rmse", np.inf))
    rmse_score = float(np.exp(-rmse / 0.03)) if np.isfinite(rmse) else 0.0
    theta_edge = float(model.g(np.array([1.0]))[0])
    theta_deg = float(np.rad2deg(theta_edge))
    theta_score = 1.0 if 65.0 <= theta_deg <= 140.0 else 0.35
    validation_score = 0.5
    if validation_metrics:
        val = float(validation_metrics.get("raw_line_rmse", np.nan))
        if np.isfinite(val) and np.isfinite(rmse):
            validation_score = float(np.clip(rmse / max(val, 1e-9), 0.0, 1.0))
    score = (
        0.20 * line_score
        + 0.18 * coverage_score
        + 0.20 * radial_score
        + 0.24 * rmse_score
        + 0.10 * theta_score
        + 0.08 * validation_score
    )
    warnings = []
    if len(line_groups) < 4:
        warnings.append("Few line constraints; calibration is weak.")
    if radial_span < 0.08:
        warnings.append(f"Largest line spans only {radial_span:.2f} of normalized radius.")
    if not (55.0 <= theta_deg <= 150.0):
        warnings.append(f"Estimated edge angle {theta_deg:.1f} deg is outside plausible bounds.")
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "components": {
            "line_count": line_score,
            "coverage": coverage_score,
            "radial_span": radial_score,
            "train_rmse": rmse_score,
            "theta_edge": theta_score,
            "validation": validation_score,
        },
        "raw": {
            "num_lines": len(line_groups),
            "max_radial_span": radial_span,
            "train_raw_line_rmse": rmse,
            "validation_raw_line_rmse": None
            if validation_metrics is None
            else validation_metrics.get("raw_line_rmse"),
            "theta_edge_deg": theta_deg,
        },
        "warnings": warnings,
    }


def zernike_coefficient_summary(model: RadialFisheyeModel) -> Dict[str, object]:
    if not model.family.startswith("zernike"):
        return {"enabled": False, "reason": "model is not Zernike-parametrized"}
    names = model.coefficient_names()
    coeffs = [float(v) for v in model.coeffs]
    abs_coeffs = np.abs(model.coeffs)
    dominant = int(np.argmax(abs_coeffs)) if abs_coeffs.size else -1
    return {
        "enabled": True,
        "family": model.family,
        "coefficients": {name: value for name, value in zip(names, coeffs)},
        "dominant_mode": None if dominant < 0 else names[dominant],
        "high_order_energy": float(np.sum(abs_coeffs[1:] ** 2)),
    }
