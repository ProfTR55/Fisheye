"""Calibration confidence scoring and failure detection.

The single-image plumb-line calibration is fundamentally under-determined and
its quality depends strongly on the constraint geometry. This module produces a
post-hoc confidence report that can be inspected, surfaced in the CLI summary,
and used to gate downstream consumers (e.g. by exiting non-zero in strict mode).

Each individual signal is bounded in [0, 1] where 1 means "ideal". The overall
confidence is a weighted geometric mean so that a single very poor signal
collapses the score, mirroring the underlying intuition that even a well-fitted
model can be unreliable if, for instance, all constraint lines cluster in one
quadrant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from model import FisheyePoly4Model


@dataclass
class ConfidenceConfig:
    """Thresholds for converting raw signals to component scores.

    All defaults are calibrated empirically on the included sample images and
    can be overridden from the CLI for ablations.
    """

    min_lines_acceptable: int = 6
    min_lines_hard_floor: int = 3
    coverage_grid_size: int = 4
    min_occupied_cells: int = 4
    radial_span_target: float = 0.55
    train_rmse_warn: float = 0.018
    train_rmse_fail: float = 0.05
    val_to_train_warn: float = 1.6
    val_to_train_fail: float = 2.4
    edge_angle_min_deg: float = 50.0
    edge_angle_max_deg: float = 160.0
    edge_angle_ideal_low_deg: float = 80.0
    edge_angle_ideal_high_deg: float = 130.0
    confidence_warn: float = 0.55
    confidence_fail: float = 0.30


@dataclass
class ConfidenceReport:
    score: float
    level: str  # "ok" | "warn" | "fail"
    components: Dict[str, float] = field(default_factory=dict)
    raw_signals: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    is_acceptable: bool = True


def _occupied_cell_count(
    line_groups: List[np.ndarray], image_shape: Tuple[int, int], grid: int
) -> int:
    if not line_groups:
        return 0
    h, w = image_shape[:2]
    g = max(1, int(grid))
    cells: set[Tuple[int, int]] = set()
    for pts in line_groups:
        center = np.mean(np.asarray(pts, dtype=np.float64).reshape(-1, 2), axis=0)
        gx = int(np.clip(np.floor(center[0] / max(w, 1) * g), 0, g - 1))
        gy = int(np.clip(np.floor(center[1] / max(h, 1) * g), 0, g - 1))
        cells.add((gx, gy))
    return len(cells)


def _radial_span_signal(
    line_groups: List[np.ndarray],
    model: FisheyePoly4Model,
    image_shape: Tuple[int, int],
) -> float:
    """Largest fraction of the [0, 1] r_hat range covered by any single line.

    A long line that crosses both center and edge regions is far more
    informative for the radial map than many short lines clustered at the same
    radius. We take the max single-line span as a robust summary.
    """
    if not line_groups:
        return 0.0
    spans: List[float] = []
    for pts in line_groups:
        rho = model.normalized_radius(pts, image_shape=image_shape)
        if rho.size:
            spans.append(float(np.max(rho) - np.min(rho)))
    return float(np.max(spans)) if spans else 0.0


def _smoothstep(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _angle_score(edge_angle_deg: float, cfg: ConfidenceConfig) -> float:
    """1.0 inside the ideal band, decaying smoothly to 0.0 at hard limits."""
    if edge_angle_deg <= cfg.edge_angle_min_deg or edge_angle_deg >= cfg.edge_angle_max_deg:
        return 0.0
    if cfg.edge_angle_ideal_low_deg <= edge_angle_deg <= cfg.edge_angle_ideal_high_deg:
        return 1.0
    if edge_angle_deg < cfg.edge_angle_ideal_low_deg:
        return _smoothstep(
            edge_angle_deg, cfg.edge_angle_min_deg, cfg.edge_angle_ideal_low_deg
        )
    return 1.0 - _smoothstep(
        edge_angle_deg, cfg.edge_angle_ideal_high_deg, cfg.edge_angle_max_deg
    )


def _component_count(num_lines: int, cfg: ConfidenceConfig) -> float:
    if num_lines <= cfg.min_lines_hard_floor:
        return 0.0
    return _smoothstep(
        float(num_lines), float(cfg.min_lines_hard_floor), float(cfg.min_lines_acceptable)
    )


def _component_coverage(occupied: int, cfg: ConfidenceConfig) -> float:
    target = max(1, cfg.min_occupied_cells)
    return float(np.clip(occupied / target, 0.0, 1.0))


def _component_radial(span: float, cfg: ConfidenceConfig) -> float:
    if cfg.radial_span_target <= 0:
        return 1.0
    return float(np.clip(span / cfg.radial_span_target, 0.0, 1.0))


def _component_rmse(rmse: float, cfg: ConfidenceConfig) -> float:
    if rmse <= cfg.train_rmse_warn:
        return 1.0
    if rmse >= cfg.train_rmse_fail:
        return 0.0
    return 1.0 - _smoothstep(rmse, cfg.train_rmse_warn, cfg.train_rmse_fail)


def _component_generalization(ratio: float | None, cfg: ConfidenceConfig) -> float:
    if ratio is None or not np.isfinite(ratio):
        return 1.0
    if ratio <= cfg.val_to_train_warn:
        return 1.0
    if ratio >= cfg.val_to_train_fail:
        return 0.0
    return 1.0 - _smoothstep(ratio, cfg.val_to_train_warn, cfg.val_to_train_fail)


def _weighted_geometric_mean(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    eps = 1e-3
    log_sum = 0.0
    weight_sum = 0.0
    for name, value in scores.items():
        w = float(weights.get(name, 1.0))
        log_sum += w * np.log(max(value, eps))
        weight_sum += w
    if weight_sum <= 0:
        return 0.0
    return float(np.exp(log_sum / weight_sum))


def assess_confidence(
    model: FisheyePoly4Model,
    image_shape: Tuple[int, int],
    line_groups_used: List[np.ndarray],
    train_metrics: Dict[str, float],
    validation_metrics: Dict[str, float] | None,
    cfg: ConfidenceConfig | None = None,
) -> ConfidenceReport:
    """Compute the calibration confidence report.

    Args:
        model: Final calibrated model.
        image_shape: Source image shape (h, w).
        line_groups_used: Line constraints actually used for calibration.
        train_metrics: Output of ``evaluate_objective`` on the training lines.
        validation_metrics: Same shape as ``train_metrics`` if a held-out split
            was used; otherwise ``None``.
        cfg: Optional override for the threshold configuration.
    """
    cfg = cfg or ConfidenceConfig()

    num_lines = len(line_groups_used)
    occupied = _occupied_cell_count(line_groups_used, image_shape, cfg.coverage_grid_size)
    radial_span = _radial_span_signal(line_groups_used, model, image_shape)
    train_rmse = float(train_metrics.get("raw_line_rmse", train_metrics.get("line_rmse", 0.0)))
    val_rmse = (
        float(validation_metrics.get("raw_line_rmse", validation_metrics.get("line_rmse", 0.0)))
        if validation_metrics
        else None
    )
    val_train_ratio = (
        (val_rmse / train_rmse) if (val_rmse is not None and train_rmse > 1e-6) else None
    )
    edge_angle_deg = float(np.rad2deg(model.g(np.array([1.0]))[0]))

    components: Dict[str, float] = {
        "line_count": _component_count(num_lines, cfg),
        "spatial_coverage": _component_coverage(occupied, cfg),
        "radial_span": _component_radial(radial_span, cfg),
        "train_rmse": _component_rmse(train_rmse, cfg),
        "edge_angle": _angle_score(edge_angle_deg, cfg),
        "generalization": _component_generalization(val_train_ratio, cfg),
    }

    weights = {
        "line_count": 1.0,
        "spatial_coverage": 1.2,
        "radial_span": 1.0,
        "train_rmse": 1.4,
        "edge_angle": 0.8,
        "generalization": 1.2 if val_train_ratio is not None else 0.0,
    }
    score = _weighted_geometric_mean(components, weights)

    warnings: List[str] = []
    if num_lines < cfg.min_lines_acceptable:
        warnings.append(
            f"Only {num_lines} usable line constraint(s); recommend >= {cfg.min_lines_acceptable}."
        )
    if occupied < cfg.min_occupied_cells:
        warnings.append(
            f"Constraints occupy only {occupied} of >= {cfg.min_occupied_cells} target grid cells; "
            "center and anisotropy may be poorly observed."
        )
    if radial_span < 0.5 * cfg.radial_span_target:
        warnings.append(
            f"Largest line spans only {radial_span:.2f} of the radius; edge calibration is weak."
        )
    if train_rmse >= cfg.train_rmse_fail:
        warnings.append(
            f"Train line RMSE {train_rmse:.4f} >= fail threshold {cfg.train_rmse_fail:.4f}."
        )
    elif train_rmse >= cfg.train_rmse_warn:
        warnings.append(
            f"Train line RMSE {train_rmse:.4f} >= warn threshold {cfg.train_rmse_warn:.4f}."
        )
    if val_train_ratio is not None and val_train_ratio >= cfg.val_to_train_warn:
        warnings.append(
            f"Validation/train RMSE ratio {val_train_ratio:.2f} suggests overfitting "
            f"(warn>={cfg.val_to_train_warn:.2f}, fail>={cfg.val_to_train_fail:.2f})."
        )
    if (
        edge_angle_deg < cfg.edge_angle_min_deg
        or edge_angle_deg > cfg.edge_angle_max_deg
    ):
        warnings.append(
            f"Edge angle theta(1)={edge_angle_deg:.1f}deg falls outside plausible band "
            f"[{cfg.edge_angle_min_deg:.0f}, {cfg.edge_angle_max_deg:.0f}]."
        )

    if score >= cfg.confidence_warn and not any(
        c < 0.05 for c in components.values()
    ):
        level = "ok"
        is_acceptable = True
    elif score >= cfg.confidence_fail:
        level = "warn"
        is_acceptable = True
    else:
        level = "fail"
        is_acceptable = False

    raw_signals: Dict[str, float] = {
        "num_lines_used": float(num_lines),
        "occupied_cells": float(occupied),
        "max_radial_span": float(radial_span),
        "train_raw_line_rmse": float(train_rmse),
        "validation_raw_line_rmse": float(val_rmse) if val_rmse is not None else float("nan"),
        "val_to_train_ratio": float(val_train_ratio) if val_train_ratio is not None else float("nan"),
        "edge_angle_deg": float(edge_angle_deg),
    }

    return ConfidenceReport(
        score=float(score),
        level=level,
        components=components,
        raw_signals=raw_signals,
        warnings=warnings,
        is_acceptable=is_acceptable,
    )


def report_to_dict(report: ConfidenceReport) -> Dict[str, object]:
    return {
        "score": float(report.score),
        "level": report.level,
        "is_acceptable": bool(report.is_acceptable),
        "components": {k: float(v) for k, v in report.components.items()},
        "raw_signals": {k: float(v) for k, v in report.raw_signals.items()},
        "warnings": list(report.warnings),
    }
