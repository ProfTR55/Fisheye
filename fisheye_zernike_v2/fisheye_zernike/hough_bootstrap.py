"""Rectified-space Hough bootstrap refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .loss import LossConfig, evaluate_objective
from .model import RadialFisheyeModel
from .optimize import CalibrationResult, OptimizeConfig, calibrate_from_lines
from .rectify import RectifyConfig, render_rectified


@dataclass
class HoughBootstrapConfig:
    canny_low: int = 60
    canny_high: int = 160
    threshold: int = 55
    min_line_length: int = 45
    max_line_gap: int = 12
    sample_spacing_px: float = 8.0
    min_samples_per_line: int = 12
    max_samples_per_line: int = 90
    max_new_lines: int = 30
    min_valid_fraction: float = 0.75
    min_improvement: float = 0.01


@dataclass
class HoughBootstrapReport:
    attempted: bool = False
    accepted: bool = False
    reason: str = ""
    candidates_detected: int = 0
    candidates_mapped: int = 0
    base_metric: float | None = None
    refined_metric: float | None = None
    new_line_lengths: List[int] = field(default_factory=list)


def _rectified_segment_to_model_points(
    segment: np.ndarray,
    meta: Dict[str, float],
    model: RadialFisheyeModel,
    image_shape: Tuple[int, int],
    cfg: HoughBootstrapConfig,
) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(v) for v in segment]
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < cfg.min_line_length:
        return None
    n = int(np.clip(length / cfg.sample_spacing_px, cfg.min_samples_per_line, cfg.max_samples_per_line))
    t = np.linspace(0.0, 1.0, n)
    xs = x1 + (x2 - x1) * t
    ys = y1 + (y2 - y1) * t
    f = max(float(meta["f_out"]), 1e-9)
    rect = np.stack([(xs - float(meta["cx_out"])) / f, (ys - float(meta["cy_out"])) / f], axis=1)
    uv, valid = model.rectified_points_to_image(rect, image_shape)
    if float(np.mean(valid)) < cfg.min_valid_fraction:
        return None
    uv = uv[valid]
    return uv if uv.shape[0] >= cfg.min_samples_per_line else None


def extract_hough_lines_from_rectified(
    rectified_bgr: np.ndarray,
    meta: Dict[str, float],
    model: RadialFisheyeModel,
    image_shape: Tuple[int, int],
    cfg: HoughBootstrapConfig | None = None,
) -> Tuple[List[np.ndarray], HoughBootstrapReport]:
    cfg = cfg or HoughBootstrapConfig()
    report = HoughBootstrapReport(attempted=True)
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY) if rectified_bgr.ndim == 3 else rectified_bgr
    blur = cv2.GaussianBlur(gray, (5, 5), 0.0)
    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    hough = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=cfg.threshold,
        minLineLength=cfg.min_line_length,
        maxLineGap=cfg.max_line_gap,
    )
    if hough is None:
        report.reason = "no Hough segments detected"
        return [], report
    segments = hough.reshape(-1, 4)
    report.candidates_detected = int(segments.shape[0])
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    order = np.argsort(lengths)[::-1]
    lines: List[np.ndarray] = []
    for idx in order[: cfg.max_new_lines * 3]:
        mapped = _rectified_segment_to_model_points(segments[idx], meta, model, image_shape, cfg)
        if mapped is None:
            continue
        lines.append(mapped)
        report.new_line_lengths.append(int(mapped.shape[0]))
        if len(lines) >= cfg.max_new_lines:
            break
    report.candidates_mapped = len(lines)
    if not lines:
        report.reason = "Hough segments did not map to valid input constraints"
    return lines, report


def run_hough_bootstrap(
    image_bgr: np.ndarray,
    base_result: CalibrationResult,
    train_line_groups: List[np.ndarray],
    validation_line_groups: List[np.ndarray],
    model_family: str,
    optimize_cfg: OptimizeConfig,
    loss_cfg: LossConfig,
    rectify_cfg: RectifyConfig,
    hough_cfg: HoughBootstrapConfig | None = None,
) -> Tuple[CalibrationResult, HoughBootstrapReport, List[np.ndarray]]:
    hough_cfg = hough_cfg or HoughBootstrapConfig()
    render = render_rectified(image_bgr, base_result.model, rectify_cfg)
    new_lines, report = extract_hough_lines_from_rectified(
        render["rectified"], render["meta"], base_result.model, image_bgr.shape[:2], hough_cfg
    )
    if not new_lines:
        return base_result, report, train_line_groups

    combined = list(train_line_groups) + new_lines
    refined = calibrate_from_lines(
        image_shape=image_bgr.shape[:2],
        line_groups=combined,
        model_family=model_family,
        optimize_cfg=optimize_cfg,
        loss_cfg=loss_cfg,
    )
    if validation_line_groups:
        base_metric = evaluate_objective(base_result.model, image_bgr.shape[:2], validation_line_groups, loss_cfg)["raw_line_rmse"]
        refined_metric = evaluate_objective(refined.model, image_bgr.shape[:2], validation_line_groups, loss_cfg)["raw_line_rmse"]
    else:
        base_metric = base_result.final_metrics["raw_line_rmse"]
        refined_metric = evaluate_objective(refined.model, image_bgr.shape[:2], train_line_groups, loss_cfg)["raw_line_rmse"]
    report.base_metric = float(base_metric)
    report.refined_metric = float(refined_metric)
    if refined_metric <= base_metric * (1.0 - hough_cfg.min_improvement):
        report.accepted = True
        report.reason = "refined metric improved"
        return refined, report, combined
    report.accepted = False
    report.reason = "refined metric did not improve enough; kept base model"
    return base_result, report, train_line_groups
