"""Line/curve constraint extraction utilities.

Automatic mode:
- detect local edge/segment support
- recover elongated edge chains (curved in fisheye domain)
- sample points per chain

Manual mode:
- load polyline annotations from JSON
- optional interactive point-click annotation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AutoLineConfig:
    canny_low: int = 60
    canny_high: int = 160
    blur_ksize: int = 5
    min_contour_points: int = 35
    min_arc_length: float = 80.0
    min_elongation_ratio: float = 2.2
    sample_spacing_px: float = 6.0
    min_samples_per_line: int = 20
    max_samples_per_line: int = 160
    merge_angle_deg: float = 15.0
    merge_distance_px: float = 50.0
    max_lines: int = 40
    processing_max_dim: int = 1600
    min_quality_score: float = 0.28
    local_min_quality_score: float = 0.20
    dark_mask_threshold: int = 14
    max_dark_fraction: float = 0.25
    border_margin_frac: float = 0.025
    max_border_fraction: float = 0.35
    diversity_grid_size: int = 3
    diversity_per_cell: int = 7
    coverage_grid_size: int = 4
    min_lines_per_cell: int = 1
    max_lines_per_cell: int = 4
    min_center_distance_frac: float = 0.10
    monotonic_curvature_min: float = 0.10
    monotonic_curvature_weight: float = 0.18


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr.copy()
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _segment_support_mask(gray: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(gray, dtype=np.uint8)
    if not hasattr(cv2, "createLineSegmentDetector"):
        return mask
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    out = lsd.detect(gray)
    if out is None or out[0] is None:
        return mask
    lines = out[0]
    for seg in lines.reshape(-1, 4):
        x1, y1, x2, y2 = seg.astype(int).tolist()
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)
    return mask


def _resample_polyline(points: np.ndarray, num_samples: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] <= 1:
        return pts
    seg = pts[1:] - pts[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 1e-8:
        return np.repeat(pts[:1], num_samples, axis=0)
    t = np.linspace(0.0, total, num_samples)
    x = np.interp(t, cum, pts[:, 0])
    y = np.interp(t, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


def _pca_elongation(points: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    idx = int(np.argmax(vals))
    major = vecs[:, idx]
    minor_val = vals[1 - idx]
    major_val = vals[idx]
    elongation = float(major_val / minor_val)
    angle = float(np.arctan2(major[1], major[0]))
    return elongation, angle


def _angle_diff_rad(a: float, b: float) -> float:
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


def _polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _sample_image_values(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return image[y, x]


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _turn_smoothness_score(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4:
        return 0.0
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    keep = seg_len > 1e-6
    if int(np.sum(keep)) < 3:
        return 0.0
    angles = np.unwrap(np.arctan2(seg[keep, 1], seg[keep, 0]))
    turn = np.abs(np.diff(angles))
    if turn.size == 0:
        return 1.0
    jitter = float(np.percentile(turn, 75))
    return float(1.0 - np.clip(jitter / 0.45, 0.0, 1.0))


def monotonic_curvature_score(points: np.ndarray) -> float:
    """Score for whether the polyline's curvature has a consistent sign.

    A world-straight line viewed through a radial fisheye lens bends in a
    single direction along its support, so its signed curvature should keep
    its sign over the curve. Truly curved world objects (round table edges,
    foliage, silhouettes) produce sign-flipping curvature. Returns a value in
    [0, 1] where 1.0 means a single dominant sign and 0.0 means perfectly
    mixed signs.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 5:
        return 0.5
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    keep = seg_len > 1e-6
    if int(np.sum(keep)) < 4:
        return 0.5
    seg = seg[keep]
    seg_len = seg_len[keep]
    tangent = seg / seg_len[:, None]
    cross = tangent[:-1, 0] * tangent[1:, 1] - tangent[:-1, 1] * tangent[1:, 0]
    if cross.size == 0:
        return 0.5
    # Drop near-zero turns that carry no directional information.
    significant = np.abs(cross) > 5e-3
    if int(np.sum(significant)) < 3:
        return 0.7
    cross_sig = cross[significant]
    pos = float(np.sum(cross_sig > 0))
    neg = float(np.sum(cross_sig < 0))
    total = pos + neg
    if total == 0:
        return 0.7
    dominant = max(pos, neg) / total
    return float(np.clip(2.0 * (dominant - 0.5), 0.0, 1.0))


def _candidate_quality_metrics(
    points: np.ndarray,
    arc: float,
    elongation: float,
    gray: np.ndarray,
    grad_mag: np.ndarray,
    cfg: AutoLineConfig,
) -> Dict[str, float]:
    h, w = gray.shape[:2]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    rho_max = float(
        np.max(
            np.hypot(
                np.array([0.0, w - 1.0, 0.0, w - 1.0]) - cx,
                np.array([0.0, 0.0, h - 1.0, h - 1.0]) - cy,
            )
        )
        + 1e-9
    )
    rho = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / rho_max
    radial_span = float(np.max(rho) - np.min(rho)) if rho.size else 0.0

    gray_vals = _sample_image_values(gray, pts)
    dark_fraction = float(np.mean(gray_vals <= cfg.dark_mask_threshold))

    margin = cfg.border_margin_frac * max(h, w)
    border = (
        (pts[:, 0] <= margin)
        | (pts[:, 0] >= w - 1 - margin)
        | (pts[:, 1] <= margin)
        | (pts[:, 1] >= h - 1 - margin)
    )
    border_fraction = float(np.mean(border)) if border.size else 0.0

    grad_vals = _sample_image_values(grad_mag, pts)
    edge_strength = float(np.median(grad_vals) / 120.0)

    length_score = float(np.clip(arc / (0.22 * max(h, w)), 0.0, 1.0))
    edge_score = float(np.clip(edge_strength, 0.0, 1.0))
    radial_score = float(np.clip(radial_span / 0.18, 0.0, 1.0))
    elongation_score = float(np.clip(np.log1p(elongation) / np.log1p(35.0), 0.0, 1.0))
    smooth_score = _turn_smoothness_score(pts)
    monotone_score = monotonic_curvature_score(pts)
    chord_ratio = float(np.linalg.norm(pts[0] - pts[-1]) / max(arc, 1e-9))
    continuity_score = float(np.clip(chord_ratio / 0.65, 0.0, 1.0))

    monotone_weight = max(0.0, float(cfg.monotonic_curvature_weight))
    # Rescale the additive component weights so adding the monotonicity term
    # leaves the score in [0, 1] without recalibrating downstream thresholds.
    base_total = 0.24 + 0.20 + 0.18 + 0.16 + 0.12 + 0.10
    rescale = base_total / (base_total + monotone_weight)
    score = (
        0.24 * length_score
        + 0.20 * edge_score
        + 0.18 * smooth_score
        + 0.16 * radial_score
        + 0.12 * elongation_score
        + 0.10 * continuity_score
    ) * rescale + monotone_weight * monotone_score * rescale
    score = score - 0.22 * np.clip(
        dark_fraction / max(cfg.max_dark_fraction, 1e-6), 0.0, 1.4
    ) - 0.18 * np.clip(
        border_fraction / max(cfg.max_border_fraction, 1e-6), 0.0, 1.4
    )
    score = float(np.clip(score, 0.0, 1.0))
    return {
        "score": score,
        "arc": float(arc),
        "elongation": float(elongation),
        "radial_span": radial_span,
        "edge_score": edge_score,
        "smooth_score": float(smooth_score),
        "monotone_score": float(monotone_score),
        "continuity_score": continuity_score,
        "dark_fraction": dark_fraction,
        "border_fraction": border_fraction,
    }


def _select_diverse_candidates(
    candidates: List[Dict[str, object]], image_shape: Tuple[int, int], cfg: AutoLineConfig
) -> List[Dict[str, object]]:
    if not candidates:
        return []
    h, w = image_shape[:2]
    grid = max(1, int(cfg.coverage_grid_size))
    min_per_cell = max(0, int(cfg.min_lines_per_cell))
    max_per_cell = max(min_per_cell, int(cfg.max_lines_per_cell))
    selected: List[Dict[str, object]] = []
    selected_ids: set[int] = set()
    selected_centers: List[np.ndarray] = []
    coverage_counts: Dict[Tuple[int, int], int] = {}
    center_counts: Dict[Tuple[int, int], int] = {}
    ordered = sorted(candidates, key=lambda x: float(x["score"]), reverse=True)
    min_center_distance = max(0.0, float(cfg.min_center_distance_frac)) * max(h, w)

    def candidate_cell(cand: Dict[str, object]) -> Tuple[int, int]:
        center = np.asarray(cand["center"], dtype=np.float64).reshape(2)
        gx = int(np.clip(np.floor(center[0] / max(w, 1) * grid), 0, grid - 1))
        gy = int(np.clip(np.floor(center[1] / max(h, 1) * grid), 0, grid - 1))
        return gx, gy

    def candidate_cells(cand: Dict[str, object]) -> Tuple[Tuple[int, int], ...]:
        pts = np.asarray(cand["points"], dtype=np.float64).reshape(-1, 2)
        gx = np.clip(np.floor(pts[:, 0] / max(w, 1) * grid).astype(int), 0, grid - 1)
        gy = np.clip(np.floor(pts[:, 1] / max(h, 1) * grid).astype(int), 0, grid - 1)
        cells = sorted({(int(x), int(y)) for x, y in zip(gx, gy)})
        return tuple(cells or [candidate_cell(cand)])

    def can_add_candidate(cand: Dict[str, object]) -> bool:
        if len(selected) >= cfg.max_lines:
            return False
        cand_id = int(cand["id"])
        if cand_id in selected_ids:
            return False
        center_cell = candidate_cell(cand)
        if center_counts.get(center_cell, 0) >= max_per_cell:
            return False
        cells = candidate_cells(cand)
        if cells and all(coverage_counts.get(cell, 0) >= max_per_cell for cell in cells):
            return False
        center = np.asarray(cand["center"], dtype=np.float64).reshape(2)
        if min_center_distance > 0.0:
            for selected_center in selected_centers:
                if float(np.linalg.norm(center - selected_center)) < min_center_distance:
                    return False
        return True

    def add_candidate(cand: Dict[str, object], reason: float) -> bool:
        if not can_add_candidate(cand):
            return False
        cand_id = int(cand["id"])
        center_cell = candidate_cell(cand)
        cells = candidate_cells(cand)
        center = np.asarray(cand["center"], dtype=np.float64).reshape(2)
        cand["cell_x"] = float(center_cell[0])
        cand["cell_y"] = float(center_cell[1])
        cand["selection_reason"] = float(reason)
        cand["covered_cell_count"] = float(len(cells))
        selected.append(cand)
        selected_ids.add(cand_id)
        selected_centers.append(center)
        center_counts[center_cell] = center_counts.get(center_cell, 0) + 1
        for cell in cells:
            coverage_counts[cell] = coverage_counts.get(cell, 0) + 1
        return True

    by_cell: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for cand in ordered:
        for cell in candidate_cells(cand):
            by_cell.setdefault(cell, []).append(cand)

    def coverage_value(cand: Dict[str, object]) -> float:
        cells = candidate_cells(cand)
        if not cells:
            return float(cand["score"])
        new_cells = sum(1 for cell in cells if coverage_counts.get(cell, 0) < min_per_cell)
        pressure = np.mean([coverage_counts.get(cell, 0) for cell in cells])
        return float(cand["score"]) + 0.12 * new_cells - 0.06 * float(pressure)

    # First guarantee spatial coverage. Prefer primary-quality candidates per cell;
    # use lower local-quality candidates only to avoid leaving a region empty.
    cells_by_strength = sorted(
        by_cell.keys(), key=lambda cell: float(by_cell[cell][0]["score"]), reverse=True
    )
    for cell in cells_by_strength:
        while coverage_counts.get(cell, 0) < min_per_cell and len(selected) < cfg.max_lines:
            primary = [
                cand
                for cand in by_cell[cell]
                if int(cand["id"]) not in selected_ids
                and float(cand["score"]) >= cfg.min_quality_score
            ]
            pool = primary or [
                cand
                for cand in by_cell[cell]
                if int(cand["id"]) not in selected_ids
                and float(cand["score"]) >= cfg.local_min_quality_score
            ]
            if not pool:
                break
            added = False
            for cand in sorted(pool, key=coverage_value, reverse=True):
                if add_candidate(cand, 1.0 if primary else 2.0):
                    added = True
                    break
            if not added:
                break

    # Then fill remaining capacity with primary candidates that improve coverage
    # instead of blindly taking more lines from already represented areas.
    while len(selected) < cfg.max_lines:
        primary_pool = [
            cand
            for cand in ordered
            if int(cand["id"]) not in selected_ids
            and float(cand["score"]) >= cfg.min_quality_score
            and can_add_candidate(cand)
        ]
        if not primary_pool:
            break
        best = max(primary_pool, key=coverage_value)
        if not add_candidate(best, 0.0):
            break

    # If the primary threshold is too strict for a sparse image, add only enough
    # local-quality backups to keep calibration from becoming underconstrained.
    min_reasonable_lines = min(
        cfg.max_lines,
        max(3, min_per_cell * min(len(by_cell), max(1, grid))),
    )
    if len(selected) < min_reasonable_lines:
        while len(selected) < min_reasonable_lines:
            backup_pool = [
                cand
                for cand in ordered
                if int(cand["id"]) not in selected_ids
                and float(cand["score"]) >= cfg.local_min_quality_score
                and can_add_candidate(cand)
            ]
            if not backup_pool:
                break
            best = max(backup_pool, key=coverage_value)
            if not add_candidate(best, 3.0):
                break
    return selected


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def extract_line_constraints_auto(
    image_bgr: np.ndarray, cfg: AutoLineConfig | None = None
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    """Extract candidate curved-line support points from a fisheye image."""
    cfg = cfg or AutoLineConfig()
    h0, w0 = image_bgr.shape[:2]
    max_dim = max(h0, w0)
    scale = 1.0
    proc_bgr = image_bgr
    if max_dim > cfg.processing_max_dim:
        scale = cfg.processing_max_dim / max_dim
        proc_bgr = cv2.resize(
            image_bgr,
            (int(round(w0 * scale)), int(round(h0 * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = _to_gray(proc_bgr)
    blur = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0.0)
    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    grad_mag = _gradient_magnitude(blur)
    seg_mask = _segment_support_mask(blur)
    support = cv2.bitwise_or(edges, seg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(support, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    raw_candidates: List[Dict[str, object]] = []
    quality_rows: List[List[float]] = []
    raw_id = 0
    for cnt in contours:
        pts = cnt[:, 0, :].astype(np.float64)
        if pts.shape[0] < cfg.min_contour_points:
            continue
        arc = cv2.arcLength(cnt, False)
        if arc < cfg.min_arc_length:
            continue

        first_last = np.linalg.norm(pts[0] - pts[-1])
        if first_last < 4.0 and cv2.contourArea(cnt) > 200.0:
            # Closed loop-like contour is usually not a plumb-line support.
            continue

        elongation, angle = _pca_elongation(pts)
        if elongation < cfg.min_elongation_ratio:
            continue

        center = np.mean(pts, axis=0)
        n_samples = int(np.clip(arc / cfg.sample_spacing_px, cfg.min_samples_per_line, cfg.max_samples_per_line))
        sampled = _resample_polyline(pts, n_samples)
        metrics = _candidate_quality_metrics(
            sampled,
            arc=float(arc),
            elongation=elongation,
            gray=gray,
            grad_mag=grad_mag,
            cfg=cfg,
        )
        monotone_ok = metrics["monotone_score"] >= cfg.monotonic_curvature_min
        in_candidate_pool = (
            metrics["score"] >= cfg.local_min_quality_score
            and metrics["dark_fraction"] <= cfg.max_dark_fraction
            and metrics["border_fraction"] <= cfg.max_border_fraction
            and monotone_ok
        )
        primary_quality = metrics["score"] >= cfg.min_quality_score
        quality_rows.append(
            [
                float(raw_id),
                metrics["score"],
                metrics["arc"],
                metrics["elongation"],
                metrics["radial_span"],
                metrics["edge_score"],
                metrics["smooth_score"],
                metrics["monotone_score"],
                metrics["continuity_score"],
                metrics["dark_fraction"],
                metrics["border_fraction"],
                float(in_candidate_pool),
                float(primary_quality),
                float(monotone_ok),
            ]
        )
        raw_id += 1
        if not in_candidate_pool:
            continue
        raw_candidates.append(
            {
                "id": raw_id - 1,
                "points": sampled,
                "center": center,
                "angle": np.array([angle], dtype=np.float64),
                "arc": np.array([arc], dtype=np.float64),
                "score": metrics["score"],
                "primary_quality": float(primary_quality),
            }
        )

    if not raw_candidates:
        debug = {
            "gray": gray,
            "edges": edges,
            "segment_mask": seg_mask,
            "support": support,
            "processing_scale": np.array([scale], dtype=np.float32),
            "quality_columns": [
                "raw_id",
                "score",
                "arc",
                "elongation",
                "radial_span",
                "edge_score",
                "smooth_score",
                "monotone_score",
                "continuity_score",
                "dark_fraction",
                "border_fraction",
                "pool_kept",
                "primary_quality",
                "monotone_ok",
            ],
            "quality_rows": np.asarray(quality_rows, dtype=np.float64),
        }
        return [], debug

    # Merge nearby and similarly oriented candidates into longer constraints.
    uf = _UnionFind(len(raw_candidates))
    max_angle = np.deg2rad(cfg.merge_angle_deg)
    for i in range(len(raw_candidates)):
        for j in range(i + 1, len(raw_candidates)):
            pi = raw_candidates[i]
            pj = raw_candidates[j]
            d = np.linalg.norm(pi["center"] - pj["center"])
            if d > cfg.merge_distance_px:
                continue
            if _angle_diff_rad(float(pi["angle"][0]), float(pj["angle"][0])) > max_angle:
                continue
            uf.union(i, j)

    groups: Dict[int, List[Dict[str, object]]] = {}
    for idx, cand in enumerate(raw_candidates):
        root = uf.find(idx)
        groups.setdefault(root, []).append(cand)

    merged_candidates: List[Dict[str, object]] = []
    merged_id = 0
    for _, cand_list in groups.items():
        pts_list = [np.asarray(cand["points"], dtype=np.float64) for cand in cand_list]
        pts = np.concatenate(pts_list, axis=0)
        if pts.shape[0] < cfg.min_samples_per_line:
            continue
        scores = np.asarray([float(cand["score"]) for cand in cand_list], dtype=np.float64)
        arcs = np.asarray([float(np.asarray(cand["arc"]).reshape(-1)[0]) for cand in cand_list], dtype=np.float64)
        score = float(np.average(scores, weights=np.maximum(arcs, 1e-6)))
        primary_quality = float(score >= cfg.min_quality_score)
        center = np.mean(pts, axis=0)
        # Deterministic downsampling to keep optimization efficient.
        if pts.shape[0] > cfg.max_samples_per_line:
            take = np.linspace(0, pts.shape[0] - 1, cfg.max_samples_per_line).astype(int)
            pts = pts[take]
        merged_candidates.append(
            {
                "id": merged_id,
                "points": pts,
                "score": score,
                "center": center,
                "num_parts": len(cand_list),
                "primary_quality": primary_quality,
            }
        )
        merged_id += 1

    if scale != 1.0:
        for cand in merged_candidates:
            cand["points"] = np.asarray(cand["points"], dtype=np.float64) / scale
            cand["center"] = np.asarray(cand["center"], dtype=np.float64) / scale

    selected_candidates = _select_diverse_candidates(
        merged_candidates, image_shape=(h0, w0), cfg=cfg
    )
    selected_candidates.sort(key=lambda x: float(x["score"]), reverse=True)
    merged = [np.asarray(cand["points"], dtype=np.float64) for cand in selected_candidates]
    selected_rows = np.asarray(
        [
            [
                float(rank),
                float(cand["id"]),
                float(cand["score"]),
                float(np.asarray(cand["points"]).shape[0]),
                float(np.asarray(cand["center"])[0]),
                float(np.asarray(cand["center"])[1]),
                float(cand["num_parts"]),
                float(cand.get("cell_x", -1.0)),
                float(cand.get("cell_y", -1.0)),
                float(cand.get("selection_reason", -1.0)),
                float(cand.get("covered_cell_count", -1.0)),
            ]
            for rank, cand in enumerate(selected_candidates)
        ],
        dtype=np.float64,
    )
    debug = {
        "gray": gray,
        "edges": edges,
        "segment_mask": seg_mask,
        "support": support,
        "processing_scale": np.array([scale], dtype=np.float32),
        "quality_columns": [
            "raw_id",
            "score",
            "arc",
            "elongation",
            "radial_span",
            "edge_score",
            "smooth_score",
            "monotone_score",
            "continuity_score",
            "dark_fraction",
            "border_fraction",
            "pool_kept",
            "primary_quality",
            "monotone_ok",
        ],
        "quality_rows": np.asarray(quality_rows, dtype=np.float64),
        "selected_columns": [
            "rank",
            "merged_id",
            "score",
            "num_points",
            "center_x",
            "center_y",
            "num_parts",
            "cell_x",
            "cell_y",
            "selection_reason",
            "covered_cell_count",
        ],
        "selected_rows": selected_rows,
    }
    return merged, debug


def _parse_lines_json(raw: object) -> List[np.ndarray]:
    if isinstance(raw, dict) and "lines" in raw:
        raw = raw["lines"]
    if not isinstance(raw, list):
        raise ValueError("Annotation JSON must be a list or an object containing key 'lines'.")
    out: List[np.ndarray] = []
    for idx, line in enumerate(raw):
        arr = np.asarray(line, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Line {idx} must be Nx2 points.")
        if arr.shape[0] < 2:
            continue
        out.append(arr)
    return out


def load_manual_lines(
    json_path: str | Path,
    sample_spacing_px: float = 8.0,
    min_samples: int = 20,
    max_samples: int = 200,
) -> List[np.ndarray]:
    path = Path(json_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    polylines = _parse_lines_json(raw)
    sampled: List[np.ndarray] = []
    for pts in polylines:
        seg_len = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        total = float(np.sum(seg_len))
        n = int(np.clip(total / sample_spacing_px, min_samples, max_samples))
        sampled.append(_resample_polyline(pts, n))
    return sampled


def annotate_lines_interactive(
    image_bgr: np.ndarray, output_json_path: str | Path
) -> List[np.ndarray]:
    """Manual annotation via click points in matplotlib.

    Flow:
    - Left click points for one line, middle click to finish that line.
    - Repeat until you add no points and middle click.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lines: List[np.ndarray] = []
    print("Manual annotation mode")
    print("Left click points along one curved line, middle click to finish each line.")
    print("Finish by middle-clicking without adding points.")
    while True:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(image_rgb)
        ax.set_title("Add one line: left-click points, middle-click to finish line")
        ax.axis("off")
        pts = plt.ginput(n=-1, timeout=0)
        plt.close(fig)
        if len(pts) < 2:
            break
        lines.append(np.asarray(pts, dtype=np.float64))
        print(f"Captured line {len(lines)} with {len(pts)} points")

    payload = {"lines": [line.tolist() for line in lines]}
    Path(output_json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(lines)} lines to {output_json_path}")
    return lines


def draw_line_overlay(
    image_bgr: np.ndarray,
    line_groups: List[np.ndarray],
    radius: int = 2,
    label_indices: bool = False,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colors = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 64, 255),
        (255, 200, 64),
        (220, 64, 220),
        (64, 220, 220),
    ]
    for i, pts in enumerate(line_groups):
        color = colors[i % len(colors)]
        pts_int = np.round(pts).astype(int)
        for p in pts_int:
            cv2.circle(overlay, (int(p[0]), int(p[1])), radius, color, -1, cv2.LINE_AA)
        if label_indices and pts_int.size:
            center = np.round(np.mean(pts_int, axis=0)).astype(int)
            label = str(i)
            org = (int(center[0]) + 6, int(center[1]) - 6)
            cv2.putText(
                overlay,
                label,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return overlay


def save_lines_json(line_groups: List[np.ndarray], path: str | Path) -> None:
    payload = {"lines": [np.asarray(line).tolist() for line in line_groups]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
