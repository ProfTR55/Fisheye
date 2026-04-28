"""Automatic and manual line constraint utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class AutoLineConfig:
    canny_low: int = 60
    canny_high: int = 160
    blur_ksize: int = 5
    min_contour_points: int = 35
    min_arc_length: float = 80.0
    min_elongation_ratio: float = 2.0
    sample_spacing_px: float = 6.0
    min_samples_per_line: int = 20
    max_samples_per_line: int = 160
    max_lines: int = 40
    processing_max_dim: int = 1600
    min_quality_score: float = 0.22
    local_min_quality_score: float = 0.12
    diversity_grid_size: int = 4
    diversity_per_cell: int = 4
    edge_preference: float = 0.18


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr.copy() if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _segment_support_mask(gray: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(gray, dtype=np.uint8)
    if not hasattr(cv2, "createLineSegmentDetector"):
        return mask
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    out = detector.detect(gray)
    if out is None or out[0] is None:
        return mask
    for seg in out[0].reshape(-1, 4):
        x1, y1, x2, y2 = seg.astype(int).tolist()
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)
    return mask


def _resample_polyline(points: np.ndarray, num_samples: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] <= 1:
        return pts
    seg = pts[1:] - pts[:-1]
    length = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(length)])
    total = cum[-1]
    if total <= 1e-8:
        return np.repeat(pts[:1], num_samples, axis=0)
    t = np.linspace(0.0, total, num_samples)
    return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], axis=1)


def _pca_elongation(points: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    idx = int(np.argmax(vals))
    return float(vals[idx] / vals[1 - idx]), float(np.arctan2(vecs[1, idx], vecs[0, idx]))


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _sample_values(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    pts = np.asarray(points, dtype=np.float64)
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return image[y, x]


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
    return float(1.0 - np.clip(np.percentile(turn, 75) / 0.45, 0.0, 1.0)) if turn.size else 1.0


def _quality_score(points: np.ndarray, arc: float, elongation: float, gray: np.ndarray, grad: np.ndarray) -> float:
    h, w = gray.shape[:2]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)
    rho = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / max(np.hypot(cx, cy), 1e-9)
    radial_span = float(np.max(rho) - np.min(rho)) if rho.size else 0.0
    edge_strength = float(np.median(_sample_values(grad, pts)) / 120.0)
    length_score = float(np.clip(arc / (0.22 * max(h, w)), 0.0, 1.0))
    edge_score = float(np.clip(edge_strength, 0.0, 1.0))
    elongation_score = float(np.clip(np.log1p(elongation) / np.log1p(35.0), 0.0, 1.0))
    radial_score = float(np.clip(radial_span / 0.15, 0.0, 1.0))
    return float(
        np.clip(
            0.30 * length_score
            + 0.24 * edge_score
            + 0.18 * _turn_smoothness_score(pts)
            + 0.16 * radial_score
            + 0.12 * elongation_score,
            0.0,
            1.0,
        )
    )


def _select_diverse(candidates: List[Dict[str, object]], image_shape: Tuple[int, int], cfg: AutoLineConfig) -> List[Dict[str, object]]:
    h, w = image_shape[:2]
    grid = max(1, int(cfg.diversity_grid_size))
    per_cell = max(1, int(cfg.diversity_per_cell))
    counts: Dict[Tuple[int, int], int] = {}
    selected: List[Dict[str, object]] = []
    def ranking_score(cand: Dict[str, object]) -> float:
        return float(cand["score"]) + float(cand.get("edge_bonus", 0.0))

    for cand in sorted(candidates, key=ranking_score, reverse=True):
        if len(selected) >= cfg.max_lines:
            break
        center = np.asarray(cand["center"], dtype=np.float64)
        cell = (
            int(np.clip(np.floor(center[0] / max(w, 1) * grid), 0, grid - 1)),
            int(np.clip(np.floor(center[1] / max(h, 1) * grid), 0, grid - 1)),
        )
        if counts.get(cell, 0) >= per_cell:
            continue
        cand["cell_x"], cand["cell_y"] = cell
        selected.append(cand)
        counts[cell] = counts.get(cell, 0) + 1
    if len(selected) >= min(cfg.max_lines, 4):
        return selected

    # Last-resort backups keep sparse scenes from becoming underconstrained.
    for cand in sorted(candidates, key=ranking_score, reverse=True):
        if len(selected) >= min(cfg.max_lines, 4):
            break
        if cand in selected:
            continue
        selected.append(cand)
    return selected


def extract_line_constraints_auto(
    image_bgr: np.ndarray, cfg: AutoLineConfig | None = None
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    cfg = cfg or AutoLineConfig()
    h0, w0 = image_bgr.shape[:2]
    scale = 1.0
    proc = image_bgr
    if max(h0, w0) > cfg.processing_max_dim:
        scale = cfg.processing_max_dim / max(h0, w0)
        proc = cv2.resize(image_bgr, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
    gray = _to_gray(proc)
    blur = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0.0)
    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    seg_mask = _segment_support_mask(blur)
    support = cv2.bitwise_or(edges, seg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(support, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    grad = _gradient_magnitude(blur)

    candidates: List[Dict[str, object]] = []
    quality_rows: List[List[float]] = []
    for raw_id, cnt in enumerate(contours):
        pts = cnt[:, 0, :].astype(np.float64)
        if pts.shape[0] < cfg.min_contour_points:
            continue
        arc = float(cv2.arcLength(cnt, False))
        if arc < cfg.min_arc_length:
            continue
        if np.linalg.norm(pts[0] - pts[-1]) < 4.0 and cv2.contourArea(cnt) > 200.0:
            continue
        elongation, angle = _pca_elongation(pts)
        if elongation < cfg.min_elongation_ratio:
            continue
        n = int(np.clip(arc / cfg.sample_spacing_px, cfg.min_samples_per_line, cfg.max_samples_per_line))
        sampled = _resample_polyline(pts, n)
        score = _quality_score(sampled, arc, elongation, gray, grad)
        center = np.mean(sampled, axis=0)
        edge_distance = float(
            np.hypot(center[0] - 0.5 * (gray.shape[1] - 1), center[1] - 0.5 * (gray.shape[0] - 1))
            / max(np.hypot(0.5 * (gray.shape[1] - 1), 0.5 * (gray.shape[0] - 1)), 1e-9)
        )
        edge_bonus = float(cfg.edge_preference) * np.clip(edge_distance, 0.0, 1.0)
        quality_rows.append([float(raw_id), score, arc, elongation, float(sampled.shape[0]), edge_distance, edge_bonus])
        if score < cfg.local_min_quality_score:
            continue
        candidates.append(
            {
                "id": raw_id,
                "points": sampled,
                "center": center,
                "angle": angle,
                "arc": arc,
                "score": score,
                "edge_bonus": edge_bonus,
                "primary_quality": float(score >= cfg.min_quality_score),
            }
        )

    if scale != 1.0:
        for cand in candidates:
            cand["points"] = np.asarray(cand["points"], dtype=np.float64) / scale
            cand["center"] = np.asarray(cand["center"], dtype=np.float64) / scale
    selected = _select_diverse(candidates, (h0, w0), cfg)
    lines = [np.asarray(c["points"], dtype=np.float64) for c in selected]
    debug = {
        "gray": gray,
        "edges": edges,
        "segment_mask": seg_mask,
        "support": support,
        "processing_scale": np.array([scale], dtype=np.float32),
        "quality_columns": ["raw_id", "score", "arc", "elongation", "num_points", "edge_distance", "edge_bonus"],
        "quality_rows": np.asarray(quality_rows, dtype=np.float64),
        "selected_rows": np.asarray(
            [
                [
                    float(i),
                    float(c["id"]),
                    float(c["score"]),
                    float(np.asarray(c["points"]).shape[0]),
                    float(np.asarray(c["center"])[0]),
                    float(np.asarray(c["center"])[1]),
                    float(c.get("cell_x", -1)),
                    float(c.get("cell_y", -1)),
                ]
                for i, c in enumerate(selected)
            ],
            dtype=np.float64,
        ),
    }
    return lines, debug


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
        if arr.shape[0] >= 2:
            out.append(arr)
    return out


def load_manual_lines(
    json_path: str | Path,
    sample_spacing_px: float = 8.0,
    min_samples: int = 20,
    max_samples: int = 200,
) -> List[np.ndarray]:
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    lines = []
    for pts in _parse_lines_json(raw):
        length = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
        n = int(np.clip(length / sample_spacing_px, min_samples, max_samples))
        lines.append(_resample_polyline(pts, n))
    return lines


def annotate_lines_interactive(
    image_bgr: np.ndarray,
    output_json_path: str | Path,
) -> List[np.ndarray]:
    """Collect manual curved-line polylines with matplotlib clicks.

    Controls:
    - left click: add points along one world-straight curved support
    - middle click / Enter: finish current line
    - finish with fewer than two points to stop and save
    """
    import matplotlib.pyplot as plt

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lines: List[np.ndarray] = []
    print("Manual annotation mode")
    print("Left click points along one curved line.")
    print("Middle click or Enter finishes the current line.")
    print("Finish by closing/ending a line with fewer than two points.")
    while True:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.set_title(
            f"Line {len(lines) + 1}: left-click points, middle-click/Enter to finish"
        )
        ax.axis("off")
        pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close(fig)
        if len(pts) < 2:
            break
        line = np.asarray(pts, dtype=np.float64)
        lines.append(line)
        print(f"Captured line {len(lines)} with {len(line)} points")

    payload = {"lines": [line.tolist() for line in lines]}
    Path(output_json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(lines)} manual lines to {output_json_path}")
    return lines


def save_lines_json(line_groups: List[np.ndarray], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({"lines": [np.asarray(line, dtype=float).tolist() for line in line_groups]}, indent=2),
        encoding="utf-8",
    )


def draw_line_overlay(
    image_bgr: np.ndarray,
    line_groups: List[np.ndarray],
    radius: int = 2,
    label_indices: bool = False,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colors = [(255, 64, 64), (64, 255, 64), (64, 64, 255), (255, 200, 64), (220, 64, 220), (64, 220, 220)]
    for i, pts in enumerate(line_groups):
        color = colors[i % len(colors)]
        pts_int = np.round(pts).astype(int)
        for x, y in pts_int:
            cv2.circle(overlay, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        if label_indices and pts_int.size:
            center = np.round(np.mean(pts_int, axis=0)).astype(int)
            cv2.putText(overlay, str(i), (int(center[0]) + 6, int(center[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, str(i), (int(center[0]) + 6, int(center[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay
