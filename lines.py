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
    seg_mask = _segment_support_mask(blur)
    support = cv2.bitwise_or(edges, seg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(support, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    raw_candidates: List[Dict[str, np.ndarray]] = []
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
        raw_candidates.append(
            {
                "points": sampled,
                "center": center,
                "angle": np.array([angle], dtype=np.float64),
                "arc": np.array([arc], dtype=np.float64),
            }
        )

    if not raw_candidates:
        debug = {
            "gray": gray,
            "edges": edges,
            "segment_mask": seg_mask,
            "support": support,
            "processing_scale": np.array([scale], dtype=np.float32),
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

    groups: Dict[int, List[np.ndarray]] = {}
    for idx, cand in enumerate(raw_candidates):
        root = uf.find(idx)
        groups.setdefault(root, []).append(cand["points"])

    merged: List[np.ndarray] = []
    for _, pts_list in groups.items():
        pts = np.concatenate(pts_list, axis=0)
        if pts.shape[0] < cfg.min_samples_per_line:
            continue
        # Deterministic downsampling to keep optimization efficient.
        if pts.shape[0] > cfg.max_samples_per_line:
            take = np.linspace(0, pts.shape[0] - 1, cfg.max_samples_per_line).astype(int)
            pts = pts[take]
        merged.append(pts)

    if scale != 1.0:
        merged = [pts / scale for pts in merged]

    merged.sort(key=lambda x: x.shape[0], reverse=True)
    merged = merged[: cfg.max_lines]
    debug = {
        "gray": gray,
        "edges": edges,
        "segment_mask": seg_mask,
        "support": support,
        "processing_scale": np.array([scale], dtype=np.float32),
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
    image_bgr: np.ndarray, line_groups: List[np.ndarray], radius: int = 2
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
    return overlay


def save_lines_json(line_groups: List[np.ndarray], path: str | Path) -> None:
    payload = {"lines": [np.asarray(line).tolist() for line in line_groups]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
