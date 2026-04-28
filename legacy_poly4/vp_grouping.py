"""Vanishing-point aware grouping and outlier rejection in rectified space.

After an initial plumb-line calibration the rectified plane should map every
world-straight line to a 2D line, and most architectural / urban scenes obey a
near-Manhattan structure: lines cluster around two or three orthogonal
vanishing points (VPs). This module exploits that structure to:

1. Cluster the rectified line directions on the projective real line RP^1
   using a small EM-style mixture (no learned components).
2. Score each line by how strongly it agrees with its assigned VP.
3. Flag outlier lines that do not belong to any sufficiently large cluster.

The output is meant to feed a second calibration pass that weights or removes
the flagged lines, sharpening the radial estimate. Nothing in this module
assumes a specific projection family — it operates purely on rectified
directions returned by the current model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from loss import fit_2d_line_pca
from model import FisheyePoly4Model


@dataclass
class VPGroupingConfig:
    max_clusters: int = 3
    min_cluster_size: int = 2
    angle_inlier_deg: float = 6.0
    angle_outlier_deg: float = 14.0
    em_iters: int = 30
    em_tol: float = 1e-5
    min_lines_for_grouping: int = 6
    soft_outlier_weight: float = 0.25
    enforce_orthogonality: bool = False


@dataclass
class VPGroupingResult:
    cluster_angles_rad: List[float] = field(default_factory=list)
    cluster_sizes: List[int] = field(default_factory=list)
    line_cluster: List[int] = field(default_factory=list)  # -1 = outlier
    line_residual_deg: List[float] = field(default_factory=list)
    line_weights: List[float] = field(default_factory=list)
    line_directions_rad: List[float] = field(default_factory=list)
    is_outlier: List[bool] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def _line_direction_in_rectified(
    line_pts: np.ndarray,
    model: FisheyePoly4Model,
    image_shape: Tuple[int, int],
) -> Tuple[float | None, int]:
    """Return the rectified-plane direction angle (mod pi) and valid count."""
    pts = np.asarray(line_pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None, 0
    rect, valid = model.image_points_to_rectified(pts, image_shape=image_shape)
    rect_valid = rect[valid]
    if rect_valid.shape[0] < 4:
        return None, int(rect_valid.shape[0])
    _, direction, _ = fit_2d_line_pca(rect_valid)
    angle = float(np.arctan2(direction[1], direction[0]))
    # Antipodal symmetric: directions in [0, pi).
    angle_mod = angle % np.pi
    return angle_mod, int(rect_valid.shape[0])


def _circular_mean_pi(angles: np.ndarray, weights: np.ndarray) -> float:
    """Mean of angles defined modulo pi using doubled-angle trick."""
    a2 = 2.0 * angles
    w = weights
    s = float(np.sum(w * np.sin(a2)))
    c = float(np.sum(w * np.cos(a2)))
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return float(np.mean(angles))
    return 0.5 * float(np.arctan2(s, c)) % np.pi


def _angular_distance_pi(a: np.ndarray, b: float) -> np.ndarray:
    """Distance on the half-circle [0, pi)."""
    d = (a - b) % np.pi
    return np.minimum(d, np.pi - d)


def _kmeans_pi(
    angles: np.ndarray, k: int, iters: int = 30, tol: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """k-means clustering on angles modulo pi.

    Returns (centers in [0, pi), assignments). Uses k evenly spaced seeds for
    determinism — we do not need random restarts because the search space is
    small (k <= 3) and we always rerun for k=1..max.
    """
    n = angles.shape[0]
    if n == 0 or k <= 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    # Sort to use evenly spaced quantile seeds; this is deterministic and
    # adequate for at most 3 clusters on a 1D circle.
    sorted_a = np.sort(angles)
    if k == 1:
        centers = np.array([_circular_mean_pi(angles, np.ones(n))])
    else:
        idx = np.linspace(0, n - 1, k, dtype=int)
        centers = sorted_a[idx]
    assignments = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.stack(
            [_angular_distance_pi(angles, centers[c]) for c in range(k)], axis=1
        )
        new_assignments = np.argmin(dists, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        new_centers = centers.copy()
        max_shift = 0.0
        for c in range(k):
            mask = assignments == c
            if not np.any(mask):
                continue
            new_centers[c] = _circular_mean_pi(angles[mask], np.ones(int(np.sum(mask))))
            max_shift = max(
                max_shift, float(_angular_distance_pi(np.array([new_centers[c]]), centers[c])[0])
            )
        centers = new_centers
        if max_shift < tol:
            break
    return centers, assignments


def _cluster_quality(
    angles: np.ndarray, centers: np.ndarray, assignments: np.ndarray
) -> float:
    """Mean inverse residual + size balance reward in [0, 1]."""
    if angles.size == 0:
        return 0.0
    res = np.array(
        [_angular_distance_pi(np.array([angles[i]]), centers[assignments[i]])[0] for i in range(angles.size)]
    )
    tightness = float(np.exp(-np.mean(res) / np.deg2rad(8.0)))
    sizes = np.array([int(np.sum(assignments == c)) for c in range(centers.size)])
    if sizes.size == 0 or np.sum(sizes) == 0:
        return 0.0
    balance = 1.0 - float(np.std(sizes) / max(np.mean(sizes), 1e-6))
    balance = float(np.clip(balance, 0.0, 1.0))
    return 0.7 * tightness + 0.3 * balance


def group_lines_by_vp(
    line_groups: List[np.ndarray],
    model: FisheyePoly4Model,
    image_shape: Tuple[int, int],
    cfg: VPGroupingConfig | None = None,
) -> VPGroupingResult:
    """Cluster rectified line directions into vanishing-point groups.

    Lines whose rectified direction does not fall within ``angle_outlier_deg``
    of any cluster center are flagged as outliers. The result is purely
    advisory; consumers decide whether to drop or down-weight them.
    """
    cfg = cfg or VPGroupingConfig()
    n = len(line_groups)
    result = VPGroupingResult(
        line_cluster=[-1] * n,
        line_residual_deg=[float("nan")] * n,
        line_weights=[1.0] * n,
        line_directions_rad=[float("nan")] * n,
        is_outlier=[False] * n,
    )
    if n < cfg.min_lines_for_grouping:
        result.skipped = True
        result.skip_reason = (
            f"only {n} lines provided; need >= {cfg.min_lines_for_grouping} for VP grouping"
        )
        return result

    valid_indices: List[int] = []
    angles: List[float] = []
    for i, line in enumerate(line_groups):
        angle, valid_count = _line_direction_in_rectified(line, model, image_shape)
        if angle is None or valid_count < 4:
            result.is_outlier[i] = True
            result.line_weights[i] = max(0.0, cfg.soft_outlier_weight)
            continue
        result.line_directions_rad[i] = float(angle)
        valid_indices.append(i)
        angles.append(float(angle))

    if len(angles) < cfg.min_lines_for_grouping:
        result.skipped = True
        result.skip_reason = (
            f"only {len(angles)} lines mapped to rectified plane; need >= {cfg.min_lines_for_grouping}"
        )
        return result

    angles_arr = np.asarray(angles, dtype=np.float64)
    best_k = 1
    best_centers = np.array([_circular_mean_pi(angles_arr, np.ones_like(angles_arr))])
    best_assignments = np.zeros(angles_arr.size, dtype=int)
    best_quality = _cluster_quality(angles_arr, best_centers, best_assignments)
    for k in range(2, cfg.max_clusters + 1):
        if angles_arr.size < cfg.min_cluster_size * k:
            break
        centers, assignments = _kmeans_pi(
            angles_arr, k, iters=cfg.em_iters, tol=cfg.em_tol
        )
        # Reject clusters that ended up too small.
        sizes = np.array([int(np.sum(assignments == c)) for c in range(k)])
        if np.any(sizes < cfg.min_cluster_size):
            continue
        quality = _cluster_quality(angles_arr, centers, assignments)
        # Prefer more clusters only when the quality gain is meaningful, to
        # avoid splitting a single dominant direction into noise sub-clusters.
        if quality > best_quality + 0.04:
            best_k = k
            best_centers = centers
            best_assignments = assignments
            best_quality = quality

    result.cluster_angles_rad = [float(a) for a in best_centers.tolist()]
    result.cluster_sizes = [int(np.sum(best_assignments == c)) for c in range(best_k)]

    inlier_thr = np.deg2rad(cfg.angle_inlier_deg)
    outlier_thr = np.deg2rad(cfg.angle_outlier_deg)

    for local_idx, global_idx in enumerate(valid_indices):
        cluster = int(best_assignments[local_idx])
        residual = float(
            _angular_distance_pi(
                np.array([angles_arr[local_idx]]), best_centers[cluster]
            )[0]
        )
        result.line_cluster[global_idx] = cluster
        result.line_residual_deg[global_idx] = float(np.rad2deg(residual))
        if residual <= inlier_thr:
            result.line_weights[global_idx] = 1.0
            result.is_outlier[global_idx] = False
        elif residual >= outlier_thr:
            result.line_weights[global_idx] = max(0.0, cfg.soft_outlier_weight)
            result.is_outlier[global_idx] = True
        else:
            t = (residual - inlier_thr) / max(outlier_thr - inlier_thr, 1e-9)
            w = 1.0 - (1.0 - cfg.soft_outlier_weight) * float(t)
            result.line_weights[global_idx] = float(np.clip(w, 0.0, 1.0))
            result.is_outlier[global_idx] = False
    return result


def report_to_dict(result: VPGroupingResult) -> Dict[str, object]:
    return {
        "skipped": bool(result.skipped),
        "skip_reason": result.skip_reason,
        "num_clusters": len(result.cluster_angles_rad),
        "cluster_angles_deg": [float(np.rad2deg(a)) for a in result.cluster_angles_rad],
        "cluster_sizes": list(result.cluster_sizes),
        "line_cluster": list(result.line_cluster),
        "line_residual_deg": list(result.line_residual_deg),
        "line_weights": list(result.line_weights),
        "line_directions_deg": [
            float(np.rad2deg(a)) if np.isfinite(a) else float("nan")
            for a in result.line_directions_rad
        ],
        "outlier_indices": [
            i for i, flag in enumerate(result.is_outlier) if flag
        ],
    }
