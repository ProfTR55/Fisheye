"""Rectification map generation and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from model import FisheyePoly4Model


@dataclass
class RectifyConfig:
    output_fov_deg: float = 100.0
    output_width: int | None = None
    output_height: int | None = None
    crop_mode: str = "none"  # "none" or "auto"
    auto_crop_valid_ratio: float = 0.995
    auto_crop_min_scale: float = 0.55
    auto_crop_step: float = 0.01
    interpolation: int = cv2.INTER_CUBIC


def _output_shape_from_input(
    input_shape: Tuple[int, int], cfg: RectifyConfig
) -> Tuple[int, int]:
    h_in, w_in = input_shape[:2]
    w_out = cfg.output_width or w_in
    if cfg.output_height is not None:
        h_out = cfg.output_height
    else:
        h_out = int(round(h_in * (w_out / w_in)))
    return int(h_out), int(w_out)


def build_rectification_map(
    model: FisheyePoly4Model,
    input_shape: Tuple[int, int],
    cfg: RectifyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    h_out, w_out = _output_shape_from_input(input_shape, cfg)
    f_out = 0.5 * w_out / np.tan(0.5 * np.deg2rad(cfg.output_fov_deg))
    cx_out = 0.5 * (w_out - 1)
    cy_out = 0.5 * (h_out - 1)

    xs = (np.arange(w_out, dtype=np.float64) - cx_out) / max(f_out, 1e-9)
    ys = (np.arange(h_out, dtype=np.float64) - cy_out) / max(f_out, 1e-9)
    xx, yy = np.meshgrid(xs, ys)
    r = np.hypot(xx, yy)
    theta = np.arctan(r)
    phi = np.arctan2(yy, xx)

    theta_edge = float(model.g(np.array([1.0]))[0])
    r_hat = model.theta_to_rhat(theta.reshape(-1)).reshape(theta.shape)
    rho = r_hat * model.rho_max(input_shape)
    xu = rho * np.cos(phi)
    yu = rho * np.sin(phi)
    xd, yd = model.distort_tangential_components(xu, yu, image_shape=input_shape)
    map_x = model.cx + model.sx * xd
    map_y = model.cy + model.sy * yd

    h_in, w_in = input_shape[:2]
    inside = (map_x >= 0.0) & (map_x <= w_in - 1.0) & (map_y >= 0.0) & (map_y <= h_in - 1.0)
    valid = inside & (theta <= theta_edge)
    return map_x.astype(np.float32), map_y.astype(np.float32), valid, {
        "f_out": float(f_out),
        "theta_edge_rad": theta_edge,
        "output_width": float(w_out),
        "output_height": float(h_out),
    }


def _auto_crop_centered(
    valid_mask: np.ndarray,
    min_valid_ratio: float = 0.995,
    min_scale: float = 0.55,
    scale_step: float = 0.01,
) -> Tuple[int, int, int, int]:
    h, w = valid_mask.shape[:2]
    cy = h // 2
    cx = w // 2
    best = (0, h, 0, w)
    scales = np.arange(1.0, min_scale - 1e-9, -scale_step)
    for s in scales:
        hh = max(8, int(round(h * s)))
        ww = max(8, int(round(w * s)))
        y0 = max(0, cy - hh // 2)
        y1 = min(h, y0 + hh)
        x0 = max(0, cx - ww // 2)
        x1 = min(w, x0 + ww)
        region = valid_mask[y0:y1, x0:x1]
        if region.size == 0:
            continue
        ratio = float(np.mean(region))
        if ratio >= min_valid_ratio:
            return y0, y1, x0, x1
    return best


def render_rectified(
    image_bgr: np.ndarray,
    model: FisheyePoly4Model,
    cfg: RectifyConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or RectifyConfig()
    map_x, map_y, valid_mask, meta = build_rectification_map(
        model=model, input_shape=image_bgr.shape[:2], cfg=cfg
    )
    rectified = cv2.remap(
        image_bgr,
        map_x,
        map_y,
        interpolation=cfg.interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rectified[~valid_mask] = 0

    crop_box = (0, rectified.shape[0], 0, rectified.shape[1])
    if cfg.crop_mode.lower() == "auto":
        crop_box = _auto_crop_centered(
            valid_mask,
            min_valid_ratio=cfg.auto_crop_valid_ratio,
            min_scale=cfg.auto_crop_min_scale,
            scale_step=cfg.auto_crop_step,
        )
    y0, y1, x0, x1 = crop_box
    rectified_crop = rectified[y0:y1, x0:x1]
    valid_crop = valid_mask[y0:y1, x0:x1]
    map_x_crop = map_x[y0:y1, x0:x1]
    map_y_crop = map_y[y0:y1, x0:x1]

    out: Dict[str, object] = {
        "rectified": rectified,
        "rectified_crop": rectified_crop,
        "valid_mask": valid_mask.astype(np.uint8) * 255,
        "valid_mask_crop": valid_crop.astype(np.uint8) * 255,
        "map_x": map_x,
        "map_y": map_y,
        "map_x_crop": map_x_crop,
        "map_y_crop": map_y_crop,
        "crop_box": crop_box,
        "meta": meta,
    }
    return out
