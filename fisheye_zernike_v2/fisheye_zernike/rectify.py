"""Rectification map generation and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from .model import RadialFisheyeModel


@dataclass
class RectifyConfig:
    output_fov_deg: float = 100.0
    output_width: int | None = None
    output_height: int | None = None
    crop_mode: str = "auto"
    auto_crop_valid_ratio: float = 0.995
    auto_crop_min_scale: float = 0.55
    auto_crop_step: float = 0.01
    interpolation: int = cv2.INTER_CUBIC


def _output_shape_from_input(input_shape: Tuple[int, int], cfg: RectifyConfig) -> Tuple[int, int]:
    h_in, w_in = input_shape[:2]
    w_out = int(cfg.output_width or w_in)
    h_out = int(cfg.output_height) if cfg.output_height is not None else int(round(h_in * (w_out / w_in)))
    return h_out, w_out


def build_rectification_map(
    model: RadialFisheyeModel,
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
    rect_pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    uv, inside = model.rectified_points_to_image(rect_pts, input_shape)
    map_x = uv[:, 0].reshape(h_out, w_out)
    map_y = uv[:, 1].reshape(h_out, w_out)
    theta = np.arctan(np.hypot(xx, yy))
    theta_edge = float(model.g(np.array([1.0]))[0])
    valid = inside.reshape(h_out, w_out) & (theta <= theta_edge)
    meta = {
        "f_out": float(f_out),
        "cx_out": float(cx_out),
        "cy_out": float(cy_out),
        "theta_edge_rad": theta_edge,
        "output_width": float(w_out),
        "output_height": float(h_out),
    }
    return map_x.astype(np.float32), map_y.astype(np.float32), valid, meta


def _auto_crop_centered(
    valid_mask: np.ndarray,
    min_valid_ratio: float,
    min_scale: float,
    scale_step: float,
) -> Tuple[int, int, int, int]:
    h, w = valid_mask.shape[:2]
    cy = h // 2
    cx = w // 2
    for scale in np.arange(1.0, min_scale - 1e-9, -scale_step):
        hh = max(8, int(round(h * scale)))
        ww = max(8, int(round(w * scale)))
        y0 = max(0, cy - hh // 2)
        y1 = min(h, y0 + hh)
        x0 = max(0, cx - ww // 2)
        x1 = min(w, x0 + ww)
        region = valid_mask[y0:y1, x0:x1]
        if region.size and float(np.mean(region)) >= min_valid_ratio:
            return y0, y1, x0, x1
    return 0, h, 0, w


def render_rectified(
    image_bgr: np.ndarray,
    model: RadialFisheyeModel,
    cfg: RectifyConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or RectifyConfig()
    map_x, map_y, valid_mask, meta = build_rectification_map(model, image_bgr.shape[:2], cfg)
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
            valid_mask, cfg.auto_crop_valid_ratio, cfg.auto_crop_min_scale, cfg.auto_crop_step
        )
    y0, y1, x0, x1 = crop_box
    return {
        "rectified": rectified,
        "rectified_crop": rectified[y0:y1, x0:x1],
        "valid_mask": valid_mask.astype(np.uint8) * 255,
        "valid_mask_crop": valid_mask[y0:y1, x0:x1].astype(np.uint8) * 255,
        "map_x": map_x,
        "map_y": map_y,
        "map_x_crop": map_x[y0:y1, x0:x1],
        "map_y_crop": map_y[y0:y1, x0:x1],
        "crop_box": crop_box,
        "meta": meta,
    }
