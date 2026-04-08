import argparse
import os

import numpy as np  # pyright: ignore[reportMissingImports]
from PIL import Image  # pyright: ignore[reportMissingImports]


def _base_radius_norm_from_theta(theta, theta_max, model, hybrid_blend):
    tmax = max(theta_max, 1e-6)
    t = np.clip(theta, 0.0, tmax)
    m = model.lower()

    if m == "equidistant":
        return t / tmax
    if m == "equisolid":
        return np.sin(0.5 * t) / max(np.sin(0.5 * tmax), 1e-6)
    if m == "stereographic":
        return np.tan(0.5 * t) / max(np.tan(0.5 * tmax), 1e-6)
    if m == "orthographic":
        return np.sin(t) / max(np.sin(tmax), 1e-6)
    if m == "hybrid":
        w = float(np.clip(hybrid_blend, 0.0, 1.0))
        r_eqd = t / tmax
        r_eqs = np.sin(0.5 * t) / max(np.sin(0.5 * tmax), 1e-6)
        return (1.0 - w) * r_eqd + w * r_eqs
    raise ValueError(
        "projection_model invalid: equidistant/equisolid/stereographic/orthographic/hybrid"
    )


def _apply_radial_correction(r_norm, radial_k2):
    # Tek parametreli (k2) düzeltme: r' = r * (1 + k2*r^2), sonra [0,1] normalize.
    r = np.clip(r_norm, 0.0, 1.0)
    k2 = float(radial_k2)
    denom = max(1.0 + k2, 1e-6)
    out = r * (1.0 + k2 * r * r) / denom
    return np.clip(out, 0.0, 1.0)


def sample_bilinear(img, x, y):
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    ia = img[y0, x0]
    ib = img[y0, x1]
    ic = img[y1, x0]
    id_ = img[y1, x1]

    if img.ndim == 2:
        return ia * wa + ib * wb + ic * wc + id_ * wd
    return (
        ia * wa[..., None]
        + ib * wb[..., None]
        + ic * wc[..., None]
        + id_ * wd[..., None]
    )


def _active_bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _crop_square_around_active(img, threshold=8, pad_ratio=1.12):
    gray = img.mean(axis=2)
    mask = gray > threshold
    bbox = _active_bbox(mask)
    if bbox is None:
        return img

    x0, y0, x1, y1 = bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = int(max(x1 - x0 + 1, y1 - y0 + 1) * pad_ratio)
    side = max(side, 32)

    half = side * 0.5
    sx0 = int(np.floor(cx - half))
    sy0 = int(np.floor(cy - half))
    sx1 = sx0 + side
    sy1 = sy0 + side

    h, w = img.shape[:2]
    cx0 = max(sx0, 0)
    cy0 = max(sy0, 0)
    cx1 = min(sx1, w)
    cy1 = min(sy1, h)
    crop = img[cy0:cy1, cx0:cx1]

    if crop.shape[0] == side and crop.shape[1] == side:
        return crop

    out = np.zeros((side, side, 3), dtype=img.dtype)
    oy = max(0, -sy0)
    ox = max(0, -sx0)
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def select_single_fisheye_lens(img, lens_mode="auto", threshold=8, pad_ratio=1.12):
    h, w = img.shape[:2]
    mode = lens_mode.lower()
    gray = img.mean(axis=2)
    mask = gray > threshold

    if mode == "auto":
        if w >= 1.7 * h:
            left_score = mask[:, : w // 2].sum()
            right_score = mask[:, w // 2:].sum()
            mode = "left" if left_score >= right_score else "right"
        elif h >= 1.7 * w:
            top_score = mask[: h // 2, :].sum()
            bottom_score = mask[h // 2:, :].sum()
            mode = "top" if top_score >= bottom_score else "bottom"
        else:
            mode = "full"

    if mode == "full":
        return _crop_square_around_active(img, threshold, pad_ratio), mode
    if mode == "left":
        return _crop_square_around_active(img[:, : w // 2], threshold, pad_ratio), mode
    if mode == "right":
        return _crop_square_around_active(img[:, w // 2:], threshold, pad_ratio), mode
    if mode == "top":
        return _crop_square_around_active(img[: h // 2, :], threshold, pad_ratio), mode
    if mode == "bottom":
        return _crop_square_around_active(img[h // 2:, :], threshold, pad_ratio), mode
    raise ValueError("lens mode invalid: auto/full/left/right/top/bottom")


def rotate_rays(x, y, z, yaw_deg, pitch_deg):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    x1 = cy * x + sy * z
    y1 = y
    z1 = -sy * x + cy * z

    x2 = x1
    y2 = cp * y1 - sp * z1
    z2 = sp * y1 + cp * z1
    return x2, y2, z2


def inverse_rotate_rays(x, y, z, yaw_deg, pitch_deg):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    x1 = x
    y1 = cp * y + sp * z
    z1 = -sp * y + cp * z

    x2 = cy * x1 - sy * z1
    y2 = y1
    z2 = sy * x1 + cy * z1
    return x2, y2, z2


def build_rectilinear_patch_rays(w, h, fov_x_deg, fov_y_deg):
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)
    fx = (w * 0.5) / np.tan(fov_x * 0.5)
    fy = (h * 0.5) / np.tan(fov_y * 0.5)

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    norm = np.sqrt(x * x + y * y + z * z)
    xr = x / norm
    yr = y / norm
    zr = z / norm

    wx = 1.0 - np.abs((u - cx) / max(cx, 1.0))
    wy = 1.0 - np.abs((v - cy) / max(cy, 1.0))
    weight = np.clip(wx, 0.0, 1.0) * np.clip(wy, 0.0, 1.0)
    weight = weight.astype(np.float32) ** 1.5
    return xr, yr, zr, weight


def _texture_confidence_map(img):
    gray = img.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = 0.5 * np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = 0.5 * np.abs(gray[2:, :] - gray[:-2, :])
    g = np.sqrt(gx * gx + gy * gy)
    p95 = float(np.percentile(g, 95.0))
    if p95 < 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip(g / p95, 0.0, 1.0).astype(np.float32)


def sample_fisheye_from_world_rays(
    fish_img,
    texture_map,
    x,
    y,
    z,
    fish_fov_deg,
    projection_model="equidistant",
    hybrid_blend=0.5,
    radial_k2=0.0,
    center_dx=0.0,
    center_dy=0.0,
    radius_scale=1.0,
    confidence_gamma=1.0,
):
    hs, ws = fish_img.shape[:2]
    base_radius = min(hs, ws) * 0.5
    cx = ws * 0.5 + center_dx * base_radius
    cy = hs * 0.5 + center_dy * base_radius
    radius = base_radius * max(radius_scale, 1e-6)
    theta_max = np.deg2rad(fish_fov_deg * 0.5)

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    alpha = np.arctan2(y, x)
    r_base = _base_radius_norm_from_theta(theta, theta_max, projection_model, hybrid_blend)
    r_norm = _apply_radial_correction(r_base, radial_k2)
    r = r_norm * radius
    xs = cx + r * np.cos(alpha)
    ys = cy + r * np.sin(alpha)

    valid = theta <= theta_max
    valid &= (xs >= 0) & (xs < ws - 1) & (ys >= 0) & (ys < hs - 1)

    patch = np.zeros((*x.shape, 3), dtype=np.uint8)
    conf = np.zeros(x.shape, dtype=np.float32)
    if np.any(valid):
        patch[valid] = sample_bilinear(fish_img, xs[valid], ys[valid]).astype(np.uint8)
        tex = sample_bilinear(texture_map, xs[valid], ys[valid]).astype(np.float32)
        theta_norm = np.clip(theta[valid] / max(theta_max, 1e-6), 0.0, 1.0)
        c_angle = np.cos(0.5 * np.pi * theta_norm) ** 2
        c_radial = np.clip(1.0 - (r[valid] / max(radius, 1e-6)) ** 2, 0.0, 1.0)
        c_texture = 0.25 + 0.75 * np.clip(tex, 0.0, 1.0)
        c = c_angle * c_radial * c_texture
        if confidence_gamma != 1.0:
            c = c ** max(confidence_gamma, 1e-4)
        conf[valid] = c.astype(np.float32)
    return patch, valid, conf


def _splat_bilinear(accum, wsum, x, y, values, w):
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    h, w_canvas = wsum.shape

    def add(xx, yy, ww):
        inside = (xx >= 0) & (xx < w_canvas) & (yy >= 0) & (yy < h) & (ww > 1e-8)
        if not np.any(inside):
            return
        xxi = xx[inside]
        yyi = yy[inside]
        wwi = ww[inside]
        vals = values[inside]
        np.add.at(wsum, (yyi, xxi), wwi)
        for c in range(accum.shape[2]):
            np.add.at(accum[:, :, c], (yyi, xxi), vals[:, c] * wwi)

    fx = x - x0
    fy = y - y0
    add(x0, y0, w * (1.0 - fx) * (1.0 - fy))
    add(x1, y0, w * fx * (1.0 - fy))
    add(x0, y1, w * (1.0 - fx) * fy)
    add(x1, y1, w * fx * fy)


def warp_patch_to_canvas(
    patch,
    valid,
    yaw,
    pitch,
    local_weight,
    out_w,
    out_h,
    yaw_min,
    yaw_max,
    pitch_min,
    pitch_max,
):
    xc = (yaw - yaw_min) / max(yaw_max - yaw_min, 1e-6) * (out_w - 1)
    yc = (pitch - pitch_min) / max(pitch_max - pitch_min, 1e-6) * (out_h - 1)

    mask = valid.copy()
    mask &= (yaw >= yaw_min) & (yaw <= yaw_max)
    mask &= (pitch >= pitch_min) & (pitch <= pitch_max)

    accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, out_w), dtype=np.float32)
    ww = local_weight * mask.astype(np.float32)

    if np.any(mask):
        _splat_bilinear(
            accum=accum,
            wsum=wsum,
            x=xc[mask].astype(np.float32),
            y=yc[mask].astype(np.float32),
            values=patch[mask].astype(np.float32),
            w=ww[mask].astype(np.float32),
        )

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ok = wsum > 1e-8
    out[ok] = (accum[ok] / wsum[ok, None]).astype(np.uint8)
    return out, wsum


def inverse_warp_patch_to_canvas(
    patch,
    patch_weight,
    patch_yaw_deg,
    patch_pitch_deg,
    patch_fov_x_deg,
    patch_fov_y_deg,
    out_w,
    out_h,
    yaw_min,
    yaw_max,
    pitch_min,
    pitch_max,
    roi_margin_deg=3.0,
):
    patch_h, patch_w = patch.shape[:2]
    cx = (patch_w - 1) * 0.5
    cy = (patch_h - 1) * 0.5

    fov_x = np.deg2rad(patch_fov_x_deg)
    fov_y = np.deg2rad(patch_fov_y_deg)
    fx = (patch_w * 0.5) / np.tan(fov_x * 0.5)
    fy = (patch_h * 0.5) / np.tan(fov_y * 0.5)

    yaw_lo = max(yaw_min, patch_yaw_deg - 0.5 * patch_fov_x_deg - roi_margin_deg)
    yaw_hi = min(yaw_max, patch_yaw_deg + 0.5 * patch_fov_x_deg + roi_margin_deg)
    pitch_lo = max(pitch_min, patch_pitch_deg - 0.5 * patch_fov_y_deg - roi_margin_deg)
    pitch_hi = min(pitch_max, patch_pitch_deg + 0.5 * patch_fov_y_deg + roi_margin_deg)

    x0 = int(np.floor((yaw_lo - yaw_min) / max(yaw_max - yaw_min, 1e-6) * (out_w - 1)))
    x1 = int(np.ceil((yaw_hi - yaw_min) / max(yaw_max - yaw_min, 1e-6) * (out_w - 1))) + 1
    y0 = int(np.floor((pitch_lo - pitch_min) / max(pitch_max - pitch_min, 1e-6) * (out_h - 1)))
    y1 = int(np.ceil((pitch_hi - pitch_min) / max(pitch_max - pitch_min, 1e-6) * (out_h - 1))) + 1

    x0 = max(0, min(x0, out_w - 1))
    x1 = max(x0 + 1, min(x1, out_w))
    y0 = max(0, min(y0, out_h - 1))
    y1 = max(y0 + 1, min(y1, out_h))

    uu = np.arange(x0, x1, dtype=np.float32)
    vv = np.arange(y0, y1, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(uu, vv)

    yaw = yaw_min + (u_grid / max(out_w - 1, 1)) * (yaw_max - yaw_min)
    pitch = pitch_min + (v_grid / max(out_h - 1, 1)) * (pitch_max - pitch_min)

    yaw_r = np.deg2rad(yaw)
    pitch_r = np.deg2rad(pitch)
    xw = np.cos(pitch_r) * np.sin(yaw_r)
    yw = np.sin(pitch_r)
    zw = np.cos(pitch_r) * np.cos(yaw_r)

    xl, yl, zl = inverse_rotate_rays(xw, yw, zw, patch_yaw_deg, patch_pitch_deg)

    valid = zl > 1e-8
    xp = fx * (xl / np.maximum(zl, 1e-8)) + cx
    yp = fy * (yl / np.maximum(zl, 1e-8)) + cy
    valid &= (xp >= 0.0) & (xp < patch_w - 1) & (yp >= 0.0) & (yp < patch_h - 1)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    wsum = np.zeros((out_h, out_w), dtype=np.float32)
    if not np.any(valid):
        return out, wsum

    sampled_rgb = sample_bilinear(patch, xp[valid], yp[valid]).astype(np.uint8)
    sampled_w = sample_bilinear(patch_weight, xp[valid], yp[valid]).astype(np.float32)
    sampled_w *= np.clip(zl[valid], 0.0, 1.0).astype(np.float32)

    roi_rgb = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    roi_w = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    roi_rgb[valid] = sampled_rgb
    roi_w[valid] = sampled_w

    out[y0:y1, x0:x1] = roi_rgb
    wsum[y0:y1, x0:x1] = roi_w
    return out, wsum


def _gaussian_pyramid(img, levels):
    pyr = [img]
    for _ in range(levels):
        src = pyr[-1]
        if src.shape[0] < 2 or src.shape[1] < 2:
            break
        h2 = src.shape[0] // 2 * 2
        w2 = src.shape[1] // 2 * 2
        src2 = src[:h2, :w2]
        down = 0.25 * (
            src2[0::2, 0::2]
            + src2[1::2, 0::2]
            + src2[0::2, 1::2]
            + src2[1::2, 1::2]
        )
        pyr.append(down.astype(np.float32))
    return pyr


def _resize_bilinear(img, out_h, out_w):
    in_h, in_w = img.shape[:2]
    if in_h == out_h and in_w == out_w:
        return img.copy()

    y = np.linspace(0.0, max(in_h - 1, 0), out_h, dtype=np.float32)
    x = np.linspace(0.0, max(in_w - 1, 0), out_w, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)

    x0 = np.clip(x0, 0, in_w - 1)
    y0 = np.clip(y0, 0, in_h - 1)

    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)

    if img.ndim == 2:
        ia = img[y0, x0]
        ib = img[y0, x1]
        ic = img[y1, x0]
        id_ = img[y1, x1]
        return ia * wa + ib * wb + ic * wc + id_ * wd

    ia = img[y0, x0, :]
    ib = img[y0, x1, :]
    ic = img[y1, x0, :]
    id_ = img[y1, x1, :]
    return (
        ia * wa[..., None]
        + ib * wb[..., None]
        + ic * wc[..., None]
        + id_ * wd[..., None]
    )


def _upsample2x(img, out_h, out_w):
    return _resize_bilinear(img, out_h, out_w).astype(np.float32, copy=False)


def _laplacian_pyramid(img, levels):
    gp = _gaussian_pyramid(img, levels)
    lp = []
    for i in range(len(gp) - 1):
        up = _upsample2x(gp[i + 1], gp[i].shape[0], gp[i].shape[1])
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp


def _multiband_blend(img_a, img_b, alpha, levels):
    mask3 = alpha[..., None].astype(np.float32)
    lap_a = _laplacian_pyramid(img_a.astype(np.float32), levels)
    lap_b = _laplacian_pyramid(img_b.astype(np.float32), levels)
    gmask = _gaussian_pyramid(mask3.astype(np.float32), levels)

    out_pyr = []
    for la, lb, gm in zip(lap_a, lap_b, gmask):
        out_pyr.append(la * (1.0 - gm) + lb * gm)

    out = out_pyr[-1]
    for i in range(len(out_pyr) - 2, -1, -1):
        out = _upsample2x(out, out_pyr[i].shape[0], out_pyr[i].shape[1]) + out_pyr[i]

    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)


def _gain_match(base_img, base_w, cur_img, cur_w):
    both = (base_w > 1e-8) & (cur_w > 1e-8)
    if both.sum() < 256:
        return cur_img
    base_mean = np.maximum(base_img[both].mean(axis=0), 1.0)
    cur_mean = np.maximum(cur_img[both].mean(axis=0), 1.0)
    gain = np.clip(base_mean / cur_mean, 0.85, 1.18)
    out = cur_img.astype(np.float32) * gain[None, None, :]
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)


def _adaptive_patch_centers(
    fish_img,
    patch_count,
    yaw_min,
    yaw_max,
    fish_fov_deg,
    threshold=8,
):
    gray = fish_img.mean(axis=2)
    active = gray > threshold
    yy, xx = np.where(active)
    if yy.size < 1024:
        return np.linspace(yaw_min, yaw_max, patch_count).tolist()

    hs, ws = fish_img.shape[:2]
    cx = ws * 0.5
    cy = hs * 0.5
    radius = min(hs, ws) * 0.5
    theta_max = np.deg2rad(fish_fov_deg * 0.5)

    dx = (xx.astype(np.float32) - cx) / max(radius, 1e-6)
    dy = (yy.astype(np.float32) - cy) / max(radius, 1e-6)
    rr = np.sqrt(dx * dx + dy * dy)
    inside = rr <= 1.0
    if inside.sum() < 512:
        return np.linspace(yaw_min, yaw_max, patch_count).tolist()

    rr = rr[inside]
    dx = dx[inside]
    dy = dy[inside]

    theta = rr * theta_max
    phi = np.arctan2(dy, dx)
    xw = np.sin(theta) * np.cos(phi)
    zw = np.cos(theta)
    yaw = np.rad2deg(np.arctan2(xw, zw))

    bins = 720
    hist, edges = np.histogram(yaw, bins=bins, range=(-180.0, 180.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float32)
    in_range = (centers >= yaw_min) & (centers <= yaw_max)
    w[~in_range] = 0.0
    if w.sum() < 1e-6:
        return np.linspace(yaw_min, yaw_max, patch_count).tolist()

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    for _ in range(2):
        wp = np.pad(w, (2, 2), mode="wrap")
        w = np.convolve(wp, kernel, mode="valid")
        w[~in_range] = 0.0

    cdf = np.cumsum(w)
    cdf /= max(cdf[-1], 1e-6)
    ux, uid = np.unique(cdf, return_index=True)
    if ux.size < 2:
        return np.linspace(yaw_min, yaw_max, patch_count).tolist()

    targets = (np.arange(patch_count) + 0.5) / patch_count
    adaptive = np.interp(targets, ux, centers[uid])
    uniform = np.linspace(yaw_min, yaw_max, patch_count)
    mixed = 0.65 * adaptive + 0.35 * uniform
    mixed = np.clip(mixed, yaw_min, yaw_max)
    mixed.sort()
    return mixed.tolist()


def fill_uncovered_pixels_rowwise(img, valid_mask):
    out = img.copy()
    h, w = valid_mask.shape
    xx = np.arange(w)
    for row in range(h):
        good = valid_mask[row]
        if good.sum() < 2:
            continue
        bad = ~good
        if not np.any(bad):
            continue
        gx = xx[good]
        bx = xx[bad]
        for c in range(3):
            vals = out[row, good, c].astype(np.float32)
            out[row, bad, c] = np.interp(bx, gx, vals).astype(np.uint8)
    return out


def run_pipeline(
    fish_img,
    out_w=3200,
    out_h=1200,
    fish_fov_deg=180.0,
    pano_span_deg=175.0,
    pano_pitch_center_deg=0.0,
    pano_pitch_span_deg=120.0,
    patch_count=8,
    patch_fov_x_deg=90.0,
    patch_fov_y_deg=120.0,
    patch_w=0,
    patch_h=0,
    multiband_levels=4,
    projection_model="equidistant",
    hybrid_blend=0.5,
    radial_k2=0.0,
    center_dx=0.0,
    center_dy=0.0,
    radius_scale=1.0,
    confidence_gamma=1.0,
    center_mode="adaptive",
    warp_mode="inverse",
    save_debug_dir=None,
):
    yaw_min = -0.5 * pano_span_deg
    yaw_max = 0.5 * pano_span_deg
    pitch_min = pano_pitch_center_deg - 0.5 * pano_pitch_span_deg
    pitch_max = pano_pitch_center_deg + 0.5 * pano_pitch_span_deg

    if patch_w <= 0:
        patch_w = max(int(round(out_w * patch_fov_x_deg / max(pano_span_deg, 1e-6))), 64)
    if patch_h <= 0:
        patch_h = max(int(round(out_h * patch_fov_y_deg / max(pano_pitch_span_deg, 1e-6))), 64)
    if warp_mode not in {"forward", "inverse"}:
        raise ValueError("warp_mode: forward veya inverse olmalı.")

    if center_mode == "adaptive":
        yaws = _adaptive_patch_centers(
            fish_img,
            patch_count,
            yaw_min,
            yaw_max,
            fish_fov_deg,
        )
    else:
        yaws = np.linspace(yaw_min, yaw_max, patch_count).tolist()

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)

    texture_map = _texture_confidence_map(fish_img)
    xr0, yr0, zr0, local_weight = build_rectilinear_patch_rays(
        patch_w, patch_h, patch_fov_x_deg, patch_fov_y_deg
    )

    comp_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    comp_w = np.zeros((out_h, out_w), dtype=np.float32)
    diagnostics = []
    overlap_err_sum = 0.0
    overlap_px_sum = 0.0

    for i, yaw_c in enumerate(yaws):
        xw, yw, zw = rotate_rays(xr0, yr0, zr0, yaw_c, pano_pitch_center_deg)
        patch, valid, conf = sample_fisheye_from_world_rays(
            fish_img=fish_img,
            texture_map=texture_map,
            x=xw,
            y=yw,
            z=zw,
            fish_fov_deg=fish_fov_deg,
            projection_model=projection_model,
            hybrid_blend=hybrid_blend,
            radial_k2=radial_k2,
            center_dx=center_dx,
            center_dy=center_dy,
            radius_scale=radius_scale,
            confidence_gamma=confidence_gamma,
        )
        patch_weight = local_weight * conf

        yaw = np.rad2deg(np.arctan2(xw, zw))
        pitch = np.rad2deg(np.arcsin(np.clip(yw, -1.0, 1.0)))
        if warp_mode == "forward":
            warped_img, warped_w = warp_patch_to_canvas(
                patch=patch,
                valid=valid,
                yaw=yaw,
                pitch=pitch,
                local_weight=patch_weight,
                out_w=out_w,
                out_h=out_h,
                yaw_min=yaw_min,
                yaw_max=yaw_max,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
            )
        else:
            warped_img, warped_w = inverse_warp_patch_to_canvas(
                patch=patch,
                patch_weight=patch_weight,
                patch_yaw_deg=yaw_c,
                patch_pitch_deg=pano_pitch_center_deg,
                patch_fov_x_deg=patch_fov_x_deg,
                patch_fov_y_deg=patch_fov_y_deg,
                out_w=out_w,
                out_h=out_h,
                yaw_min=yaw_min,
                yaw_max=yaw_max,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
            )

        if i == 0:
            comp_img = warped_img
            comp_w = warped_w
            alpha_for_debug = np.zeros((out_h, out_w), dtype=np.float32)
            alpha_for_debug[warped_w > 1e-8] = 1.0
        else:
            overlap = (comp_w > 1e-8) & (warped_w > 1e-8)
            if np.any(overlap):
                diff = np.abs(
                    comp_img[overlap].astype(np.float32) - warped_img[overlap].astype(np.float32)
                )
                overlap_err_sum += float(diff.mean())
                overlap_px_sum += float(overlap.sum())

            # Spherical davranisla uyumlu birlesim: gain-match + yumusak alpha.
            warped_img = _gain_match(comp_img, comp_w, warped_img, warped_w)
            alpha = warped_w / np.maximum(comp_w + warped_w, 1e-8)
            alpha[(comp_w <= 1e-8) & (warped_w > 1e-8)] = 1.0
            alpha[(comp_w > 1e-8) & (warped_w <= 1e-8)] = 0.0
            blended = _multiband_blend(comp_img, warped_img, alpha, multiband_levels)

            union = (comp_w > 1e-8) | (warped_w > 1e-8)
            comp_img[union] = blended[union]
            comp_w = np.maximum(comp_w, warped_w)
            alpha_for_debug = alpha

        diagnostics.append(
            (
                i,
                float(yaw_c),
                float(valid.mean()),
                float((warped_w > 1e-8).mean()),
                float(conf[valid].mean()) if np.any(valid) else 0.0,
            )
        )

        if save_debug_dir:
            Image.fromarray(patch).save(os.path.join(save_debug_dir, f"patch_{i:02d}_yaw_{yaw_c:+.1f}.png"))
            Image.fromarray(warped_img).save(os.path.join(save_debug_dir, f"warp_{i:02d}_yaw_{yaw_c:+.1f}.png"))
            alpha_vis = np.clip(alpha_for_debug * 255.0, 0.0, 255.0).astype(np.uint8)
            Image.fromarray(alpha_vis).save(os.path.join(save_debug_dir, f"alpha_{i:02d}_yaw_{yaw_c:+.1f}.png"))

    coverage = float((comp_w > 1e-8).mean())
    overlap_mae = overlap_err_sum / max(overlap_px_sum, 1.0)
    valid_mask = comp_w > 1e-8
    if np.any(valid_mask):
        gray = comp_img.astype(np.float32).mean(axis=2)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
        gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
        sharpness = float(np.sqrt(gx * gx + gy * gy)[valid_mask].mean())
    else:
        sharpness = 0.0
    quality = {"overlap_mae": overlap_mae, "sharpness": sharpness}
    return comp_img, comp_w > 1e-8, coverage, diagnostics, patch_w, patch_h, yaws, quality


def auto_calibrate_fisheye_params(
    fish_img,
    out_w,
    out_h,
    fish_fov_deg,
    pano_span_deg,
    pano_pitch_center_deg,
    pano_pitch_span_deg,
    patch_count,
    patch_fov_x_deg,
    patch_fov_y_deg,
    patch_w,
    patch_h,
    multiband_levels,
    confidence_gamma,
    center_mode,
    warp_mode,
):
    scale = min(1.0, 960.0 / max(out_w, 1), 360.0 / max(out_h, 1))
    cal_w = max(int(round(out_w * scale)), 320)
    cal_h = max(int(round(out_h * scale)), 180)
    cal_patch_count = min(max(patch_count, 4), 6)
    cal_patch_w = 0 if patch_w <= 0 else max(int(round(patch_w * scale)), 64)
    cal_patch_h = 0 if patch_h <= 0 else max(int(round(patch_h * scale)), 64)

    model_candidates = [
        ("equidistant", 0.5),
        ("equisolid", 0.5),
        ("stereographic", 0.5),
        ("orthographic", 0.5),
        ("hybrid", 0.35),
        ("hybrid", 0.50),
        ("hybrid", 0.65),
    ]
    fov_candidates = [fish_fov_deg + d for d in (-10.0, -5.0, 0.0, 5.0, 10.0)]
    fov_candidates = [float(np.clip(v, 120.0, 220.0)) for v in fov_candidates]
    fov_candidates = sorted(set(fov_candidates))
    radius_candidates = [0.96, 1.00, 1.04]
    k2_candidates = [-0.20, -0.10, 0.00, 0.10, 0.20]

    best = None
    candidates = []

    def eval_one(params):
        _, _, coverage, _, _, _, _, quality = run_pipeline(
            fish_img=fish_img,
            out_w=cal_w,
            out_h=cal_h,
            fish_fov_deg=params["fish_fov_deg"],
            pano_span_deg=pano_span_deg,
            pano_pitch_center_deg=pano_pitch_center_deg,
            pano_pitch_span_deg=pano_pitch_span_deg,
            patch_count=cal_patch_count,
            patch_fov_x_deg=patch_fov_x_deg,
            patch_fov_y_deg=patch_fov_y_deg,
            patch_w=cal_patch_w,
            patch_h=cal_patch_h,
            multiband_levels=min(multiband_levels, 2),
            projection_model=params["projection_model"],
            hybrid_blend=params["hybrid_blend"],
            radial_k2=params["radial_k2"],
            center_dx=params["center_dx"],
            center_dy=params["center_dy"],
            radius_scale=params["radius_scale"],
            confidence_gamma=confidence_gamma,
            center_mode=center_mode,
            warp_mode=warp_mode,
            save_debug_dir=None,
        )
        overlap_mae = quality["overlap_mae"] / 255.0
        sharp = quality["sharpness"] / 255.0
        score = overlap_mae + 0.35 * (1.0 - coverage) - 0.10 * sharp
        row = dict(params)
        row["coverage"] = coverage
        row["overlap_mae"] = quality["overlap_mae"]
        row["sharpness"] = quality["sharpness"]
        row["score"] = score
        return row

    for model, blend in model_candidates:
        for fov in fov_candidates:
            for radius_scale in radius_candidates:
                for k2 in k2_candidates:
                    params = {
                        "projection_model": model,
                        "hybrid_blend": blend,
                        "fish_fov_deg": fov,
                        "radius_scale": radius_scale,
                        "radial_k2": k2,
                        "center_dx": 0.0,
                        "center_dy": 0.0,
                    }
                    row = eval_one(params)
                    candidates.append(row)
                    if (best is None) or (row["score"] < best["score"]):
                        best = row

    center_offsets = [-0.02, 0.0, 0.02]
    for dx in center_offsets:
        for dy in center_offsets:
            params = {
                "projection_model": best["projection_model"],
                "hybrid_blend": best["hybrid_blend"],
                "fish_fov_deg": best["fish_fov_deg"],
                "radius_scale": best["radius_scale"],
                "radial_k2": best["radial_k2"],
                "center_dx": dx,
                "center_dy": dy,
            }
            row = eval_one(params)
            candidates.append(row)
            if row["score"] < best["score"]:
                best = row

    candidates.sort(key=lambda x: x["score"])
    return best, candidates[:8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="academic_stitch.png")
    parser.add_argument("--lens", default="auto")
    parser.add_argument("--crop-pad", type=float, default=1.12)
    parser.add_argument("--fish-fov", type=float, default=180.0)
    parser.add_argument("--out-w", type=int, default=3200)
    parser.add_argument("--out-h", type=int, default=1200)
    parser.add_argument("--span", type=float, default=175.0)
    parser.add_argument("--pitch-center", type=float, default=0.0)
    parser.add_argument("--pitch-span", type=float, default=120.0)
    parser.add_argument("--patch-count", type=int, default=8)
    parser.add_argument("--patch-w", type=int, default=0)
    parser.add_argument("--patch-h", type=int, default=0)
    parser.add_argument("--patch-fov-x", type=float, default=90.0)
    parser.add_argument("--patch-fov-y", type=float, default=120.0)
    parser.add_argument(
        "--projection-model",
        choices=["equidistant", "equisolid", "stereographic", "orthographic", "hybrid"],
        default="equidistant",
    )
    parser.add_argument("--hybrid-blend", type=float, default=0.5)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--radial-k2", type=float, default=0.0)
    parser.add_argument("--center-dx", type=float, default=0.0)
    parser.add_argument("--center-dy", type=float, default=0.0)
    parser.add_argument("--auto-calibrate", action="store_true")
    parser.add_argument("--center-mode", choices=["adaptive", "uniform"], default="adaptive")
    parser.add_argument("--warp-mode", choices=["forward", "inverse"], default="inverse")
    parser.add_argument("--multiband-levels", type=int, default=4)
    parser.add_argument("--confidence-gamma", type=float, default=1.0)
    parser.add_argument("--save-debug-dir", default="")
    parser.add_argument("--fill-holes", action="store_true")
    args = parser.parse_args()

    raw = np.array(Image.open(args.input).convert("RGB"))
    fish_img, used_lens = select_single_fisheye_lens(raw, args.lens, threshold=8, pad_ratio=args.crop_pad)

    best_params = None
    top_candidates = []
    if args.auto_calibrate:
        best_params, top_candidates = auto_calibrate_fisheye_params(
            fish_img=fish_img,
            out_w=args.out_w,
            out_h=args.out_h,
            fish_fov_deg=args.fish_fov,
            pano_span_deg=args.span,
            pano_pitch_center_deg=args.pitch_center,
            pano_pitch_span_deg=args.pitch_span,
            patch_count=args.patch_count,
            patch_fov_x_deg=args.patch_fov_x,
            patch_fov_y_deg=args.patch_fov_y,
            patch_w=args.patch_w,
            patch_h=args.patch_h,
            multiband_levels=args.multiband_levels,
            confidence_gamma=args.confidence_gamma,
            center_mode=args.center_mode,
            warp_mode=args.warp_mode,
        )
        args.fish_fov = best_params["fish_fov_deg"]
        args.projection_model = best_params["projection_model"]
        args.hybrid_blend = best_params["hybrid_blend"]
        args.radius_scale = best_params["radius_scale"]
        args.radial_k2 = best_params["radial_k2"]
        args.center_dx = best_params["center_dx"]
        args.center_dy = best_params["center_dy"]

    pano, valid, coverage, diagnostics, used_patch_w, used_patch_h, yaws, quality = run_pipeline(
        fish_img=fish_img,
        out_w=args.out_w,
        out_h=args.out_h,
        fish_fov_deg=args.fish_fov,
        pano_span_deg=args.span,
        pano_pitch_center_deg=args.pitch_center,
        pano_pitch_span_deg=args.pitch_span,
        patch_count=args.patch_count,
        patch_fov_x_deg=args.patch_fov_x,
        patch_fov_y_deg=args.patch_fov_y,
        patch_w=args.patch_w,
        patch_h=args.patch_h,
        multiband_levels=args.multiband_levels,
        projection_model=args.projection_model,
        hybrid_blend=args.hybrid_blend,
        radial_k2=args.radial_k2,
        center_dx=args.center_dx,
        center_dy=args.center_dy,
        radius_scale=args.radius_scale,
        confidence_gamma=args.confidence_gamma,
        center_mode=args.center_mode,
        warp_mode=args.warp_mode,
        save_debug_dir=args.save_debug_dir if args.save_debug_dir else None,
    )

    if args.fill_holes:
        pano = fill_uncovered_pixels_rowwise(pano, valid)

    Image.fromarray(pano).save(args.output)

    print("Completed")
    print("Output:", os.path.abspath(args.output))
    print("Lens:", used_lens, "| Processed:", f"{fish_img.shape[1]}x{fish_img.shape[0]}")
    print("Warp mode:", args.warp_mode)
    print(
        "Projection:",
        args.projection_model,
        "| Hybrid:",
        f"{args.hybrid_blend:.2f}",
        "| RadiusScale:",
        f"{args.radius_scale:.3f}",
        "| k2:",
        f"{args.radial_k2:+.3f}",
        "| Center dx/dy:",
        f"{args.center_dx:+.3f}/{args.center_dy:+.3f}",
    )
    print("Center mode:", args.center_mode, "| Patch centers:", ", ".join(f"{v:+.2f}" for v in yaws))
    print("Patch:", args.patch_count, "| PatchWxH:", f"{used_patch_w}x{used_patch_h}", "| PatchFOV:", f"{args.patch_fov_x:.1f}/{args.patch_fov_y:.1f}")
    print("Pano span/pitch span:", f"{args.span:.1f}/{args.pitch_span:.1f}")
    print("Coverage:", f"{coverage * 100:.2f}%")
    print("Quality overlap_mae/sharpness:", f"{quality['overlap_mae']:.3f}/{quality['sharpness']:.3f}")
    if best_params is not None:
        print("Auto-calibrate best score:", f"{best_params['score']:.6f}")
        print("Auto-calibrate top candidates:")
        for i, row in enumerate(top_candidates):
            print(
                f"  #{i+1:02d} score {row['score']:.6f} | model {row['projection_model']:<12}"
                f" | fov {row['fish_fov_deg']:.2f} | rs {row['radius_scale']:.3f}"
                f" | k2 {row['radial_k2']:+.3f} | cdx/cdy {row['center_dx']:+.3f}/{row['center_dy']:+.3f}"
                f" | cov {row['coverage']*100:6.2f}% | ov {row['overlap_mae']:.3f} | sh {row['sharpness']:.3f}"
            )
    for i, yaw_c, src_cov, warp_cov, conf_mean in diagnostics:
        print(
            f"  patch {i:02d} | yaw {yaw_c:+6.2f} | srcCov {src_cov * 100:6.2f}%"
            f" | warpCov {warp_cov * 100:6.2f}% | conf {conf_mean:0.4f}"
        )


if __name__ == "__main__":
    main()
