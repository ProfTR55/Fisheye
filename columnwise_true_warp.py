import argparse
import os
import numpy as np
from PIL import Image


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

    out = (
        ia * wa[..., None]
        + ib * wb[..., None]
        + ic * wc[..., None]
        + id_ * wd[..., None]
    )
    return out.astype(np.uint8)


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
            right_score = mask[:, w // 2 :].sum()
            mode = "left" if left_score >= right_score else "right"
        elif h >= 1.7 * w:
            top_score = mask[: h // 2, :].sum()
            bottom_score = mask[h // 2 :, :].sum()
            mode = "top" if top_score >= bottom_score else "bottom"
        else:
            mode = "full"

    if mode == "full":
        return _crop_square_around_active(img, threshold, pad_ratio), mode
    if mode == "left":
        return _crop_square_around_active(img[:, : w // 2], threshold, pad_ratio), mode
    if mode == "right":
        return _crop_square_around_active(img[:, w // 2 :], threshold, pad_ratio), mode
    if mode == "top":
        return _crop_square_around_active(img[: h // 2, :], threshold, pad_ratio), mode
    if mode == "bottom":
        return _crop_square_around_active(img[h // 2 :, :], threshold, pad_ratio), mode
    raise ValueError("lens mode geçersiz: auto/full/left/right/top/bottom")


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


def make_panel_rays(panel_w, panel_h, fov_x_deg, fov_y_deg, projection):
    cx = (panel_w - 1) * 0.5
    cy = (panel_h - 1) * 0.5
    u, v = np.meshgrid(np.arange(panel_w), np.arange(panel_h))

    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)
    fx = (panel_w * 0.5) / np.tan(fov_x * 0.5)
    fy = (panel_h * 0.5) / np.tan(fov_y * 0.5)

    x = (u - cx) / fx
    y = (v - cy) / fy

    if projection == "rect":
        z = np.ones_like(x)
        norm = np.sqrt(x * x + y * y + z * z)
        xr = x / norm
        yr = y / norm
        zr = z / norm
    else:
        lam = np.arctan(x)
        phi = np.arctan(y)
        xr = np.cos(phi) * np.sin(lam)
        yr = np.sin(phi)
        zr = np.cos(phi) * np.cos(lam)

    edge = 1.0 - np.abs((u - cx) / max(cx, 1.0))
    edge = np.clip(edge, 0.0, 1.0) ** 2
    return xr, yr, zr, edge.astype(np.float32)


def sample_fisheye_from_rays(fish_img, x, y, z, fish_fov_deg):
    hs, ws = fish_img.shape[:2]
    cx = ws * 0.5
    cy = hs * 0.5
    radius = min(hs, ws) * 0.5
    theta_max = np.deg2rad(fish_fov_deg * 0.5)

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    alpha = np.arctan2(y, x)
    r = (theta / theta_max) * radius
    xs = cx + r * np.cos(alpha)
    ys = cy + r * np.sin(alpha)

    valid = theta <= theta_max
    valid &= (xs >= 0) & (xs < ws - 1) & (ys >= 0) & (ys < hs - 1)

    panel = np.zeros((*x.shape, 3), dtype=np.uint8)
    panel[valid] = sample_bilinear(fish_img, xs[valid], ys[valid])
    return panel, valid


def _add_splat(accum, wsum, x, y, rgb, w):
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
        rgbi = rgb[inside]
        np.add.at(wsum, (yyi, xxi), wwi)
        np.add.at(accum[:, :, 0], (yyi, xxi), rgbi[:, 0] * wwi)
        np.add.at(accum[:, :, 1], (yyi, xxi), rgbi[:, 1] * wwi)
        np.add.at(accum[:, :, 2], (yyi, xxi), rgbi[:, 2] * wwi)

    fx = x - x0
    fy = y - y0
    add(x0, y0, w * (1.0 - fx) * (1.0 - fy))
    add(x1, y0, w * fx * (1.0 - fy))
    add(x0, y1, w * (1.0 - fx) * fy)
    add(x1, y1, w * fx * fy)


def choose_projection(seg_idx, seg_count, mode, rect_ratio):
    if mode == "all_rect":
        return "rect"
    if mode == "all_cyl":
        return "cyl"
    center = (seg_count - 1) * 0.5
    denom = max(center, 1.0)
    dist = abs(seg_idx - center) / denom
    return "rect" if dist <= rect_ratio else "cyl"


def parse_projection_pattern(pattern, segments):
    if not pattern:
        return None
    vals = [p.strip().lower() for p in pattern.split(",")]
    if len(vals) != segments:
        raise ValueError("projection-pattern uzunluğu segments ile aynı olmalı.")
    for p in vals:
        if p not in {"rect", "cyl"}:
            raise ValueError("projection-pattern sadece rect,cyl içermeli.")
    return vals


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


def render_columnwise_true_warp(
    fish_img,
    out_w=3200,
    out_h=1200,
    fish_fov_deg=180.0,
    span_deg=175.0,
    pitch_center_deg=0.0,
    pitch_span_deg=120.0,
    segments=8,
    panel_fov_x_deg=95.0,
    panel_fov_y_deg=120.0,
    panel_w=0,
    mode="adaptive",
    rect_ratio=0.35,
    projection_pattern=None,
    save_segments_dir=None,
):
    if segments < 2:
        raise ValueError("segments en az 2 olmalı.")
    if mode not in {"adaptive", "all_rect", "all_cyl"}:
        raise ValueError("mode: adaptive/all_rect/all_cyl")

    yaw_min = -0.5 * span_deg
    yaw_max = 0.5 * span_deg
    pitch_min = pitch_center_deg - 0.5 * pitch_span_deg
    pitch_max = pitch_center_deg + 0.5 * pitch_span_deg

    if panel_w <= 0:
        panel_w = max(int(round(out_w * panel_fov_x_deg / max(span_deg, 1e-6))), 64)
    panel_h = out_h

    yaw_centers = np.linspace(yaw_min, yaw_max, segments).tolist()
    pattern = parse_projection_pattern(projection_pattern, segments)

    accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, out_w), dtype=np.float32)
    diagnostics = []

    if save_segments_dir:
        os.makedirs(save_segments_dir, exist_ok=True)

    for i, yaw_c in enumerate(yaw_centers):
        proj = pattern[i] if pattern is not None else choose_projection(i, segments, mode, rect_ratio)
        xr, yr, zr, edge = make_panel_rays(panel_w, panel_h, panel_fov_x_deg, panel_fov_y_deg, proj)
        xw, yw, zw = rotate_rays(xr, yr, zr, yaw_c, pitch_center_deg)

        panel, valid = sample_fisheye_from_rays(fish_img, xw, yw, zw, fish_fov_deg)
        yaw = np.rad2deg(np.arctan2(xw, zw))
        pitch = np.rad2deg(np.arcsin(np.clip(yw, -1.0, 1.0)))

        mask = valid
        mask &= (yaw >= yaw_min) & (yaw <= yaw_max)
        mask &= (pitch >= pitch_min) & (pitch <= pitch_max)

        xc = (yaw - yaw_min) / max(yaw_max - yaw_min, 1e-6) * (out_w - 1)
        yc = (pitch_max - pitch) / max(pitch_max - pitch_min, 1e-6) * (out_h - 1)
        ww = edge * mask.astype(np.float32)

        _add_splat(
            accum=accum,
            wsum=wsum,
            x=xc[mask].astype(np.float32),
            y=yc[mask].astype(np.float32),
            rgb=panel[mask].astype(np.float32),
            w=ww[mask].astype(np.float32),
        )

        diagnostics.append((i, yaw_c, proj, float(valid.mean()), int(mask.sum())))
        if save_segments_dir:
            name = f"seg_{i:02d}_{proj}_yaw_{yaw_c:+.1f}.png"
            Image.fromarray(panel).save(os.path.join(save_segments_dir, name))

    pano = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    valid_canvas = wsum > 1e-7
    pano[valid_canvas] = (accum[valid_canvas] / wsum[valid_canvas, None]).astype(np.uint8)
    coverage = float(valid_canvas.mean())
    return pano, valid_canvas, coverage, diagnostics, panel_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="columnwise_true_warp.png")
    parser.add_argument("--lens", default="auto")
    parser.add_argument("--crop-pad", type=float, default=1.12)
    parser.add_argument("--fish-fov", type=float, default=180.0)
    parser.add_argument("--out-w", type=int, default=3200)
    parser.add_argument("--out-h", type=int, default=1200)
    parser.add_argument("--span", type=float, default=175.0)
    parser.add_argument("--pitch-center", type=float, default=0.0)
    parser.add_argument("--pitch-span", type=float, default=120.0)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--panel-w", type=int, default=0)
    parser.add_argument("--panel-fov-x", type=float, default=95.0)
    parser.add_argument("--panel-fov-y", type=float, default=120.0)
    parser.add_argument("--mode", default="adaptive")
    parser.add_argument("--rect-ratio", type=float, default=0.35)
    parser.add_argument("--projection-pattern", default="")
    parser.add_argument("--save-segments-dir", default="")
    parser.add_argument("--fill-holes", action="store_true")
    args = parser.parse_args()

    raw = np.array(Image.open(args.input).convert("RGB"))
    fish_img, used_lens = select_single_fisheye_lens(
        raw, lens_mode=args.lens, threshold=8, pad_ratio=args.crop_pad
    )

    pano, valid, coverage, diagnostics, used_panel_w = render_columnwise_true_warp(
        fish_img=fish_img,
        out_w=args.out_w,
        out_h=args.out_h,
        fish_fov_deg=args.fish_fov,
        span_deg=args.span,
        pitch_center_deg=args.pitch_center,
        pitch_span_deg=args.pitch_span,
        segments=args.segments,
        panel_fov_x_deg=args.panel_fov_x,
        panel_fov_y_deg=args.panel_fov_y,
        panel_w=args.panel_w,
        mode=args.mode,
        rect_ratio=args.rect_ratio,
        projection_pattern=args.projection_pattern,
        save_segments_dir=args.save_segments_dir if args.save_segments_dir else None,
    )

    if args.fill_holes:
        pano = fill_uncovered_pixels_rowwise(pano, valid)

    Image.fromarray(pano).save(args.output)

    print("Tamamlandı")
    print("Çıktı:", os.path.abspath(args.output))
    print("Lens:", used_lens, "| İşlenen boyut:", f"{fish_img.shape[1]}x{fish_img.shape[0]}")
    print("Segments:", args.segments, "| PanelW:", used_panel_w)
    print("Span/PitchSpan:", f"{args.span:.1f}/{args.pitch_span:.1f}")
    print("Kapsama oranı:", f"{coverage * 100:.2f}%")
    for i, yaw_c, proj, cov, used in diagnostics:
        print(f"  seg {i:02d} | yaw {yaw_c:+6.2f} | {proj:4s} | srcCov {cov * 100:6.2f}% | warpedPx {used}")


if __name__ == "__main__":
    main()

