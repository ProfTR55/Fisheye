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


def rays_to_fisheye(fish_img, x, y, z, fish_fov_deg):
    hs, ws = fish_img.shape[:2]
    cx_src = ws * 0.5
    cy_src = hs * 0.5
    radius_src = min(hs, ws) * 0.5
    theta_max = np.deg2rad(fish_fov_deg * 0.5)

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    alpha = np.arctan2(y, x)
    r = (theta / theta_max) * radius_src

    xs = cx_src + r * np.cos(alpha)
    ys = cy_src + r * np.sin(alpha)

    valid = theta <= theta_max
    valid &= (xs >= 0) & (xs < ws - 1) & (ys >= 0) & (ys < hs - 1)

    out = np.zeros((*x.shape, 3), dtype=np.uint8)
    out[valid] = sample_bilinear(fish_img, xs[valid], ys[valid])
    return out, valid


def render_rect_panel(
    fish_img,
    out_w,
    out_h,
    view_yaw_deg,
    view_pitch_deg,
    fov_deg,
    fish_fov_deg,
):
    rect_fov = np.deg2rad(fov_deg)
    fx = (out_w * 0.5) / np.tan(rect_fov * 0.5)
    fy = fx
    cx = out_w * 0.5
    cy = out_h * 0.5

    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)

    norm = np.sqrt(x * x + y * y + z * z)
    x /= norm
    y /= norm
    z /= norm

    x2, y2, z2 = rotate_rays(x, y, z, view_yaw_deg, view_pitch_deg)
    return rays_to_fisheye(fish_img, x2, y2, z2, fish_fov_deg)


def render_cyl_panel(
    fish_img,
    out_w,
    out_h,
    view_yaw_deg,
    view_pitch_deg,
    fov_deg,
    fish_fov_deg,
):
    cyl_fov = np.deg2rad(fov_deg)
    f = (out_w * 0.5) / np.tan(cyl_fov * 0.5)
    cx = out_w * 0.5
    cy = out_h * 0.5

    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (u - cx) / f
    y = (v - cy) / f

    lam = np.arctan(x)
    phi = np.arctan(y / np.sqrt(x * x + 1.0))

    xr = np.cos(phi) * np.sin(lam)
    yr = np.sin(phi)
    zr = np.cos(phi) * np.cos(lam)

    x2, y2, z2 = rotate_rays(xr, yr, zr, view_yaw_deg, view_pitch_deg)
    return rays_to_fisheye(fish_img, x2, y2, z2, fish_fov_deg)


def choose_projection(seg_idx, seg_count, mode, rect_threshold):
    if mode == "all_rect":
        return "rect"
    if mode == "all_cyl":
        return "cyl"

    center = (seg_count - 1) * 0.5
    denom = max(center, 1.0)
    d = abs(seg_idx - center) / denom
    return "rect" if d <= rect_threshold else "cyl"


def cosine_blend_weights(panel_w, overlap_px, is_first, is_last):
    weights = np.ones(panel_w, dtype=np.float32)

    if overlap_px > 0 and not is_first:
        t = np.linspace(0.0, 1.0, overlap_px, dtype=np.float32)
        weights[:overlap_px] = 0.5 - 0.5 * np.cos(np.pi * t)

    if overlap_px > 0 and not is_last:
        t = np.linspace(0.0, 1.0, overlap_px, dtype=np.float32)
        weights[-overlap_px:] = 0.5 + 0.5 * np.cos(np.pi * t)

    return weights


def fill_uncovered_pixels_rowwise(img, valid_mask):
    out = img.copy()
    h, w = valid_mask.shape
    x = np.arange(w)

    for row in range(h):
        good = valid_mask[row]
        if good.sum() < 2:
            continue
        bad = ~good
        if not np.any(bad):
            continue

        good_x = x[good]
        bad_x = x[bad]
        for c in range(3):
            values = out[row, good, c].astype(np.float32)
            out[row, bad, c] = np.interp(bad_x, good_x, values).astype(np.uint8)

    return out


def render_hybrid_panorama(
    fish_img,
    out_w=3000,
    out_h=1200,
    fish_fov_deg=180.0,
    pano_span_deg=170.0,
    pitch_deg=0.0,
    panel_count=8,
    overlap=0.35,
    panel_fov_deg=0.0,
    mode="hybrid",
    rect_threshold=0.35,
    save_panels_dir=None,
):
    if panel_count < 2:
        raise ValueError("panel_count en az 2 olmalı.")
    if not (0.05 <= overlap < 0.8):
        raise ValueError("overlap 0.05 ile 0.8 arasında olmalı.")
    if mode not in {"hybrid", "all_rect", "all_cyl"}:
        raise ValueError("mode: hybrid/all_rect/all_cyl")

    step_yaw = pano_span_deg / (panel_count - 1)
    yaw_centers = np.linspace(-0.5 * pano_span_deg, 0.5 * pano_span_deg, panel_count).tolist()

    if panel_fov_deg <= 0:
        auto_fov = step_yaw * 2.2
        panel_fov_deg = float(np.clip(auto_fov, 80.0, 110.0))

    deg_per_px = pano_span_deg / max(out_w, 1)
    stride_px = int(round(step_yaw / deg_per_px))
    panel_w = int(round(panel_fov_deg / deg_per_px))
    panel_w = max(panel_w, 64)
    stride_px = max(stride_px, 1)

    geom_overlap_px = max(panel_w - stride_px, 0)
    overlap_px = int(round(geom_overlap_px * overlap))
    overlap_px = min(overlap_px, panel_w // 2)
    stitched_w = panel_w + (panel_count - 1) * stride_px

    accum = np.zeros((out_h, stitched_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, stitched_w), dtype=np.float32)
    segments = []

    if save_panels_dir:
        os.makedirs(save_panels_dir, exist_ok=True)

    for i, yaw in enumerate(yaw_centers):
        proj = choose_projection(i, panel_count, mode, rect_threshold)
        if proj == "rect":
            panel, panel_valid = render_rect_panel(
                fish_img,
                out_w=panel_w,
                out_h=out_h,
                view_yaw_deg=yaw,
                view_pitch_deg=pitch_deg,
                fov_deg=panel_fov_deg,
                fish_fov_deg=fish_fov_deg,
            )
        else:
            panel, panel_valid = render_cyl_panel(
                fish_img,
                out_w=panel_w,
                out_h=out_h,
                view_yaw_deg=yaw,
                view_pitch_deg=pitch_deg,
                fov_deg=panel_fov_deg,
                fish_fov_deg=fish_fov_deg,
            )

        segments.append((i, yaw, proj, float(panel_valid.mean())))

        if save_panels_dir:
            panel_name = f"panel_{i:02d}_{proj}_yaw_{yaw:+.1f}.png"
            Image.fromarray(panel).save(os.path.join(save_panels_dir, panel_name))

        x0 = i * stride_px
        x1 = x0 + panel_w
        w1d = cosine_blend_weights(panel_w, overlap_px, i == 0, i == panel_count - 1)
        w2d = np.tile(w1d, (out_h, 1))
        w2d *= panel_valid.astype(np.float32)

        accum[:, x0:x1] += panel.astype(np.float32) * w2d[..., None]
        wsum[:, x0:x1] += w2d

    pano = np.zeros((out_h, stitched_w, 3), dtype=np.uint8)
    valid = wsum > 1e-6
    pano[valid] = (accum[valid] / wsum[valid, None]).astype(np.uint8)
    coverage_ratio = float(valid.mean())

    if stitched_w != out_w:
        start = max((stitched_w - out_w) // 2, 0)
        end = start + out_w
        pano = pano[:, start:end]
        valid = valid[:, start:end]

    return pano, valid, coverage_ratio, yaw_centers, panel_fov_deg, step_yaw, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="hybrid_pano.png")
    parser.add_argument("--lens", default="auto")
    parser.add_argument("--fish-fov", type=float, default=180.0)
    parser.add_argument("--out-w", type=int, default=3000)
    parser.add_argument("--out-h", type=int, default=1200)
    parser.add_argument("--span", type=float, default=170.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--panels", type=int, default=8)
    parser.add_argument("--overlap", type=float, default=0.35)
    parser.add_argument("--panel-fov", type=float, default=0.0)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--rect-threshold", type=float, default=0.35)
    parser.add_argument("--save-panels-dir", default="")
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--crop-pad", type=float, default=1.12)
    args = parser.parse_args()

    raw_img = np.array(Image.open(args.input).convert("RGB"))
    fish_img, used_lens = select_single_fisheye_lens(
        raw_img,
        args.lens,
        threshold=8,
        pad_ratio=args.crop_pad,
    )

    pano, valid, coverage, yaw_centers, used_panel_fov, step_yaw, segments = render_hybrid_panorama(
        fish_img,
        out_w=args.out_w,
        out_h=args.out_h,
        fish_fov_deg=args.fish_fov,
        pano_span_deg=args.span,
        pitch_deg=args.pitch,
        panel_count=args.panels,
        overlap=args.overlap,
        panel_fov_deg=args.panel_fov,
        mode=args.mode,
        rect_threshold=args.rect_threshold,
        save_panels_dir=args.save_panels_dir if args.save_panels_dir else None,
    )

    if args.fill_holes:
        pano = fill_uncovered_pixels_rowwise(pano, valid)

    Image.fromarray(pano).save(args.output)

    print("Tamamlandı")
    print("Çıktı:", os.path.abspath(args.output))
    print("Lens:", used_lens, "| İşlenen boyut:", f"{fish_img.shape[1]}x{fish_img.shape[0]}")
    print("Mode:", args.mode, "| Rect threshold:", f"{args.rect_threshold:.2f}")
    print("Panel sayısı:", args.panels, "| Panel FOV:", f"{used_panel_fov:.2f}°", "| Step yaw:", f"{step_yaw:.2f}°")
    print("Kapsama oranı:", f"{coverage * 100:.2f}%")
    print("Yaw merkezleri:", ", ".join(f"{y:+.1f}" for y in yaw_centers))
    for idx, yaw, proj, panel_cov in segments:
        print(f"  panel {idx:02d} | yaw {yaw:+6.2f} | {proj:4s} | cov {panel_cov * 100:6.2f}%")


if __name__ == "__main__":
    main()
