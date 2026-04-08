import argparse
import os
import numpy as np # pyright: ignore[reportMissingImports]
from PIL import Image # pyright: ignore[reportMissingImports]


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


def sample_fisheye_from_world_rays(fish_img, x, y, z, fish_fov_deg):
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

    patch = np.zeros((*x.shape, 3), dtype=np.uint8)
    patch[valid] = sample_bilinear(fish_img, xs[valid], ys[valid])
    return patch, valid


def _splat_bilinear(accum, wsum, x, y, rgb, w):
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


def warp_patch_to_canvas(patch, valid, yaw, pitch, local_weight, out_w, out_h, yaw_min, yaw_max, pitch_min, pitch_max):
    xc = (yaw - yaw_min) / max(yaw_max - yaw_min, 1e-6) * (out_w - 1)
    yc = (pitch - pitch_min) / (pitch_max - pitch_min) * (out_h - 1)

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
            rgb=patch[mask].astype(np.float32),
            w=ww[mask].astype(np.float32),
        )

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ok = wsum > 1e-8
    out[ok] = (accum[ok] / wsum[ok, None]).astype(np.uint8)
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
    multiband_levels=4,
    save_debug_dir=None,
):
    yaw_min = -0.5 * pano_span_deg
    yaw_max = 0.5 * pano_span_deg
    pitch_min = pano_pitch_center_deg - 0.5 * pano_pitch_span_deg
    pitch_max = pano_pitch_center_deg + 0.5 * pano_pitch_span_deg

    if patch_w <= 0:
        patch_w = max(int(round(out_w * patch_fov_x_deg / max(pano_span_deg, 1e-6))), 64)
    patch_h = out_h
    yaws = np.linspace(yaw_min, yaw_max, patch_count).tolist()

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)

    xr0, yr0, zr0, local_weight = build_rectilinear_patch_rays(patch_w, patch_h, patch_fov_x_deg, patch_fov_y_deg)

    comp_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    comp_w = np.zeros((out_h, out_w), dtype=np.float32)
    diagnostics = []

    for i, yaw_c in enumerate(yaws):
        xw, yw, zw = rotate_rays(xr0, yr0, zr0, yaw_c, pano_pitch_center_deg)
        patch, valid = sample_fisheye_from_world_rays(fish_img, xw, yw, zw, fish_fov_deg)

        yaw = np.rad2deg(np.arctan2(xw, zw))
        pitch = np.rad2deg(np.arcsin(np.clip(yw, -1.0, 1.0)))
        warped_img, warped_w = warp_patch_to_canvas(
            patch=patch,
            valid=valid,
            yaw=yaw,
            pitch=pitch,
            local_weight=local_weight,
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
        else:
            warped_img = _gain_match(comp_img, comp_w, warped_img, warped_w)
            alpha = warped_w / np.maximum(comp_w + warped_w, 1e-8)
            alpha[(comp_w <= 1e-8) & (warped_w > 1e-8)] = 1.0
            alpha[(comp_w > 1e-8) & (warped_w <= 1e-8)] = 0.0
            blended = _multiband_blend(comp_img, warped_img, alpha, multiband_levels)

            union = (comp_w > 1e-8) | (warped_w > 1e-8)
            comp_img[union] = blended[union]
            comp_w = np.maximum(comp_w, warped_w)

        diagnostics.append((i, yaw_c, float(valid.mean()), float((warped_w > 1e-8).mean())))

        if save_debug_dir:
            Image.fromarray(patch).save(os.path.join(save_debug_dir, f"patch_{i:02d}_yaw_{yaw_c:+.1f}.png"))
            Image.fromarray(warped_img).save(os.path.join(save_debug_dir, f"warp_{i:02d}_yaw_{yaw_c:+.1f}.png"))

    coverage = float((comp_w > 1e-8).mean())
    return comp_img, comp_w > 1e-8, coverage, diagnostics, patch_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="spherical_patch_stitch.png")
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
    parser.add_argument("--patch-fov-x", type=float, default=90.0)
    parser.add_argument("--patch-fov-y", type=float, default=120.0)
    parser.add_argument("--multiband-levels", type=int, default=4)
    parser.add_argument("--save-debug-dir", default="")
    parser.add_argument("--fill-holes", action="store_true")
    args = parser.parse_args()

    raw = np.array(Image.open(args.input).convert("RGB"))
    fish_img, used_lens = select_single_fisheye_lens(raw, args.lens, threshold=8, pad_ratio=args.crop_pad)

    pano, valid, coverage, diagnostics, used_patch_w = run_pipeline(
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
        multiband_levels=args.multiband_levels,
        save_debug_dir=args.save_debug_dir if args.save_debug_dir else None,
    )

    if args.fill_holes:
        pano = fill_uncovered_pixels_rowwise(pano, valid)

    Image.fromarray(pano).save(args.output)

    print("Tamamlandı")
    print("Çıktı:", os.path.abspath(args.output))
    print("Lens:", used_lens, "| İşlenen boyut:", f"{fish_img.shape[1]}x{fish_img.shape[0]}")
    print("Patch:", args.patch_count, "| PatchW:", used_patch_w, "| PatchFOV:", f"{args.patch_fov_x:.1f}/{args.patch_fov_y:.1f}")
    print("Pano span/pitch span:", f"{args.span:.1f}/{args.pitch_span:.1f}")
    print("Kapsama oranı:", f"{coverage * 100:.2f}%")
    for i, yaw_c, src_cov, warp_cov in diagnostics:
        print(f"  patch {i:02d} | yaw {yaw_c:+6.2f} | srcCov {src_cov * 100:6.2f}% | warpCov {warp_cov * 100:6.2f}%")


if __name__ == "__main__":
    main()
