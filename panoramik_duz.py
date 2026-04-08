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

    return (ia * wa[..., None] + ib * wb[..., None] + ic * wc[..., None] + id_ * wd[..., None]).astype(np.uint8)


def _active_bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _crop_square_around_active(img, threshold=8):
    gray = img.mean(axis=2)
    mask = gray > threshold
    bbox = _active_bbox(mask)
    if bbox is None:
        return img

    x0, y0, x1, y1 = bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = int(max(x1 - x0 + 1, y1 - y0 + 1))
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


def select_single_fisheye_lens(img, lens_mode="auto"):
    h, w = img.shape[:2]
    mode = lens_mode.lower()
    gray = img.mean(axis=2)
    mask = gray > 8

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
        return _crop_square_around_active(img), mode
    if mode == "left":
        return _crop_square_around_active(img[:, : w // 2]), mode
    if mode == "right":
        return _crop_square_around_active(img[:, w // 2 :]), mode
    if mode == "top":
        return _crop_square_around_active(img[: h // 2, :]), mode
    if mode == "bottom":
        return _crop_square_around_active(img[h // 2 :, :]), mode
    raise ValueError("lens mode geçersiz. full/left/right/top/bottom/auto kullan.")


def virtual_view_from_fisheye(
    fish_img,
    out_w=640,
    out_h=640,
    view_yaw_deg=0.0,
    view_pitch_deg=0.0,
    rect_fov_deg=90.0,
    fish_fov_deg=180.0,
):
    hs, ws = fish_img.shape[:2]
    cx_src = ws * 0.5
    cy_src = hs * 0.5
    radius_src = min(hs, ws) * 0.5

    rect_fov = np.deg2rad(rect_fov_deg)
    fish_fov = np.deg2rad(fish_fov_deg)
    theta_max = fish_fov * 0.5

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

    yaw = np.deg2rad(view_yaw_deg)
    pitch = np.deg2rad(view_pitch_deg)

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)

    x1 = cos_y * x + sin_y * z
    y1 = y
    z1 = -sin_y * x + cos_y * z

    x2 = x1
    y2 = cos_p * y1 - sin_p * z1
    z2 = sin_p * y1 + cos_p * z1

    theta = np.arccos(np.clip(z2, -1.0, 1.0))
    alpha = np.arctan2(y2, x2)
    r = (theta / theta_max) * radius_src

    xs = cx_src + r * np.cos(alpha)
    ys = cy_src + r * np.sin(alpha)

    valid = (theta <= theta_max)
    valid &= (xs >= 0) & (xs < ws - 1) & (ys >= 0) & (ys < hs - 1)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out[valid] = sample_bilinear(fish_img, xs[valid], ys[valid])
    return out, valid


def _panel_blend_weights(panel_w, overlap_px, is_first, is_last):
    weights = np.ones(panel_w, dtype=np.float32)
    if overlap_px > 0 and not is_first:
        weights[:overlap_px] = np.linspace(0.0, 1.0, overlap_px, dtype=np.float32)
    if overlap_px > 0 and not is_last:
        weights[-overlap_px:] = np.linspace(1.0, 0.0, overlap_px, dtype=np.float32)
    return weights


def multi_rectilinear_panorama(
    fish_img,
    out_w=2200,
    out_h=900,
    fish_fov_deg=180.0,
    pano_span_deg=160.0,
    pitch_deg=0.0,
    panel_count=6,
    overlap=0.35,
    panel_fov_deg=0.0,
    save_panels_dir=None,
):
    if panel_count < 2:
        raise ValueError("panel_count en az 2 olmalı.")
    if not (0.05 <= overlap < 0.8):
        raise ValueError("overlap 0.05 ile 0.8 arasında olmalı.")

    if panel_fov_deg <= 0:
        panel_fov_deg = pano_span_deg / (1.0 + (panel_count - 1) * (1.0 - overlap))

    step_yaw = panel_fov_deg * (1.0 - overlap)
    start_yaw = -0.5 * pano_span_deg + 0.5 * panel_fov_deg
    yaw_centers = [start_yaw + i * step_yaw for i in range(panel_count)]

    panel_w = int(round(out_w / (panel_count - (panel_count - 1) * overlap)))
    panel_w = max(panel_w, 64)
    overlap_px = int(round(panel_w * overlap))
    stride_px = panel_w - overlap_px
    stitched_w = panel_w + (panel_count - 1) * stride_px

    accum = np.zeros((out_h, stitched_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, stitched_w), dtype=np.float32)

    if save_panels_dir:
        os.makedirs(save_panels_dir, exist_ok=True)

    for i, yaw in enumerate(yaw_centers):
        panel, panel_valid = virtual_view_from_fisheye(
            fish_img,
            out_w=panel_w,
            out_h=out_h,
            view_yaw_deg=yaw,
            view_pitch_deg=pitch_deg,
            rect_fov_deg=panel_fov_deg,
            fish_fov_deg=fish_fov_deg,
        )

        if save_panels_dir:
            panel_path = os.path.join(save_panels_dir, f"panel_{i:02d}_yaw_{yaw:+.1f}.png")
            Image.fromarray(panel).save(panel_path)

        x0 = i * stride_px
        x1 = x0 + panel_w
        w1d = _panel_blend_weights(panel_w, overlap_px, i == 0, i == panel_count - 1)
        w2d = np.tile(w1d, (out_h, 1))
        w2d *= panel_valid.astype(np.float32)

        accum[:, x0:x1] += panel.astype(np.float32) * w2d[..., None]
        wsum[:, x0:x1] += w2d

    stitched = np.zeros_like(accum, dtype=np.uint8)
    valid = wsum > 1e-6
    stitched[valid] = (accum[valid] / wsum[valid, None]).astype(np.uint8)

    if stitched_w != out_w:
        start = max((stitched_w - out_w) // 2, 0)
        stitched = stitched[:, start:start + out_w]

    coverage_ratio = float((wsum > 1e-6).mean())
    return stitched, yaw_centers, panel_fov_deg, coverage_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="panoramik_duz.png")
    parser.add_argument("--fish-fov", type=float, default=180.0)
    parser.add_argument("--out-w", type=int, default=2200)
    parser.add_argument("--out-h", type=int, default=900)
    parser.add_argument("--span", type=float, default=160.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--overlap", type=float, default=0.35)
    parser.add_argument("--panel-fov", type=float, default=0.0)
    parser.add_argument("--save-panels-dir", default="")
    parser.add_argument("--lens", default="auto")
    args = parser.parse_args()

    raw_img = np.array(Image.open(args.input).convert("RGB"))
    fish_img, used_lens = select_single_fisheye_lens(raw_img, args.lens)
    pano, yaw_centers, used_panel_fov, coverage_ratio = multi_rectilinear_panorama(
        fish_img,
        out_w=args.out_w,
        out_h=args.out_h,
        fish_fov_deg=args.fish_fov,
        pano_span_deg=args.span,
        pitch_deg=args.pitch,
        panel_count=args.panels,
        overlap=args.overlap,
        panel_fov_deg=args.panel_fov,
        save_panels_dir=args.save_panels_dir if args.save_panels_dir else None,
    )

    Image.fromarray(pano).save(args.output)

    print("Tamamlandı")
    print("Çıktı:", os.path.abspath(args.output))
    print("Lens seçimi:", used_lens)
    print("İşlenen boyut:", f"{fish_img.shape[1]}x{fish_img.shape[0]}")
    print("Panel sayısı:", args.panels)
    print("Kullanılan panel FOV:", f"{used_panel_fov:.2f}°")
    print("Geçerli piksel oranı:", f"{coverage_ratio * 100:.2f}%")
    print("Yaw merkezleri:", ", ".join(f"{y:+.1f}" for y in yaw_centers))


if __name__ == "__main__":
    main()
