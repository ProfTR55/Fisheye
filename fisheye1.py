import os
import math
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Utils: Otsu threshold (no cv2)
# -----------------------------
def otsu_threshold(gray_u8: np.ndarray) -> int:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = gray_u8.size
    if total == 0:
        return 127
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return int(np.argmax(sigma_b2))


# -----------------------------
# Utils: Sobel (no cv2)
# -----------------------------
def sobel_grad(gray_f32: np.ndarray):
    g = gray_f32
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")

    gx = (
        (-1 * p[:-2, :-2]) + (1 * p[:-2, 2:]) +
        (-2 * p[1:-1, :-2]) + (2 * p[1:-1, 2:]) +
        (-1 * p[2:, :-2]) + (1 * p[2:, 2:])
    )

    gy = (
        (-1 * p[:-2, :-2]) + (-2 * p[:-2, 1:-1]) + (-1 * p[:-2, 2:]) +
        (1 * p[2:, :-2]) + (2 * p[2:, 1:-1]) + (1 * p[2:, 2:])
    )

    mag = np.sqrt(gx * gx + gy * gy) + 1e-12
    return gx.astype(np.float32), gy.astype(np.float32), mag.astype(np.float32)


# -----------------------------
# Fast box blur (vectorized integral image)
# -----------------------------
def box_blur(img_f32: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img_f32.astype(np.float32)
    r = int(radius)
    k = 2 * r + 1
    pad = np.pad(img_f32, ((r, r), (r, r)), mode="edge")
    ii = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    out = s / float(k * k)
    return out.astype(np.float32)


# -----------------------------
# Bilinear sampling (RGB)
# -----------------------------
def bilinear_sample_rgb(img_f32: np.ndarray, u: np.ndarray, v: np.ndarray, valid: np.ndarray):
    H, W = img_f32.shape[:2]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    u0c = np.clip(u0, 0, W - 1)
    u1c = np.clip(u1, 0, W - 1)
    v0c = np.clip(v0, 0, H - 1)
    v1c = np.clip(v1, 0, H - 1)

    Ia = img_f32[v0c, u0c]
    Ib = img_f32[v0c, u1c]
    Ic = img_f32[v1c, u0c]
    Id = img_f32[v1c, u1c]

    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)

    wa = (1 - du) * (1 - dv)
    wb = du * (1 - dv)
    wc = (1 - du) * dv
    wd = du * dv

    out = Ia * wa[..., None] + Ib * wb[..., None] + Ic * wc[..., None] + Id * wd[..., None]
    out[~valid] = 0.0
    return out


# -----------------------------
# Estimate fisheye circle (cx,cy,R)
# -----------------------------
def estimate_circle(gray_u8: np.ndarray):
    H, W = gray_u8.shape
    border = np.concatenate([gray_u8[0, :], gray_u8[-1, :], gray_u8[:, 0], gray_u8[:, -1]])
    bmed = float(np.median(border))
    gmed = float(np.median(gray_u8))

    th = otsu_threshold(gray_u8)

    if bmed < gmed:
        mask = gray_u8 > th
    else:
        mask = gray_u8 < th

    frac = mask.mean()
    if frac > 0.90 or frac < 0.05:
        cx, cy = W * 0.5, H * 0.5
        R = min(H, W) * 0.5
        return cx, cy, R, mask

    ys, xs = np.nonzero(mask)
    cx = float(xs.mean())
    cy = float(ys.mean())

    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    R = float(np.percentile(d, 99.0))
    R = min(R, min(H, W) * 0.5)
    return cx, cy, R, mask


def save_circle_debug(fish_u8, mask, cx, cy, R, outpath):
    from PIL import ImageDraw
    H, W = fish_u8.shape[:2]
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    im = Image.fromarray(fish_u8.copy())
    dr = ImageDraw.Draw(im)

    cx_i = int(round(cx))
    cy_i = int(round(cy))

    dr.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(0, 255, 0), width=4)

    r0 = 14
    dr.ellipse((cx_i - r0, cy_i - r0, cx_i + r0, cy_i + r0), fill=(255, 0, 0))
    L = 60
    dr.line((cx_i - L, cy_i, cx_i + L, cy_i), fill=(255, 0, 0), width=4)
    dr.line((cx_i, cy_i - L, cx_i, cy_i + L), fill=(255, 0, 0), width=4)

    dr.text((10, 10), f"cx={cx:.1f}, cy={cy:.1f}, R={R:.1f}", fill=(255, 255, 0))

    im.save(outpath)

    m = (mask.astype(np.uint8) * 255)
    Image.fromarray(m).save(outpath.replace(".png", "_mask.png"))

    half = 200
    y0 = max(0, cy_i - half); y1 = min(H, cy_i + half)
    x0 = max(0, cx_i - half); x1 = min(W, cx_i + half)
    crop = fish_u8[y0:y1, x0:x1].copy()
    Image.fromarray(crop).save(outpath.replace(".png", "_center_crop.png"))


# -----------------------------
# Ray models
# +X forward, +Y right, +Z up
# -----------------------------
def rays_rectilinear(out_w, out_h, hfov_deg, vfov_deg):
    xs = (np.arange(out_w) + 0.5) / out_w * 2 - 1
    ys = 1 - (np.arange(out_h) + 0.5) / out_h * 2
    X, Y = np.meshgrid(xs, ys)

    hx = math.tan(math.radians(hfov_deg) * 0.5)
    hy = math.tan(math.radians(vfov_deg) * 0.5)

    dx = np.ones_like(X, dtype=np.float32)
    dy = (X * hx).astype(np.float32)
    dz = (Y * hy).astype(np.float32)

    norm = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
    return dx / norm, dy / norm, dz / norm


def rays_cylindrical(out_w, out_h, hfov_deg, vfov_deg):
    xs = (np.arange(out_w) + 0.5) / out_w * 2 - 1
    ys = 1 - (np.arange(out_h) + 0.5) / out_h * 2
    X, Y = np.meshgrid(xs, ys)

    yaw = (X * math.radians(hfov_deg) * 0.5).astype(np.float32)

    vf = min(float(vfov_deg), 179.0)  # avoid tan(90)
    pitch = np.arctan((Y * math.tan(math.radians(vf) * 0.5)).astype(np.float32))

    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw);   sy = np.sin(yaw)

    dx = (cp * cy).astype(np.float32)
    dy = (cp * sy).astype(np.float32)
    dz = (sp).astype(np.float32)
    return dx, dy, dz


# -----------------------------
# Fisheye sampling (equidistant)
# -----------------------------
def sample_fisheye_equdist(fish_f32, cx, cy, R, fish_fov_deg, dx, dy, dz, flip_u=False, flip_v=False):
    theta_max = math.radians(fish_fov_deg) * 0.5
    f = (R / theta_max) if theta_max > 1e-9 else 1.0

    theta = np.arccos(np.clip(dx, -1.0, 1.0)).astype(np.float32)
    phi = np.arctan2(dz, dy).astype(np.float32)

    r = (f * theta).astype(np.float32)

    cu = -1.0 if flip_u else 1.0
    sv =  1.0 if flip_v else -1.0

    u = (cx + cu * r * np.cos(phi)).astype(np.float32)
    v = (cy + sv * r * np.sin(phi)).astype(np.float32)

    H, W = fish_f32.shape[:2]
    valid = (r <= R) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)

    return bilinear_sample_rgb(fish_f32, u, v, valid), valid, theta


# -----------------------------
# WARP: Rectilinear -> Cylindrical grid (GHOST FIX)
# -----------------------------
def warp_rect_to_cyl_grid(rect_img_f32: np.ndarray,
                          dx_c: np.ndarray, dy_c: np.ndarray, dz_c: np.ndarray,
                          rect_hfov: float, rect_vfov: float):
    H, W = rect_img_f32.shape[:2]
    eps = 1e-6

    hx = math.tan(math.radians(rect_hfov) * 0.5)
    hy = math.tan(math.radians(rect_vfov) * 0.5)

    x = (dy_c / np.maximum(dx_c, eps)) / hx
    y = (dz_c / np.maximum(dx_c, eps)) / hy

    u = (x + 1.0) * 0.5 * W - 0.5
    v = (1.0 - y) * 0.5 * H - 0.5

    valid = (dx_c > 0) & (x > -1.0) & (x < 1.0) & (y > -1.0) & (y < 1.0)
    warped = bilinear_sample_rgb(rect_img_f32, u.astype(np.float32), v.astype(np.float32), valid)
    return warped, valid


# -----------------------------
# FOV estimation
# -----------------------------
def orientation_continuity_score(pano_rgb_u8: np.ndarray) -> float:
    gray = (0.299*pano_rgb_u8[..., 0] + 0.587*pano_rgb_u8[..., 1] + 0.114*pano_rgb_u8[..., 2]).astype(np.float32) / 255.0
    gx, gy, mag = sobel_grad(gray)

    m = mag / (mag.max() + 1e-12)
    mu8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    th = otsu_threshold(mu8)
    edge = mu8 >= th

    edge_count = int(edge.sum())
    if edge_count < (pano_rgb_u8.shape[0] * pano_rgb_u8.shape[1] * 0.003):
        return 1e6

    ang = np.arctan2(gy, gx).astype(np.float32)
    c2 = np.cos(2 * ang).astype(np.float32)
    s2 = np.sin(2 * ang).astype(np.float32)

    dc2x = np.abs(c2[:, 1:] - c2[:, :-1])
    dc2y = np.abs(c2[1:, :] - c2[:-1, :])
    ds2x = np.abs(s2[:, 1:] - s2[:, :-1])
    ds2y = np.abs(s2[1:, :] - s2[:-1, :])

    var_map = np.zeros_like(c2, dtype=np.float32)
    var_map[:, :-1] += dc2x + ds2x
    var_map[:-1, :] += dc2y + ds2y
    return float(var_map[edge].mean())


def estimate_fov(
    fish_f32, cx, cy, R,
    fov_min=150, fov_max=220, step=5,
    probe_w=640, probe_h=320,
    out_hfov=180, out_vfov=140,
    flip_u=False, flip_v=False
):
    candidates = list(range(int(fov_min), int(fov_max) + 1, int(step)))
    scores = []

    dx, dy, dz = rays_cylindrical(probe_w, probe_h, out_hfov, out_vfov)

    for fov in candidates:
        pano_f32, _, _ = sample_fisheye_equdist(
            fish_f32, cx, cy, R, fov, dx, dy, dz, flip_u=flip_u, flip_v=flip_v
        )
        pano_u8 = np.clip(pano_f32 * 255.0, 0, 255).astype(np.uint8)
        scores.append(orientation_continuity_score(pano_u8))

    best_idx = int(np.argmin(scores))
    return candidates[best_idx], np.array(candidates), np.array(scores, dtype=np.float32)


# -----------------------------
# Hybrid panorama (GHOST-FREE)
# Output grid = cylindrical
# Rectilinear content is WARPED onto cylindrical grid before blending
# -----------------------------
def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def hybrid_panorama(
    fish_f32, cx, cy, R, fish_fov_deg,
    out_w=2048, out_h=1024,
    out_hfov=180, out_vfov=140,
    rect_hfov=120, rect_vfov=90,
    theta1_deg=35, theta2_deg=70,
    beta_content=1.0,
    blur_radius=8,
    flip_u=False, flip_v=False
):
    # 1) Cylindrical render (this is the reference grid)
    dx_c, dy_c, dz_c = rays_cylindrical(out_w, out_h, out_hfov, out_vfov)
    pano_c_f32, valid_c, theta_c = sample_fisheye_equdist(
        fish_f32, cx, cy, R, fish_fov_deg, dx_c, dy_c, dz_c, flip_u=flip_u, flip_v=flip_v
    )

    # 2) Rectilinear render (in its own grid)
    dx_r, dy_r, dz_r = rays_rectilinear(out_w, out_h, rect_hfov, rect_vfov)
    pano_r_f32, valid_r, _ = sample_fisheye_equdist(
        fish_f32, cx, cy, R, fish_fov_deg, dx_r, dy_r, dz_r, flip_u=flip_u, flip_v=flip_v
    )

    # 3) Warp rectilinear image onto cylindrical grid (GHOST FIX)
    pano_r_on_cyl, valid_rw = warp_rect_to_cyl_grid(pano_r_f32, dx_c, dy_c, dz_c, rect_hfov, rect_vfov)

    # 4) Angular weight based on theta from cylindrical rays
    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)
    w_angle = 1.0 - smoothstep(t1, t2, theta_c)  # center=1 -> rect, edge=0 -> cyl

    # 5) Content weight (optional): edges in cylindrical pano
    pano_c_u8 = np.clip(pano_c_f32 * 255.0, 0, 255).astype(np.uint8)
    gray = (0.299*pano_c_u8[..., 0] + 0.587*pano_c_u8[..., 1] + 0.114*pano_c_u8[..., 2]).astype(np.float32) / 255.0
    _, _, mag = sobel_grad(gray)
    m = mag / (mag.max() + 1e-12)
    mu8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    th = otsu_threshold(mu8)
    edge = (mu8 >= th).astype(np.float32)

    edge_blur = box_blur(edge, blur_radius)
    edge_blur = edge_blur / (edge_blur.max() + 1e-12)

    # 6) Combine weights
    w_rect = np.clip(w_angle * (1.0 + beta_content * edge_blur), 0.0, 1.0).astype(np.float32)

    # Validity: rect only where warped-rect is valid, cyl where cyl valid
    w_rect = w_rect * valid_rw.astype(np.float32)
    w_cyl = (1.0 - w_rect) * valid_c.astype(np.float32)

    # Normalize
    wsum = w_rect + w_cyl
    wsum[wsum < 1e-6] = 1.0
    w_rect /= wsum
    w_cyl  /= wsum

    # 7) Blend in the SAME GRID => no ghosting
    out = pano_r_on_cyl * w_rect[..., None] + pano_c_f32 * w_cyl[..., None]
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    return out, pano_c_f32, pano_r_on_cyl, w_rect, edge_blur


# -----------------------------
# Main CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="fisheye image path")
    ap.add_argument("--outdir", default="out_steps", help="output directory")
    ap.add_argument("--out_w", type=int, default=2048)
    ap.add_argument("--out_h", type=int, default=1024)

    ap.add_argument("--fov_min", type=int, default=150)
    ap.add_argument("--fov_max", type=int, default=220)
    ap.add_argument("--fov_step", type=int, default=5)

    ap.add_argument("--probe_w", type=int, default=640)
    ap.add_argument("--probe_h", type=int, default=320)

    ap.add_argument("--out_hfov", type=float, default=180.0)
    ap.add_argument("--out_vfov", type=float, default=140.0, help="keep < 180 (e.g. 120-160)")

    ap.add_argument("--mode", choices=["step2", "hybrid"], default="step2",
                    help="step2 = cylindrical baseline; hybrid = ghost-free hybrid blending")

    ap.add_argument("--flip_u", action="store_true")
    ap.add_argument("--flip_v", action="store_true")

    ap.add_argument("--rect_hfov", type=float, default=120.0)
    ap.add_argument("--rect_vfov", type=float, default=90.0)
    ap.add_argument("--theta1_deg", type=float, default=35.0)
    ap.add_argument("--theta2_deg", type=float, default=70.0)
    ap.add_argument("--beta_content", type=float, default=1.0)
    ap.add_argument("--blur_radius", type=int, default=8)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    img = Image.open(args.input).convert("RGB")
    fish_u8 = np.asarray(img)
    fish_f32 = fish_u8.astype(np.float32) / 255.0

    gray_u8 = (0.299*fish_u8[..., 0] + 0.587*fish_u8[..., 1] + 0.114*fish_u8[..., 2]).astype(np.uint8)
    cx, cy, R, mask = estimate_circle(gray_u8)
    print(f"[circle] cx={cx:.2f}, cy={cy:.2f}, R={R:.2f}")

    save_circle_debug(fish_u8, mask, cx, cy, R, os.path.join(args.outdir, "STEP1_debug_circle.png"))

    best_fov, fovs, scores = estimate_fov(
        fish_f32, cx, cy, R,
        fov_min=args.fov_min, fov_max=args.fov_max, step=args.fov_step,
        probe_w=args.probe_w, probe_h=args.probe_h,
        out_hfov=args.out_hfov, out_vfov=args.out_vfov,
        flip_u=args.flip_u, flip_v=args.flip_v
    )
    print(f"[fov] best_fov={best_fov} deg")

    plt.figure()
    plt.plot(fovs, scores)
    plt.xlabel("FOV (deg)")
    plt.ylabel("Orientation continuity score (lower=better)")
    plt.title(f"Best FOV = {best_fov} deg (probe_vfov={args.out_vfov})")
    plt.grid(True)
    plt.savefig(os.path.join(args.outdir, "fov_sweep.png"), dpi=150)
    plt.close()

    if args.mode == "step2":
        dx_c, dy_c, dz_c = rays_cylindrical(args.out_w, args.out_h, args.out_hfov, args.out_vfov)
        pano_c_f32, valid_c, theta_c = sample_fisheye_equdist(
            fish_f32, cx, cy, R, best_fov, dx_c, dy_c, dz_c, flip_u=args.flip_u, flip_v=args.flip_v
        )

        Image.fromarray(np.clip(pano_c_f32 * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.outdir, "STEP2_cyl.png"))
        Image.fromarray((valid_c.astype(np.uint8) * 255)).save(os.path.join(args.outdir, "STEP2_valid_cyl.png"))

        t = theta_c / (np.max(theta_c) + 1e-12)
        Image.fromarray(np.clip(t * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.outdir, "STEP2_theta.png"))

        print(f"[done] STEP1+STEP2 saved to: {args.outdir}")
        return

    out, pano_c, pano_r_on_cyl, w_rect, edge_blur = hybrid_panorama(
        fish_f32, cx, cy, R, best_fov,
        out_w=args.out_w, out_h=args.out_h,
        out_hfov=args.out_hfov, out_vfov=args.out_vfov,
        rect_hfov=args.rect_hfov, rect_vfov=args.rect_vfov,
        theta1_deg=args.theta1_deg, theta2_deg=args.theta2_deg,
        beta_content=args.beta_content, blur_radius=args.blur_radius,
        flip_u=args.flip_u, flip_v=args.flip_v
    )

    Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.outdir, "hybrid.png"))
    Image.fromarray(np.clip(pano_c * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.outdir, "cylindrical.png"))
    Image.fromarray(np.clip(pano_r_on_cyl * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.outdir, "rect_on_cyl.png"))

    Image.fromarray(np.clip(w_rect * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.outdir, "w_rect.png"))
    Image.fromarray(np.clip(edge_blur * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.outdir, "edge_weight.png"))

    print(f"[done] outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
