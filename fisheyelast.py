import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ==========================================================
# BILINEAR SAMPLING
# ==========================================================
def sample_bilinear(img, x, y):
    H, W = img.shape[:2]

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    return (Ia * wa[..., None] +
            Ib * wb[..., None] +
            Ic * wc[..., None] +
            Id * wd[..., None]).astype(np.uint8)


# ==========================================================
# SANAL PINHOLE (RECTILINEAR)  (senin fonksiyonun)
# ==========================================================
def virtual_view_from_fisheye(
    fish_img,
    out_w=480,
    out_h=480,
    view_yaw_deg=0.0,
    view_pitch_deg=0.0,
    rect_fov_deg=90.0,
    fish_fov_deg=180.0,
):
    Hs, Ws = fish_img.shape[:2]
    cx_src = Ws * 0.5
    cy_src = Hs * 0.5
    R_src = min(Hs, Ws) * 0.5

    rect_fov = np.deg2rad(rect_fov_deg)
    fish_fov = np.deg2rad(fish_fov_deg)

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

    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # yaw rot
    X = cyaw * x + syaw * z
    Y = y
    Z = -syaw * x + cyaw * z

    # pitch rot (pozitif pitch: yukarı bakar)
    X2 = X
    Y2 = cp * Y - sp * Z
    Z2 = sp * Y + cp * Z

    theta = np.arccos(np.clip(Z2, -1.0, 1.0))
    phi = np.arctan2(Y2, X2)

    theta_max = fish_fov * 0.5
    r = (theta / theta_max) * R_src  # equidistant varsayım

    xs = cx_src + r * np.cos(phi)
    ys = cy_src + r * np.sin(phi)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = (xs >= 0) & (xs < Ws - 1) & (ys >= 0) & (ys < Hs - 1) & (theta <= theta_max)
    out[mask] = sample_bilinear(fish_img, xs[mask], ys[mask])

    return out


# ==========================================================
# 1) EQUIRECTANGULAR (tek görüntü, tekrar yok, maksimum kapsama)
#    ama çizgiler her yerde dümdüz olmaz (sferik proj.)
# ==========================================================
def fisheye_to_equirect_hemisphere(
    fish_img,
    out_w=1400,
    out_h=700,
    fish_fov_deg=180.0,
    yaw_span_deg=180.0,   # 180 fisheye için yaw: -90..+90
    pitch_span_deg=180.0  # pitch: -90..+90
):
    Hs, Ws = fish_img.shape[:2]
    cx_src = Ws * 0.5
    cy_src = Hs * 0.5
    R_src  = min(Hs, Ws) * 0.5

    fish_fov = np.deg2rad(fish_fov_deg)
    theta_max = fish_fov * 0.5

    yaw_span   = np.deg2rad(yaw_span_deg)
    pitch_span = np.deg2rad(pitch_span_deg)

    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))

    # yaw: -span/2 .. +span/2
    yaw = (u / (out_w - 1) - 0.5) * yaw_span

    # pitch: -span/2 .. +span/2  (yukarı negatif)
    pitch = (v / (out_h - 1) - 0.5) * pitch_span

    # direction from yaw/pitch (senin yaw=atan2(X,Z), pitch=asin(Y) tanımına uygun)
    X = np.sin(yaw) * np.cos(pitch)
    Y = np.sin(pitch)
    Z = np.cos(yaw) * np.cos(pitch)

    theta = np.arccos(np.clip(Z, -1.0, 1.0))
    phi   = np.arctan2(Y, X)

    r = (theta / theta_max) * R_src
    xs = cx_src + r * np.cos(phi)
    ys = cy_src + r * np.sin(phi)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = (
        (theta <= theta_max) &
        (xs >= 0) & (xs < Ws - 1) &
        (ys >= 0) & (ys < Hs - 1)
    )
    out[mask] = sample_bilinear(fish_img, xs[mask], ys[mask])
    return out


# ==========================================================
# 2) CUBEMAP CROSS (tek görsel dosyası, her yüzde çizgiler dümdüz)
#    180 fisheye hemisphere için 5 yüz yeter: front/left/right/up/down
# ==========================================================
def cubemap_cross_from_fisheye(
    fish_img,
    face=512,
    fish_fov_deg=180.0
):
    # her yüz rectilinear 90°
    fov = 90.0

    # Not: bu pitch tanımı virtual_view fonksiyonundakiyle uyumlu:
    # +90 -> yukarı bakar, -90 -> aşağı bakar
    faces = {
        "front": virtual_view_from_fisheye(fish_img, face, face, 0,   0,   fov, fish_fov_deg),
        "left":  virtual_view_from_fisheye(fish_img, face, face, -90, 0,   fov, fish_fov_deg),
        "right": virtual_view_from_fisheye(fish_img, face, face, 90,  0,   fov, fish_fov_deg),
        "up":    virtual_view_from_fisheye(fish_img, face, face, 0,   90,  fov, fish_fov_deg),
        "down":  virtual_view_from_fisheye(fish_img, face, face, 0,  -90,  fov, fish_fov_deg),
    }

    # 3x3 cross canvas
    H = face * 3
    W = face * 3
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # yerleşim:
    #        [   up   ]
    # [left][ front ][right]
    #        [  down  ]
    canvas[0:face, face:2*face] = faces["up"]
    canvas[face:2*face, 0:face] = faces["left"]
    canvas[face:2*face, face:2*face] = faces["front"]
    canvas[face:2*face, 2*face:3*face] = faces["right"]
    canvas[2*face:3*face, face:2*face] = faces["down"]

    return canvas, faces


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    fish_path = r"C:/Users/Doğukan/Desktop/fisheye/fisheye.jpg"
    fish_img = np.array(Image.open(fish_path).convert("RGB"))

    save_dir = r"C:/Users/Doğukan/Desktop/fisheye/out_single"
    os.makedirs(save_dir, exist_ok=True)

    # --- A) Equirectangular (tek görüntü, tekrar yok, tam kapsama) ---
    eq = fisheye_to_equirect_hemisphere(
        fish_img,
        out_w=1400,
        out_h=700,
        fish_fov_deg=180,
        yaw_span_deg=180,
        pitch_span_deg=180
    )
    eq_path = os.path.join(save_dir, "EQUIRECT_HEMISPHERE.png")
    Image.fromarray(eq).save(eq_path)
    print("Saved:", eq_path)

    # --- B) Cubemap cross (tek görsel; yüzlerde çizgiler dümdüz) ---
    cross, faces = cubemap_cross_from_fisheye(fish_img, face=512, fish_fov_deg=180)
    cross_path = os.path.join(save_dir, "CUBEMAP_CROSS_5FACES.png")
    Image.fromarray(cross).save(cross_path)
    print("Saved:", cross_path)

    # quick preview
    plt.figure(figsize=(14, 6))
    plt.imshow(eq)
    plt.title("Equirectangular (Hemisphere) - tek görüntü, tekrar yok")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(cross)
    plt.title("Cubemap Cross (5 faces) - yüzlerde dümdüz çizgiler")
    plt.axis("off")
    plt.show()
