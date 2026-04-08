import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================================
# FISHEYE PIXEL -> (yaw, pitch) 
# ==========================================================
def fisheye_pixel_to_angles(x, y, img_shape, fish_fov_deg=180.0):
    H, W = img_shape[:2]
    cx = W * 0.5
    cy = H * 0.5
    R  = min(H, W) * 0.5

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx * dx + dy * dy)

    theta_max = np.deg2rad(fish_fov_deg * 0.5)
    theta = (r / R) * theta_max
    phi = np.arctan2(dy, dx)

    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)

    yaw = np.rad2deg(np.arctan2(X, Z))
    pitch = np.rad2deg(np.arcsin(Y))
    return yaw, pitch


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


def unwrap_fisheye_to_strip(
    fish_img,
    out_w=1200,
    out_h=400,
    fish_fov_deg=180.0
):
    H, W = fish_img.shape[:2]
    cx, cy = W * 0.5, H * 0.5
    R = min(H, W) * 0.5

    theta_max = np.deg2rad(fish_fov_deg * 0.5)

    u = np.linspace(0, 1, out_w, endpoint=False)
    v = np.linspace(0, 1, out_h, endpoint=False)
    uu, vv = np.meshgrid(u, v)

    phi = uu * (2.0 * np.pi)
    theta = vv * theta_max

    r = (theta / theta_max) * R

    xs = cx + r * np.cos(phi)
    ys = cy + r * np.sin(phi)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
    out[mask] = sample_bilinear(fish_img, xs[mask], ys[mask])

    return out, xs, ys


# ==========================================================
# SANAL PINHOLE (RECTILINEAR)
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

    # yaw
    X = cyaw * x + syaw * z
    Y = y
    Z = -syaw * x + cyaw * z

    # pitch
    X2 = X
    Y2 = cp * Y - sp * Z
    Z2 = sp * Y + cp * Z

    theta = np.arccos(np.clip(Z2, -1.0, 1.0))
    phi = np.arctan2(Y2, X2)

    theta_max = fish_fov * 0.5
    r = (theta / theta_max) * R_src

    xs = cx_src + r * np.cos(phi)
    ys = cy_src + r * np.sin(phi)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = (xs >= 0) & (xs < Ws - 1) & (ys >= 0) & (ys < Hs - 1)
    out[mask] = sample_bilinear(fish_img, xs[mask], ys[mask])

    return out, xs, ys


# ==========================================================
# PAGED UI
# ==========================================================
def paged_views_ui(fish_img, results, show_keys, cols=4, rows=3, dist_thresh=8.0):
    """
    results[name] = (img, xs, ys)
    show_keys: view isimlerinin sıralı listesi
    """
    N = len(show_keys)
    page_size = cols * rows
    num_pages = max(1, math.ceil(N / page_size))
    page_idx = 0

    # last_uv[name] = (u,v) ya da None
    last_uv = {k: None for k in show_keys}

    fig = plt.figure(figsize=(4 + 3.2*cols, 2.7*rows))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax_fish = fig.add_subplot(gs[0, 0])
    ax_fish.imshow(fish_img)
    ax_fish.set_title("Fisheye (click) | ←/→ veya A/D sayfa")
    ax_fish.axis("off")
    fish_dot, = ax_fish.plot([], [], "ro", markersize=6)

    # slot eksenleri (sayfada görünen kutular)
    slot_axes = []
    slot_names = [None] * page_size
    slot_dots = [None] * page_size

    slot_axes = []
    slot_names = [None]
    slot_dots = [None]

    ax = fig.add_subplot(gs[0, 1])   # unwrap
    ax.axis("off")
    slot_axes.append(ax)

    suptitle = fig.suptitle("", fontsize=12)

    def set_slot(i, name_or_none):
        ax = slot_axes[i]
        ax.clear()
        ax.axis("off")
        slot_names[i] = name_or_none

        if name_or_none is None:
            slot_dots[i] = None
            return

        img, xs, ys = results[name_or_none]
        ax.imshow(img)
        ax.set_title(name_or_none, fontsize=9)
        ax.axis("off")
        dot, = ax.plot([], [], "ro", markersize=4)
        slot_dots[i] = dot

        # sayfa yenilenince last_uv varsa göster
        uv = last_uv.get(name_or_none, None)
        if uv is not None:
            u, v = uv
            dot.set_data([u], [v])

    def refresh_page():
        nonlocal page_idx
        page_idx = max(0, min(page_idx, num_pages - 1))

        start = page_idx * page_size
        end = min(start + page_size, N)
        page_keys = show_keys[start:end]

        for i in range(page_size):
            if i < len(page_keys):
                set_slot(i, page_keys[i])
            else:
                set_slot(i, None)

        suptitle.set_text(f"Sayfa {page_idx+1}/{num_pages} | Toplam view: {N} | (←/→ veya A/D)")

        fig.canvas.draw_idle()

    def compute_all_uv_for_click(x, y):
        # tüm viewlar için (u,v) hesapla
        for k in show_keys:
            img, xs, ys = results[k]
            d = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
            idx = np.unravel_index(np.argmin(d), d.shape)

            if d[idx] < dist_thresh:
                u, v = idx[1], idx[0]
                last_uv[k] = (float(u), float(v))
            else:
                last_uv[k] = None

    def on_click(event):
        if event.inaxes != ax_fish:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)
        fish_dot.set_data([x], [y])

        yaw, pitch = fisheye_pixel_to_angles(x, y, fish_img.shape)
        print("\nClick:", (x, y), " -> yaw:", f"{yaw:+.2f}", "pitch:", f"{pitch:+.2f}")

        compute_all_uv_for_click(x, y)
        refresh_page()

    def on_key(event):
        nonlocal page_idx
        if event.key in ["right", "d", "pagedown", " "]:
            page_idx += 1
            refresh_page()
        elif event.key in ["left", "a", "pageup", "backspace"]:
            page_idx -= 1
            refresh_page()
        elif event.key in ["home"]:
            page_idx = 0
            refresh_page()
        elif event.key in ["end"]:
            page_idx = num_pages - 1
            refresh_page()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    refresh_page()
    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    fish_path = r"C:/Users/Doğukan/Desktop/fisheye/fisheye9.jpg"
    fish_img = np.array(Image.open(fish_path).convert("RGB"))

    save_dir = r"C:/Users/Doğukan/Desktop/fisheye/virtual_views_21"
    os.makedirs(save_dir, exist_ok=True)

    # 21 view (7 yaw x 3 pitch)
    yaws = [-135, -90, -45, 0, 45, 90, 135]
    pitchs = [45, 0, -45]

    views = []
    for p in pitchs:
        for y in yaws:
            name = f"Y{y:+}_P{p:+}"
            views.append((name, y, p))

    # ======================================================
    # SADECE UI ICIN UNWRAP
    # ======================================================
    results = {}

    unwrap_img, unwrap_xs, unwrap_ys = unwrap_fisheye_to_strip(
        fish_img,
        out_w=2400,
        out_h=800,
        fish_fov_deg=180
    )

    results["UNWRAP"] = (unwrap_img, unwrap_xs, unwrap_ys)

    # ======================================================
    # 21 SANAL PINHOLE VIEW (SADECE DISKE KAYIT)
    # ======================================================
    for name, yaw, pitch in views:
        img, xs, ys = virtual_view_from_fisheye(
            fish_img,
            out_w=480,
            out_h=480,
            view_yaw_deg=yaw,
            view_pitch_deg=pitch,
            rect_fov_deg=90,
            fish_fov_deg=180
        )

        out_path = os.path.join(save_dir, f"{name}.png")
        Image.fromarray(img).save(out_path)

    print(f"\nKaydedildi: {len(views)} view")
    print("Klasör:", save_dir)

    # ======================================================
    # UI: SADECE UNWRAP GOSTER
    # ======================================================
    show_keys = ["UNWRAP"]
    paged_views_ui(
        fish_img,
        results,
        show_keys,
        cols=1,
        rows=1,
        dist_thresh=8.0
    )
