import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================================
# BILINEAR SAMPLING (OpenCV yok)
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
# TEK SANAL PINHOLE KAMERA GÖRÜNÜMÜ
# ==========================================================

def virtual_view_from_fisheye(
    fish_img,
    out_w=480,
    out_h=480,
    view_yaw_deg=0.0,
    view_pitch_deg=0.0,
    rect_fov_deg=110.0,
    fish_fov_deg=180.0,
    cx_src=None,
    cy_src=None
):
    Hs, Ws = fish_img.shape[:2]
    if cx_src is None: cx_src = Ws * 0.5
    if cy_src is None: cy_src = Hs * 0.5
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

    norm = np.sqrt(x*x + y*y + z*z)
    x /= norm; y /= norm; z /= norm

    yaw = np.deg2rad(view_yaw_deg)
    pitch = np.deg2rad(view_pitch_deg)

    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    X =  cyaw * x + syaw * z
    Y =  y
    Z = -syaw * x + cyaw * z

    X2 = X
    Y2 =  cp * Y - sp * Z
    Z2 =  sp * Y + cp * Z

    theta = np.arccos(np.clip(Z2, -1.0, 1.0))
    phi = np.arctan2(Y2, X2)

    theta_max = fish_fov * 0.5
    r = (theta / theta_max) * R_src

    xs = cx_src + r * np.cos(phi)
    ys = cy_src + r * np.sin(phi)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = (xs >= 0) & (xs < Ws-1) & (ys >= 0) & (ys < Hs-1)
    out[mask] = sample_bilinear(fish_img, xs[mask], ys[mask])

    return out, xs, ys

# ==========================================================
# MAIN + UI
# ==========================================================
if __name__ == "__main__":
    fish_img = np.array(Image.open("C:/Users/Doğukan/Desktop/fisheye/fisheye1.jpg").convert("RGB"))

    views = [
    ("FRONT",  0,   0),
    ("RIGHT",  90,  0),
    ("BACK",  180,  0),
    ("LEFT",  -90,  0),
    ("UP",     0,  60),
    ("DOWN",   0, -60),
    ]

    results = {}
    for name, yaw in views:
        img, xs, ys = virtual_view_from_fisheye(
            fish_img,
            view_yaw_deg=yaw,
            rect_fov_deg=90
        )
        results[name] = (img, xs, ys)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3)

    ax_fish = fig.add_subplot(gs[:, 0])
    ax_views = {
        "FRONT": fig.add_subplot(gs[0, 1]),
        "RIGHT": fig.add_subplot(gs[0, 2]),
        "BACK":  fig.add_subplot(gs[1, 1]),
        "LEFT":  fig.add_subplot(gs[1, 2]),
    }

    ax_fish.imshow(fish_img)
    ax_fish.set_title("Fisheye")
    ax_fish.axis("off")

    fish_dot, = ax_fish.plot([], [], "ro")

    view_dots = {}
    for name, ax in ax_views.items():
        ax.imshow(results[name][0])
        ax.set_title(name)
        ax.axis("off")
        view_dots[name] = ax.plot([], [], "ro")[0]

    def on_click(event):
        if event.inaxes != ax_fish:
            return
        x, y = int(event.xdata), int(event.ydata)
        fish_dot.set_data(x, y)

        for name, (img, xs, ys) in results.items():
            d = np.sqrt((xs - x)**2 + (ys - y)**2)
            idx = np.unravel_index(np.argmin(d), d.shape)
            if d[idx] < 2.0:
                view_dots[name].set_data(idx[1], idx[0])
            else:
                view_dots[name].set_data([], [])

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
