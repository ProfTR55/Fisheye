import sys
import math
import numpy as np
from PIL import Image


# ==========================================================
# Bilinear sampling
# ==========================================================
def bilinear_sample(img, x, y):
    H, W = img.shape[:2]

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return None

    dx = x - x0
    dy = y - y0

    p00 = img[y0, x0].astype(np.float32)
    p10 = img[y0, x1].astype(np.float32)
    p01 = img[y1, x0].astype(np.float32)
    p11 = img[y1, x1].astype(np.float32)

    p0 = p00 * (1 - dx) + p10 * dx
    p1 = p01 * (1 - dx) + p11 * dx
    p  = p0 * (1 - dy) + p1 * dy

    return p.astype(np.uint8)


# ==========================================================
# Fisheye -> Equirectangular Panorama
# - 180 deg fisheye
# - Equidistant lens model
# ==========================================================
def fisheye_to_equirect(
    fish_img,
    out_w=2048,
    out_h=1024,
    fov_deg=180.0
):
    H, W = fish_img.shape[:2]

    # fisheye center
    cx = W * 0.5
    cy = H * 0.5

    # fisheye radius
    R = min(W, H) * 0.5

    # focal length (equidistant)
    theta_max = math.radians(fov_deg * 0.5)
    f = R / theta_max

    pano = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for y in range(out_h):
        # pitch: top -> bottom
        pitch = (y / out_h) * math.pi - math.pi / 2

        for x in range(out_w):
            # yaw: left -> right
            yaw = (x / out_w) * 2 * math.pi - math.pi

            # direction vector
            vx = math.cos(pitch) * math.sin(yaw)
            vy = math.sin(pitch)
            vz = math.cos(pitch) * math.cos(yaw)

            # fisheye angles
            theta = math.acos(vz)
            if theta > theta_max:
                continue

            phi = math.atan2(vy, vx)

            # equidistant projection
            r = f * theta
            u = cx + r * math.cos(phi)
            v = cy + r * math.sin(phi)

            color = bilinear_sample(fish_img, u, v)
            if color is not None:
                pano[y, x] = color

    return pano


# ==========================================================
# MAIN
# ==========================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python fisheye_to_pano.py input.jpg output.png")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    fish = Image.open(input_path).convert("RGB")
    fish_np = np.array(fish)

    pano = fisheye_to_equirect(
        fish_np,
        out_w=4096,
        out_h=2048,
        fov_deg=180.0
    )

    Image.fromarray(pano).save(output_path)
    print("Panorama saved:", output_path)


if __name__ == "__main__":
    main()
