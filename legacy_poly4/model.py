"""Camera model and projection helpers for calibration-first fisheye rectification.

The model uses a generic radial polynomial:
    theta = g(r_hat) = a1*r + a2*r^2 + a3*r^3 + a4*r^4
where r_hat is the normalized image radius in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


EPS = 1e-9


@dataclass
class FisheyePoly4Model:
    """Flexible fisheye model parameterization.

    Attributes:
        cx, cy: Distortion center in pixel coordinates.
        coeffs: Polynomial coefficients [a1, a2, a3, a4].
        sx, sy: Anisotropic radius scales. sx=sy=1 reduces to isotropic model.
        p1, p2: Tangential (decentering) distortion coefficients.
    """

    cx: float
    cy: float
    coeffs: np.ndarray
    sx: float = 1.0
    sy: float = 1.0
    p1: float = 0.0
    p2: float = 0.0

    def __post_init__(self) -> None:
        self.coeffs = np.asarray(self.coeffs, dtype=np.float64).reshape(4)
        self.sx = float(self.sx)
        self.sy = float(self.sy)
        self.p1 = float(self.p1)
        self.p2 = float(self.p2)
        if self.sx <= 0.0 or self.sy <= 0.0:
            raise ValueError("sx and sy must be positive.")

    @staticmethod
    def initial_from_shape(
        image_shape: Tuple[int, int], max_half_angle_deg: float = 105.0
    ) -> "FisheyePoly4Model":
        """Practical initialization:
        - center at image center
        - mostly linear radial mapping
        """
        h, w = image_shape[:2]
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)
        theta_max = np.deg2rad(max_half_angle_deg)
        coeffs = np.array([theta_max, 0.0, 0.0, 0.0], dtype=np.float64)
        return FisheyePoly4Model(
            cx=cx, cy=cy, coeffs=coeffs, sx=1.0, sy=1.0, p1=0.0, p2=0.0
        )

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.cx,
                self.cy,
                *self.coeffs.tolist(),
                self.sx,
                self.sy,
                self.p1,
                self.p2,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def from_vector(vec: np.ndarray) -> "FisheyePoly4Model":
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        if vec.size != 10:
            raise ValueError(f"Expected parameter vector of size 10, got {vec.size}")
        return FisheyePoly4Model(
            cx=float(vec[0]),
            cy=float(vec[1]),
            coeffs=vec[2:6],
            sx=float(vec[6]),
            sy=float(vec[7]),
            p1=float(vec[8]),
            p2=float(vec[9]),
        )

    def _scaled_components(self, points_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
        dx = (pts[:, 0] - self.cx) / max(self.sx, EPS)
        dy = (pts[:, 1] - self.cy) / max(self.sy, EPS)
        return dx, dy

    def _rho_scale_raw(self, image_shape: Tuple[int, int]) -> float:
        """Scale used to normalize tangential model domain for numeric stability."""
        h, w = image_shape[:2]
        corners = np.array(
            [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]],
            dtype=np.float64,
        )
        dx = (corners[:, 0] - self.cx) / max(self.sx, EPS)
        dy = (corners[:, 1] - self.cy) / max(self.sy, EPS)
        return float(np.max(np.hypot(dx, dy)) + EPS)

    def distort_tangential_components(
        self, xu: np.ndarray, yu: np.ndarray, image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply tangential distortion in scaled-coordinate domain."""
        x = np.asarray(xu, dtype=np.float64)
        y = np.asarray(yu, dtype=np.float64)
        s = self._rho_scale_raw(image_shape)
        xn = x / s
        yn = y / s
        r2 = xn * xn + yn * yn
        dxn = 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn)
        dyn = self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn
        xd = (xn + dxn) * s
        yd = (yn + dyn) * s
        return xd, yd

    def undistort_tangential_components(
        self,
        xd: np.ndarray,
        yd: np.ndarray,
        image_shape: Tuple[int, int],
        num_iters: int = 7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Invert tangential distortion using fixed-point iterations."""
        x = np.asarray(xd, dtype=np.float64)
        y = np.asarray(yd, dtype=np.float64)
        s = self._rho_scale_raw(image_shape)
        xdn = x / s
        ydn = y / s
        xu = xdn.copy()
        yu = ydn.copy()
        for _ in range(max(1, int(num_iters))):
            r2 = xu * xu + yu * yu
            dxn = 2.0 * self.p1 * xu * yu + self.p2 * (r2 + 2.0 * xu * xu)
            dyn = self.p1 * (r2 + 2.0 * yu * yu) + 2.0 * self.p2 * xu * yu
            xu = xdn - dxn
            yu = ydn - dyn
            xu = np.clip(xu, -4.0, 4.0)
            yu = np.clip(yu, -4.0, 4.0)
        return xu * s, yu * s

    def normalized_radius(
        self, points_uv: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        xd, yd = self._scaled_components(points_uv)
        xu, yu = self.undistort_tangential_components(xd, yd, image_shape=image_shape)
        rho = np.hypot(xu, yu)
        return np.clip(rho / self.rho_max(image_shape), 0.0, 1.0)

    def rho_max(self, image_shape: Tuple[int, int]) -> float:
        # Keep normalization scale stable even when p1/p2 explores aggressive values.
        return self._rho_scale_raw(image_shape)

    def g(self, r_hat: np.ndarray) -> np.ndarray:
        r = np.asarray(r_hat, dtype=np.float64)
        a1, a2, a3, a4 = self.coeffs
        return a1 * r + a2 * (r**2) + a3 * (r**3) + a4 * (r**4)

    def g_prime(self, r_hat: np.ndarray) -> np.ndarray:
        r = np.asarray(r_hat, dtype=np.float64)
        a1, a2, a3, a4 = self.coeffs
        return a1 + 2.0 * a2 * r + 3.0 * a3 * (r**2) + 4.0 * a4 * (r**3)

    def g_second(self, r_hat: np.ndarray) -> np.ndarray:
        r = np.asarray(r_hat, dtype=np.float64)
        a2, a3, a4 = self.coeffs[1:]
        return 2.0 * a2 + 6.0 * a3 * r + 12.0 * a4 * (r**2)

    def image_points_to_rectified(
        self,
        points_uv: np.ndarray,
        image_shape: Tuple[int, int],
        tan_clip: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map image pixels to normalized rectified coordinates (f_out = 1).

        Returns:
            q_xy: Nx2 array of rectified coordinates.
            valid: N bool mask.
        """
        pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
        rho_max = self.rho_max(image_shape)
        xd, yd = self._scaled_components(pts)
        xu, yu = self.undistort_tangential_components(xd, yd, image_shape=image_shape)
        rho = np.hypot(xu, yu)
        phi = np.arctan2(yu, xu)
        r_hat = np.clip(rho / rho_max, 0.0, 1.0)
        theta = self.g(r_hat)
        tan_theta = np.tan(theta)
        valid = (
            np.isfinite(tan_theta)
            & np.isfinite(theta)
            & (theta > 0.0)
            & (theta < np.deg2rad(89.9))
        )
        tan_theta = np.clip(tan_theta, -tan_clip, tan_clip)
        x = tan_theta * np.cos(phi)
        y = tan_theta * np.sin(phi)
        return np.stack([x, y], axis=1), valid

    def _inverse_lut(
        self, num_samples: int = 4096
    ) -> Tuple[np.ndarray, np.ndarray]:
        r = np.linspace(0.0, 1.0, num_samples)
        theta = self.g(r)
        # Make inversion numerically safe even when optimization temporarily breaks monotonicity.
        theta_mono = np.maximum.accumulate(theta)
        theta_mono += np.linspace(0.0, 1e-8, num_samples)
        return theta_mono, r

    def theta_to_rhat(self, theta: np.ndarray, num_samples: int = 4096) -> np.ndarray:
        theta_samples, r_samples = self._inverse_lut(num_samples=num_samples)
        t = np.asarray(theta, dtype=np.float64)
        t_clipped = np.clip(t, theta_samples[0], theta_samples[-1])
        return np.interp(t_clipped, theta_samples, r_samples)

    def rectified_points_to_image(
        self, points_xy: np.ndarray, image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse mapping from normalized rectified plane (f=1) to fisheye image."""
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        r = np.hypot(pts[:, 0], pts[:, 1])
        theta = np.arctan(r)
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        r_hat = self.theta_to_rhat(theta)
        rho = r_hat * self.rho_max(image_shape)
        xu = rho * np.cos(phi)
        yu = rho * np.sin(phi)
        xd, yd = self.distort_tangential_components(xu, yu, image_shape=image_shape)
        u = self.cx + self.sx * xd
        v = self.cy + self.sy * yd
        h, w = image_shape[:2]
        valid = (u >= 0.0) & (u <= w - 1.0) & (v >= 0.0) & (v <= h - 1.0)
        return np.stack([u, v], axis=1), valid

    def curve(self, n: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        r = np.linspace(0.0, 1.0, n)
        return r, self.g(r)

    def to_dict(self, image_shape: Tuple[int, int] | None = None) -> Dict[str, float]:
        out = {
            "cx": float(self.cx),
            "cy": float(self.cy),
            "a1": float(self.coeffs[0]),
            "a2": float(self.coeffs[1]),
            "a3": float(self.coeffs[2]),
            "a4": float(self.coeffs[3]),
            "sx": float(self.sx),
            "sy": float(self.sy),
            "p1": float(self.p1),
            "p2": float(self.p2),
        }
        if image_shape is not None:
            out["rho_max"] = float(self.rho_max(image_shape))
        return out
