"""Radial fisheye models with poly4 and Zernike basis families."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Dict, Tuple

import numpy as np


EPS = 1e-9
SUPPORTED_MODELS = ("zernike2", "zernike4", "zernike6", "poly4")


def zernike_orders_for_family(family: str) -> Tuple[int, ...]:
    family = family.lower()
    if family == "zernike2":
        return (1, 3)
    if family == "zernike4":
        return (1, 3, 5, 7)
    if family == "zernike6":
        return (1, 3, 5, 7, 9, 11)
    raise ValueError(f"{family!r} is not a Zernike model family.")


def coefficient_count(family: str) -> int:
    family = family.lower()
    if family == "poly4":
        return 4
    if family in ("zernike2", "zernike4", "zernike6"):
        return len(zernike_orders_for_family(family))
    raise ValueError(f"Unsupported model family: {family}")


@lru_cache(maxsize=None)
def _zernike_radial_coefficients(n: int, m: int = 1) -> Tuple[float, ...]:
    """Return power-basis coefficients for radial Zernike polynomial R_n^m."""
    if n < 0 or m < 0 or n < m or (n - m) % 2:
        raise ValueError(f"Invalid Zernike order n={n}, m={m}")
    coeffs = np.zeros(n + 1, dtype=np.float64)
    for s in range((n - m) // 2 + 1):
        power = n - 2 * s
        num = (-1.0) ** s * factorial(n - s)
        den = (
            factorial(s)
            * factorial((n + m) // 2 - s)
            * factorial((n - m) // 2 - s)
        )
        coeffs[power] = num / den
    return tuple(float(c) for c in coeffs)


def _poly_eval_power(coeffs: Tuple[float, ...], r: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r, dtype=np.float64)
    for power, coeff in enumerate(coeffs):
        if coeff:
            out = out + coeff * (r**power)
    return out


def _poly_derivative_coefficients(
    coeffs: Tuple[float, ...], order: int
) -> Tuple[float, ...]:
    arr = np.asarray(coeffs, dtype=np.float64)
    for _ in range(order):
        if arr.size <= 1:
            return (0.0,)
        arr = np.asarray([p * arr[p] for p in range(1, arr.size)], dtype=np.float64)
    return tuple(float(c) for c in arr)


def zernike_basis(family: str, r_hat: np.ndarray, derivative_order: int = 0) -> np.ndarray:
    """Evaluate odd radial Zernike basis rows for r in [0, 1]."""
    r = np.asarray(r_hat, dtype=np.float64)
    cols = []
    for n in zernike_orders_for_family(family):
        coeffs = _zernike_radial_coefficients(n, 1)
        if derivative_order:
            coeffs = _poly_derivative_coefficients(coeffs, derivative_order)
        cols.append(_poly_eval_power(coeffs, r))
    return np.stack(cols, axis=-1)


@dataclass
class RadialFisheyeModel:
    """Fisheye projection model with interchangeable radial basis.

    The radial function maps normalized image radius ``r_hat`` to ray angle
    ``theta``. For Zernike families, the basis uses odd ``R_n^1`` polynomials,
    which naturally keep ``g(0)=0``.
    """

    family: str
    cx: float
    cy: float
    coeffs: np.ndarray
    sx: float = 1.0
    sy: float = 1.0
    p1: float = 0.0
    p2: float = 0.0

    def __post_init__(self) -> None:
        self.family = self.family.lower()
        n = coefficient_count(self.family)
        self.coeffs = np.asarray(self.coeffs, dtype=np.float64).reshape(n)
        self.sx = float(self.sx)
        self.sy = float(self.sy)
        self.p1 = float(self.p1)
        self.p2 = float(self.p2)
        if self.sx <= 0.0 or self.sy <= 0.0:
            raise ValueError("sx and sy must be positive.")

    @staticmethod
    def initial_from_shape(
        image_shape: Tuple[int, int],
        family: str = "zernike4",
        max_half_angle_deg: float = 105.0,
    ) -> "RadialFisheyeModel":
        h, w = image_shape[:2]
        coeffs = np.zeros(coefficient_count(family), dtype=np.float64)
        coeffs[0] = np.deg2rad(max_half_angle_deg)
        return RadialFisheyeModel(
            family=family,
            cx=0.5 * (w - 1),
            cy=0.5 * (h - 1),
            coeffs=coeffs,
        )

    def to_vector(self) -> np.ndarray:
        return np.array(
            [self.cx, self.cy, *self.coeffs.tolist(), self.sx, self.sy, self.p1, self.p2],
            dtype=np.float64,
        )

    @staticmethod
    def from_vector(family: str, vec: np.ndarray) -> "RadialFisheyeModel":
        family = family.lower()
        n = coefficient_count(family)
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        expected = n + 6
        if v.size != expected:
            raise ValueError(f"Expected vector of size {expected}, got {v.size}")
        return RadialFisheyeModel(
            family=family,
            cx=float(v[0]),
            cy=float(v[1]),
            coeffs=v[2 : 2 + n],
            sx=float(v[2 + n]),
            sy=float(v[3 + n]),
            p1=float(v[4 + n]),
            p2=float(v[5 + n]),
        )

    def _scaled_components(self, points_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
        dx = (pts[:, 0] - self.cx) / max(self.sx, EPS)
        dy = (pts[:, 1] - self.cy) / max(self.sy, EPS)
        return dx, dy

    def _rho_scale_raw(self, image_shape: Tuple[int, int]) -> float:
        h, w = image_shape[:2]
        corners = np.array(
            [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]],
            dtype=np.float64,
        )
        dx = (corners[:, 0] - self.cx) / max(self.sx, EPS)
        dy = (corners[:, 1] - self.cy) / max(self.sy, EPS)
        return float(np.max(np.hypot(dx, dy)) + EPS)

    def rho_max(self, image_shape: Tuple[int, int]) -> float:
        return self._rho_scale_raw(image_shape)

    def distort_tangential_components(
        self, xu: np.ndarray, yu: np.ndarray, image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(xu, dtype=np.float64)
        y = np.asarray(yu, dtype=np.float64)
        s = self._rho_scale_raw(image_shape)
        xn = x / s
        yn = y / s
        r2 = xn * xn + yn * yn
        dxn = 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn)
        dyn = self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn
        return (xn + dxn) * s, (yn + dyn) * s

    def undistort_tangential_components(
        self,
        xd: np.ndarray,
        yd: np.ndarray,
        image_shape: Tuple[int, int],
        num_iters: int = 7,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            xu = np.clip(xdn - dxn, -4.0, 4.0)
            yu = np.clip(ydn - dyn, -4.0, 4.0)
        return xu * s, yu * s

    def normalized_radius(
        self, points_uv: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        xd, yd = self._scaled_components(points_uv)
        xu, yu = self.undistort_tangential_components(xd, yd, image_shape)
        return np.clip(np.hypot(xu, yu) / self.rho_max(image_shape), 0.0, 1.0)

    def _basis(self, r_hat: np.ndarray, derivative_order: int = 0) -> np.ndarray:
        r = np.asarray(r_hat, dtype=np.float64)
        if self.family == "poly4":
            if derivative_order == 0:
                return np.stack([r, r**2, r**3, r**4], axis=-1)
            if derivative_order == 1:
                return np.stack([np.ones_like(r), 2.0 * r, 3.0 * r**2, 4.0 * r**3], axis=-1)
            if derivative_order == 2:
                return np.stack([np.zeros_like(r), 2.0 * np.ones_like(r), 6.0 * r, 12.0 * r**2], axis=-1)
            return np.zeros((*r.shape, 4), dtype=np.float64)
        return zernike_basis(self.family, r, derivative_order)

    def g(self, r_hat: np.ndarray) -> np.ndarray:
        return self._basis(r_hat, 0) @ self.coeffs

    def g_prime(self, r_hat: np.ndarray) -> np.ndarray:
        return self._basis(r_hat, 1) @ self.coeffs

    def g_second(self, r_hat: np.ndarray) -> np.ndarray:
        return self._basis(r_hat, 2) @ self.coeffs

    def image_points_to_rectified(
        self,
        points_uv: np.ndarray,
        image_shape: Tuple[int, int],
        tan_clip: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
        xd, yd = self._scaled_components(pts)
        xu, yu = self.undistort_tangential_components(xd, yd, image_shape)
        rho = np.hypot(xu, yu)
        phi = np.arctan2(yu, xu)
        r_hat = np.clip(rho / self.rho_max(image_shape), 0.0, 1.0)
        theta = self.g(r_hat)
        tan_theta = np.tan(theta)
        valid = (
            np.isfinite(theta)
            & np.isfinite(tan_theta)
            & (theta > 0.0)
            & (theta < np.deg2rad(89.9))
        )
        tan_theta = np.clip(tan_theta, -tan_clip, tan_clip)
        return np.stack([tan_theta * np.cos(phi), tan_theta * np.sin(phi)], axis=1), valid

    def _inverse_lut(self, num_samples: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        r = np.linspace(0.0, 1.0, num_samples)
        theta = self.g(r)
        theta = np.maximum.accumulate(theta)
        theta += np.linspace(0.0, 1e-8, num_samples)
        return theta, r

    def theta_to_rhat(self, theta: np.ndarray, num_samples: int = 4096) -> np.ndarray:
        theta_samples, r_samples = self._inverse_lut(num_samples)
        t = np.asarray(theta, dtype=np.float64)
        return np.interp(np.clip(t, theta_samples[0], theta_samples[-1]), theta_samples, r_samples)

    def rectified_points_to_image(
        self, points_xy: np.ndarray, image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        r = np.hypot(pts[:, 0], pts[:, 1])
        theta = np.arctan(r)
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        r_hat = self.theta_to_rhat(theta)
        rho = r_hat * self.rho_max(image_shape)
        xu = rho * np.cos(phi)
        yu = rho * np.sin(phi)
        xd, yd = self.distort_tangential_components(xu, yu, image_shape)
        u = self.cx + self.sx * xd
        v = self.cy + self.sy * yd
        h, w = image_shape[:2]
        valid = (u >= 0.0) & (u <= w - 1.0) & (v >= 0.0) & (v <= h - 1.0)
        return np.stack([u, v], axis=1), valid

    def curve(self, n: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        r = np.linspace(0.0, 1.0, n)
        return r, self.g(r)

    def coefficient_names(self) -> Tuple[str, ...]:
        if self.family == "poly4":
            return ("a1", "a2", "a3", "a4")
        return tuple(f"c_R{n}_1" for n in zernike_orders_for_family(self.family))

    def coeff_regularization_weights(self) -> np.ndarray:
        if self.family == "poly4":
            return np.ones_like(self.coeffs)
        orders = np.asarray(zernike_orders_for_family(self.family), dtype=np.float64)
        weights = (orders / max(orders[0], 1.0)) ** 0.5
        weights[0] = 1.0
        return weights

    def to_dict(self, image_shape: Tuple[int, int] | None = None) -> Dict[str, float | str]:
        out: Dict[str, float | str] = {
            "family": self.family,
            "cx": float(self.cx),
            "cy": float(self.cy),
            "sx": float(self.sx),
            "sy": float(self.sy),
            "p1": float(self.p1),
            "p2": float(self.p2),
        }
        for name, value in zip(self.coefficient_names(), self.coeffs):
            out[name] = float(value)
        if image_shape is not None:
            out["rho_max"] = float(self.rho_max(image_shape))
            out["theta_edge_rad"] = float(self.g(np.array([1.0]))[0])
        return out
