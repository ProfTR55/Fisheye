from __future__ import annotations

import unittest

import numpy as np

from loss import LossConfig, evaluate_objective
from model import FisheyePoly4Model
from optimize import OptimizeConfig, calibrate_from_lines


def _make_synthetic_line_groups(
    model: FisheyePoly4Model,
    image_shape: tuple[int, int],
    noise_px: float = 0.6,
    seed: int = 7,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    line_groups: list[np.ndarray] = []

    # Straight lines in rectified plane (x, y) with f=1.
    # Use a wide rectified coverage so radial nonlinearity is observable.
    specs = [
        ("slope", 0.20, -0.40),
        ("slope", -0.35, 0.25),
        ("slope", 0.65, -0.10),
        ("slope", -0.75, 0.50),
        ("vert", -1.20, 0.0),
        ("vert", 1.10, 0.0),
        ("horiz", 0.0, -1.10),
        ("horiz", 0.0, 0.95),
    ]
    base = np.linspace(-2.4, 2.4, 220)

    for kind, p1, p2 in specs:
        if kind == "slope":
            x = base
            y = p1 * x + p2
        elif kind == "vert":
            x = np.full_like(base, p1)
            y = base
        else:
            x = base
            y = np.full_like(base, p2)

        rect = np.stack([x, y], axis=1)
        uv, valid = model.rectified_points_to_image(rect, image_shape=image_shape)
        uv = uv[valid]
        if uv.shape[0] < 60:
            continue
        uv = uv + rng.normal(scale=noise_px, size=uv.shape)
        line_groups.append(uv)

    return line_groups


class SyntheticCalibrationTest(unittest.TestCase):
    def test_recover_poly4_model_from_line_constraints(self) -> None:
        image_shape = (800, 1200)
        true_model = FisheyePoly4Model(
            cx=615.0,
            cy=392.0,
            coeffs=np.array([1.72, 0.12, -0.08, 0.03], dtype=np.float64),
            sx=1.08,
            sy=0.93,
            p1=0.035,
            p2=-0.022,
        )
        line_groups = _make_synthetic_line_groups(true_model, image_shape=image_shape)
        self.assertGreaterEqual(len(line_groups), 6)

        loss_cfg = LossConfig(lambda_anisotropy=1.5, lambda_tangential=0.8)
        init_model = FisheyePoly4Model.initial_from_shape(image_shape, max_half_angle_deg=105.0)
        init_obj = evaluate_objective(
            init_model, image_shape=image_shape, line_groups=line_groups, cfg=loss_cfg
        )["objective"]

        result = calibrate_from_lines(
            image_shape=image_shape,
            line_groups=line_groups,
            optimize_cfg=OptimizeConfig(
                max_iters=420,
                alternating_rounds=4,
                multi_start=3,
                start_anisotropy_jitter=0.12,
                start_tangential_jitter=0.06,
                verbose=0,
            ),
            loss_cfg=loss_cfg,
        )
        est = result.model
        final_obj = result.final_metrics["objective"]
        self.assertLess(final_obj, init_obj)
        self.assertLess(abs(est.cx - true_model.cx), 35.0)
        self.assertLess(abs(est.cy - true_model.cy), 35.0)

        r = np.linspace(0.0, 1.0, 50)
        theta_err = np.sqrt(np.mean((est.g(r) - true_model.g(r)) ** 2))
        self.assertLess(theta_err, 0.10)
        self.assertLess(abs(est.sx - true_model.sx), 0.20)
        self.assertLess(abs(est.sy - true_model.sy), 0.20)
        self.assertLess(abs(est.p1 - true_model.p1), 0.08)
        self.assertLess(abs(est.p2 - true_model.p2), 0.08)


if __name__ == "__main__":
    unittest.main()
