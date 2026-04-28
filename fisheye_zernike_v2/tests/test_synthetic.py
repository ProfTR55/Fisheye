from __future__ import annotations

import unittest

import numpy as np

from fisheye_zernike.loss import LossConfig, evaluate_objective
from fisheye_zernike.model import RadialFisheyeModel
from fisheye_zernike.optimize import OptimizeConfig, calibrate_from_lines


def make_synthetic_line_groups(
    model: RadialFisheyeModel,
    image_shape: tuple[int, int],
    noise_px: float = 0.35,
    seed: int = 7,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    line_groups: list[np.ndarray] = []
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
    base = np.linspace(-2.2, 2.2, 180)
    for kind, p1, p2 in specs:
        if kind == "slope":
            rect = np.stack([base, p1 * base + p2], axis=1)
        elif kind == "vert":
            rect = np.stack([np.full_like(base, p1), base], axis=1)
        else:
            rect = np.stack([base, np.full_like(base, p2)], axis=1)
        uv, valid = model.rectified_points_to_image(rect, image_shape)
        uv = uv[valid]
        if uv.shape[0] >= 50:
            line_groups.append(uv + rng.normal(scale=noise_px, size=uv.shape))
    return line_groups


class SyntheticCalibrationTest(unittest.TestCase):
    def test_recover_zernike4_model_from_line_constraints(self) -> None:
        image_shape = (720, 1000)
        true_model = RadialFisheyeModel(
            family="zernike4",
            cx=508.0,
            cy=356.0,
            coeffs=np.array([1.72, 0.07, -0.025, 0.012], dtype=np.float64),
            sx=1.04,
            sy=0.97,
            p1=0.018,
            p2=-0.014,
        )
        lines = make_synthetic_line_groups(true_model, image_shape)
        self.assertGreaterEqual(len(lines), 6)
        loss_cfg = LossConfig(lambda_anisotropy=1.5, lambda_tangential=0.8)
        init = RadialFisheyeModel.initial_from_shape(image_shape, family="zernike4")
        init_obj = evaluate_objective(init, image_shape, lines, loss_cfg)["objective"]
        result = calibrate_from_lines(
            image_shape,
            lines,
            model_family="zernike4",
            optimize_cfg=OptimizeConfig(
                max_iters=180,
                alternating_rounds=2,
                multi_start=1,
                verbose=0,
                use_projection_priors=False,
            ),
            loss_cfg=loss_cfg,
        )
        self.assertLess(result.final_metrics["objective"], init_obj)
        r = np.linspace(0.0, 1.0, 60)
        theta_err = float(np.sqrt(np.mean((result.model.g(r) - true_model.g(r)) ** 2)))
        self.assertLess(theta_err, 0.16)

    def test_poly4_and_zernike4_both_improve_baseline(self) -> None:
        image_shape = (640, 900)
        true_model = RadialFisheyeModel(
            family="zernike4",
            cx=455.0,
            cy=318.0,
            coeffs=np.array([1.65, 0.05, -0.02, 0.01], dtype=np.float64),
        )
        lines = make_synthetic_line_groups(true_model, image_shape, noise_px=0.25)
        loss_cfg = LossConfig(lambda_anisotropy=2.0, lambda_tangential=1.5)
        cfg = OptimizeConfig(
            max_iters=120,
            alternating_rounds=2,
            multi_start=1,
            verbose=0,
            use_projection_priors=False,
        )
        for family in ("poly4", "zernike4"):
            init = RadialFisheyeModel.initial_from_shape(image_shape, family=family)
            init_obj = evaluate_objective(init, image_shape, lines, loss_cfg)["objective"]
            result = calibrate_from_lines(image_shape, lines, family, cfg, loss_cfg)
            self.assertLess(result.final_metrics["objective"], init_obj)
            self.assertGreater(result.final_metrics["accepted_lines"], 3.0)


if __name__ == "__main__":
    unittest.main()
