from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from fisheye_zernike.lines import AutoLineConfig, extract_line_constraints_auto
from fisheye_zernike.optimize import OptimizeConfig, calibrate_from_lines
from fisheye_zernike.rectify import RectifyConfig, render_rectified


class RealImageSmokeTest(unittest.TestCase):
    def test_real_image_pipeline_smoke(self) -> None:
        root = Path.cwd()
        candidates = [
            root / "fisheye1.jpg",
            root / "fisheye2.jpg",
            root.parent / "fisheye1.jpg",
            root.parent / "fisheye2.jpg",
            root.parent / "data" / "raw" / "fisheye1.jpg",
            root.parent / "data" / "raw" / "fisheye2.jpg",
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is None:
            self.skipTest("No real fisheye sample image found.")
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            self.skipTest(f"Could not read {img_path}")
        h, w = image.shape[:2]
        if w > 850:
            scale = 850 / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        lines, _ = extract_line_constraints_auto(image, AutoLineConfig(max_lines=18, min_quality_score=0.18))
        if len(lines) < 3:
            self.skipTest("Automatic line extraction found too few lines.")
        calib = calibrate_from_lines(
            image.shape[:2],
            lines,
            model_family="zernike4",
            optimize_cfg=OptimizeConfig(
                max_iters=70,
                alternating_rounds=2,
                multi_start=1,
                verbose=0,
                use_projection_priors=False,
            ),
        )
        render = render_rectified(image, calib.model, RectifyConfig(output_fov_deg=100.0, crop_mode="auto"))
        rect = render["rectified_crop"]
        self.assertGreater(rect.shape[0], 100)
        self.assertGreater(rect.shape[1], 100)
        self.assertGreaterEqual(calib.final_metrics["accepted_lines"], 1.0)


if __name__ == "__main__":
    unittest.main()
