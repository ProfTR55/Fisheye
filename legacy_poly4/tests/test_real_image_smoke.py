from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from lines import AutoLineConfig, extract_line_constraints_auto
from optimize import OptimizeConfig, calibrate_from_lines
from rectify import RectifyConfig, render_rectified


class RealImageSmokeTest(unittest.TestCase):
    def test_real_image_pipeline_smoke(self) -> None:
        img_path = None
        for p in [Path("fisheye1.jpg"), Path("fisheye2.jpg")]:
            if p.exists():
                img_path = p
                break
        if img_path is None:
            self.skipTest("No real fisheye sample image found in repository root.")

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            self.skipTest(f"Could not read {img_path}")

        # Keep the smoke test fast.
        max_w = 900
        h, w = image.shape[:2]
        if w > max_w:
            scale = max_w / w
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        lines, _ = extract_line_constraints_auto(image, AutoLineConfig())
        if len(lines) < 3:
            self.skipTest("Automatic line extraction found too few lines for a stable smoke test.")

        calib = calibrate_from_lines(
            image_shape=image.shape[:2],
            line_groups=lines,
            optimize_cfg=OptimizeConfig(
                max_iters=100, alternating_rounds=2, multi_start=2, verbose=0
            ),
        )
        render = render_rectified(
            image,
            calib.model,
            cfg=RectifyConfig(output_fov_deg=100.0, crop_mode="auto"),
        )

        rect = render["rectified_crop"]
        self.assertGreater(rect.shape[0], 100)
        self.assertGreater(rect.shape[1], 100)
        self.assertGreaterEqual(calib.final_metrics["accepted_lines"], 1.0)


if __name__ == "__main__":
    unittest.main()
