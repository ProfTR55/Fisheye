from __future__ import annotations

import unittest

import cv2
import numpy as np

from fisheye_zernike.hough_bootstrap import HoughBootstrapConfig, extract_hough_lines_from_rectified
from fisheye_zernike.model import RadialFisheyeModel


class HoughBootstrapSmokeTest(unittest.TestCase):
    def test_extract_hough_lines_maps_back_to_input(self) -> None:
        image_shape = (480, 640)
        model = RadialFisheyeModel.initial_from_shape(image_shape, family="zernike4", max_half_angle_deg=110.0)
        rectified = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(rectified, (80, 120), (560, 140), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(rectified, (120, 360), (520, 300), (255, 255, 255), 3, cv2.LINE_AA)
        f = 0.5 * 640 / np.tan(0.5 * np.deg2rad(100.0))
        meta = {"f_out": float(f), "cx_out": 319.5, "cy_out": 239.5}
        lines, report = extract_hough_lines_from_rectified(
            rectified,
            meta,
            model,
            image_shape,
            HoughBootstrapConfig(threshold=30, min_line_length=30, max_new_lines=4),
        )
        self.assertTrue(report.attempted)
        self.assertGreaterEqual(report.candidates_detected, 1)
        self.assertGreaterEqual(len(lines), 1)
        self.assertGreaterEqual(lines[0].shape[0], 12)


if __name__ == "__main__":
    unittest.main()
