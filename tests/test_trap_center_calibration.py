from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1] / "analysis" / "particle_tracking_app_v0_1"
sys.path.insert(0, str(APP_ROOT))

from particle_tracking_app.core import (  # noqa: E402
    ParticleDetection,
    TrackingConfig,
    calibrate_trap_center,
    write_image,
)


def detection(x: float, y: float, confidence: float = 0.9) -> ParticleDetection:
    mask = np.zeros((80, 120), dtype=np.uint8)
    return ParticleDetection(x, y, 8.0, confidence, 200.0, 0.02, 0, mask)


class TrapCenterCalibrationTests(unittest.TestCase):
    def test_averages_valid_detections_and_records_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_paths = []
            for index in range(3):
                path = root / f"image_{index}.png"
                self.assertTrue(write_image(path, np.zeros((80, 120, 3), dtype=np.uint8)))
                image_paths.append(path)

            results = [
                [detection(10.0, 20.0)],
                [detection(3.0, 4.0, confidence=0.2), detection(30.0, 40.0, confidence=0.95)],
                [],
            ]
            config = TrackingConfig(model_path=str(root / "unused.pt"), roi_x_min=0, roi_x_max=120, roi_y_min=0, roi_y_max=80)
            with patch("particle_tracking_app.core.infer_particle_detections", side_effect=results):
                result = calibrate_trap_center(image_paths, config, root / "output", model=object())

            summary = result["summary"]
            self.assertEqual(summary["valid_detection_count"], 2)
            self.assertEqual(summary["excluded_or_failed_count"], 1)
            self.assertAlmostEqual(summary["trap_x_px"], 20.0)
            self.assertAlmostEqual(summary["trap_y_px"], 30.0)
            self.assertEqual(result["rows"][1]["status"], "ok_multiple_candidates")
            self.assertEqual(result["rows"][2]["status"], "no_particle_detected")
            self.assertTrue((result["output_dir"] / "detections.csv").is_file())
            self.assertTrue((result["output_dir"] / "trap_center_summary.csv").is_file())
            self.assertTrue((result["output_dir"] / "overlay_preview.png").is_file())
            saved_summary = json.loads((result["output_dir"] / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["method"], "arithmetic_mean_of_detected_particle_centers_px")

    def test_existing_output_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            self.assertTrue(write_image(image_path, np.zeros((80, 120, 3), dtype=np.uint8)))
            requested = root / "output"
            requested.mkdir()
            marker = requested / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            config = TrackingConfig(model_path=str(root / "unused.pt"))
            with patch("particle_tracking_app.core.infer_particle_detections", return_value=[detection(10.0, 20.0)]):
                result = calibrate_trap_center([image_path], config, requested, model=object())

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(result["output_dir"], root / "output_002")


if __name__ == "__main__":
    unittest.main()
