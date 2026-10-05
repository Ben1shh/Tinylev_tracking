from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from particle_tracking_app.analysis_service import audit_video, run_phase_c, run_phase_d_core  # noqa: E402
from particle_tracking_app.catalog import (  # noqa: E402
    ExperimentCatalog,
    ExperimentRecord,
    VideoRecord,
    load_catalog,
    save_catalog,
)


class ExperimentCatalogTests(unittest.TestCase):
    @staticmethod
    def phase_d_tracks(
        turns: float,
        count: int = 241,
        center_start: complex = 12.0 - 8.0j,
        center_end: complex | None = None,
    ) -> pd.DataFrame:
        psi = np.linspace(0, 2 * np.pi * turns, count)
        if center_end is None:
            center_end = center_start
        center = np.linspace(center_start.real, center_end.real, count) + 1j * np.linspace(
            center_start.imag, center_end.imag, count
        )
        amplitude = 4.0 * np.exp(0.25j)
        z = center + amplitude * np.exp(1j * psi)
        return pd.DataFrame(
            {
                "frame": np.arange(count),
                "time_s": np.arange(count) / 120.0,
                "valid": True,
                "cm_x_px": z.real,
                "cm_y_px_image": -z.imag,
                "psi_rad": (psi + np.pi) % (2 * np.pi) - np.pi,
                "psi_unwrapped_rad": psi,
            }
        )

    def test_catalog_roundtrip_and_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            catalog = ExperimentCatalog(experiments=[ExperimentRecord("exp", "Experiment")])
            save_catalog(catalog, path)
            self.assertEqual(load_catalog(path).experiment("exp").display_name, "Experiment")
            catalog.experiments.append(ExperimentRecord("exp", "Duplicate"))
            with self.assertRaises(ValueError):
                save_catalog(catalog, path)

    def test_phase_a_blocks_truncated_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_path = root / "input.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((24, 32, 3), np.uint8))
            writer.release()
            tracking = root / "tracking.csv"
            pd.DataFrame([{"frame": 0}]).to_csv(tracking, index=False)
            experiment = ExperimentRecord("exp", "Experiment", trap_x_px=1, trap_y_px=1)
            video = VideoRecord("vid", str(video_path), tracking_csv=str(tracking))
            audit = audit_video(experiment, video)
            self.assertFalse(audit["analysis_allowed"])
            self.assertIn("tracking_truncated_or_empty", audit["issues"])

    def test_phase_c_fits_known_center_and_marks_unvalidated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            time = np.linspace(0, 2, 201)
            psi = 2 * np.pi * time
            center = 12.0 - 8.0j
            amplitude = 4.0 * np.exp(0.25j)
            z = center + amplitude * np.exp(1j * psi)
            derived = pd.DataFrame(
                {
                    "frame": np.arange(len(time)),
                    "time_s": time,
                    "valid": True,
                    "cm_x_px": z.real,
                    "cm_y_px_image": -z.imag,
                    "psi_rad": (psi + np.pi) % (2 * np.pi) - np.pi,
                    "psi_unwrapped_rad": psi,
                }
            )
            experiment = ExperimentRecord("exp", "Experiment", trap_x_px=10.0, trap_y_px=10.0)
            video = VideoRecord("vid", "unused.avi")
            result = run_phase_c(derived, experiment, video, 0, 200, Path(temporary))
            summary = result["summary"]
            self.assertAlmostEqual(summary["phenomenological_center_x_px"], center.real, places=6)
            self.assertAlmostEqual(summary["phenomenological_center_y_image_px"], -center.imag, places=6)
            self.assertEqual(summary["phase_d_validation"], "not_performed")
            self.assertIn("cannot qualify Rlc", summary["interpretation"])

    def test_phase_d_recovers_fixed_center_with_half_and_cycle_holdouts(self) -> None:
        derived = self.phase_d_tracks(6.0)
        experiment = ExperimentRecord("exp", "Experiment", trap_x_px=10.0, trap_y_px=10.0)
        video = VideoRecord("vid", "unused.avi")
        with tempfile.TemporaryDirectory() as temporary:
            import particle_tracking_app.analysis_service as service

            original_fit = service.fit_phase_locked_orbit
            fit_lengths: list[int] = []

            def recording_fit(z, psi):
                fit_lengths.append(len(z))
                return original_fit(z, psi)

            with patch("particle_tracking_app.analysis_service.fit_phase_locked_orbit", side_effect=recording_fit):
                result = run_phase_d_core(derived, experiment, video, 0, 240, Path(temporary))
            summary = result["summary"]
            validation = result["cross_validation"]
            self.assertEqual(summary["phase_d_status"], "core_cross_validation_complete")
            self.assertGreaterEqual(summary["complete_cycle_count"], 5)
            self.assertTrue((validation["validation_refit_performed"] == False).all())  # noqa: E712
            self.assertLess(validation["validation_rmse_px"].max(), 1e-9)
            self.assertLess((validation["train_center_x_px"] - 12.0).abs().max(), 1e-9)
            self.assertTrue(fit_lengths)
            self.assertLess(max(fit_lengths), len(derived))
            self.assertNotIn("validated_effective_center_orbit", summary.values())

    def test_phase_d_drift_degrades_holdout_without_scientific_promotion(self) -> None:
        derived = self.phase_d_tracks(5.0, center_end=18.0 - 8.0j)
        experiment = ExperimentRecord("exp", "Experiment", trap_x_px=10.0, trap_y_px=10.0)
        video = VideoRecord("vid", "unused.avi")
        with tempfile.TemporaryDirectory() as temporary:
            result = run_phase_d_core(derived, experiment, video, 0, 240, Path(temporary))
        half = result["cross_validation"].query("method == 'temporal_half_holdout'")
        self.assertTrue((half["validation_rmse_px"] > half["train_rmse_px"]).all())
        self.assertIn("manual_review", result["summary"]["promotion_status"])

    def test_phase_d_reports_insufficient_and_single_cycle_states(self) -> None:
        experiment = ExperimentRecord("exp", "Experiment", trap_x_px=10.0, trap_y_px=10.0)
        video = VideoRecord("vid", "unused.avi")
        with tempfile.TemporaryDirectory() as temporary:
            first = run_phase_d_core(self.phase_d_tracks(0.8), experiment, video, 0, 240, Path(temporary) / "a")
            second = run_phase_d_core(self.phase_d_tracks(1.5), experiment, video, 0, 240, Path(temporary) / "b")
        self.assertEqual(first["summary"]["phase_d_status"], "insufficient_for_half_holdout")
        self.assertEqual(second["summary"]["phase_d_status"], "half_holdout_complete_cycle_holdout_unavailable")
        self.assertEqual(second["summary"]["complete_cycle_count"], 1)

    def test_phase_d_cycles_do_not_bridge_invalid_gap(self) -> None:
        derived = self.phase_d_tracks(1.2)
        midpoint = len(derived) // 2
        derived.loc[midpoint, "valid"] = False
        # Each side spans only 0.6 turn. A gap-blind implementation would
        # incorrectly assemble one complete cycle from the two pieces.
        experiment = ExperimentRecord("exp", "Experiment", trap_x_px=10.0, trap_y_px=10.0)
        video = VideoRecord("vid", "unused.avi")
        with tempfile.TemporaryDirectory() as temporary:
            result = run_phase_d_core(derived, experiment, video, 0, 240, Path(temporary))
        self.assertEqual(result["summary"]["complete_cycle_count"], 0)
        self.assertEqual(result["summary"]["cycle_holdout_status"], "cycle_holdout_unavailable")

    def test_phase_d_cycles_do_not_bridge_frame_gap_or_direction_reversal(self) -> None:
        experiment = ExperimentRecord("exp", "Experiment", trap_x_px=10.0, trap_y_px=10.0)
        video = VideoRecord("vid", "unused.avi")
        frame_gap = self.phase_d_tracks(1.2)
        midpoint = len(frame_gap) // 2
        frame_gap.loc[midpoint:, "frame"] += 1

        reversal = self.phase_d_tracks(1.0)
        psi = np.concatenate(
            [np.linspace(0, 1.5 * np.pi, midpoint + 1), np.linspace(1.5 * np.pi, 0, len(reversal) - midpoint - 1)]
        )
        center = 12.0 - 8.0j
        amplitude = 4.0 * np.exp(0.25j)
        z = center + amplitude * np.exp(1j * psi)
        reversal["psi_unwrapped_rad"] = psi
        reversal["psi_rad"] = (psi + np.pi) % (2 * np.pi) - np.pi
        reversal["cm_x_px"] = z.real
        reversal["cm_y_px_image"] = -z.imag

        with tempfile.TemporaryDirectory() as temporary:
            gap_result = run_phase_d_core(frame_gap, experiment, video, 0, 241, Path(temporary) / "gap")
            reversal_result = run_phase_d_core(reversal, experiment, video, 0, 240, Path(temporary) / "reversal")
        self.assertEqual(gap_result["summary"]["complete_cycle_count"], 0)
        self.assertEqual(reversal_result["summary"]["complete_cycle_count"], 0)

    def test_phase_d_records_tracking_row_count_mismatch(self) -> None:
        derived = self.phase_d_tracks(2.0)
        experiment = ExperimentRecord("exp", "Experiment", trap_x_px=10.0, trap_y_px=10.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_path = root / "input.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((24, 32, 3), np.uint8))
            writer.release()
            tracking = root / "tracking.csv"
            pd.DataFrame({"frame": [0, 1]}).to_csv(tracking, index=False)
            record = VideoRecord("vid", str(video_path), tracking_csv=str(tracking))
            result = run_phase_d_core(derived, experiment, record, 0, 240, root / "output")
        self.assertEqual(result["summary"]["current_tracking_rows"], 2)
        self.assertEqual(result["summary"]["video_frame_count"], 5)
        self.assertEqual(result["summary"]["upstream_tracking_status"], "row_count_mismatch")


if __name__ == "__main__":
    unittest.main()
