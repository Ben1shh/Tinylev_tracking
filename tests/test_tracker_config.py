from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = APP_ROOT / "analysis" / "particle_tracking_app_v0_1"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))

from particle_tracking_app.core import TrackingConfig  # noqa: E402
from tinylev_tracker.app import (  # noqa: E402
    FAST_SPECTRUM_MAX_HZ,
    FAST_SPECTRUM_MIN_HZ,
    PLOT_MODES,
    dominant_peak_frequency,
    derive_plot_kinematics,
    tracker_defaults,
    normalized_power_spectrum,
    scaled_limits,
    spectrum_display_curve,
    unused_output_dir,
)


class LightConfigTests(unittest.TestCase):
    def test_defaults_are_fast_full_frame_and_include_set02_large_particles(self) -> None:
        config = tracker_defaults(TrackingConfig(video_path="sample.avi"), 1280, 1024)
        self.assertTrue(config.fast_mode)
        self.assertFalse(config.save_annotated_video)
        self.assertFalse(config.save_debug_frames)
        self.assertEqual((config.roi_x_min, config.roi_x_max), (0, 1280))
        self.assertEqual((config.roi_y_min, config.roi_y_max), (0, 1024))
        self.assertLessEqual(config.min_radius, 107.7)
        self.assertGreaterEqual(config.max_radius, 173.2)

    def test_existing_output_gets_numbered_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            requested = Path(temporary) / "run"
            requested.mkdir()
            (requested / "tracking.csv").write_text("existing", encoding="utf-8")
            self.assertEqual(unused_output_dir(requested), requested.with_name("run_002"))

    def test_plot_modes_replace_xy_plots_with_professor_note_kinematics(self) -> None:
        self.assertNotIn("CM XY", PLOT_MODES)
        self.assertNotIn("Particle XY", PLOT_MODES)
        for mode in (
            "d(t)", "psi(t)", "phi(t)", "Omega(t)",
            "d spectrum", "psi + Omega spectrum", "phi spectrum",
        ):
            self.assertIn(mode, PLOT_MODES)

    def test_scroll_zoom_scales_axis_around_cursor(self) -> None:
        lower, upper = scaled_limits((0.0, 100.0), center=25.0, scale=0.5)
        self.assertEqual((lower, upper), (12.5, 62.5))
        self.assertEqual(upper - lower, 50.0)
        log_lower, log_upper = scaled_limits((1e-7, 1.0), center=1e-3, scale=0.5, logarithmic=True)
        self.assertAlmostEqual(np.log10(log_upper) - np.log10(log_lower), 3.5)

    def test_kinematics_and_spectrum_follow_cartesian_sign_and_units(self) -> None:
        time_s = np.arange(0.0, 4.01, 0.01)
        psi = 2.0 * np.pi * 2.0 * time_s
        theta = 2.0 * np.pi * 0.5 * time_s
        separation = 50.0 + 2.0 * np.sin(2.0 * np.pi * 5.0 * time_s)
        rows = []
        for frame, (time_value, psi_value, theta_value, distance) in enumerate(
            zip(time_s, psi, theta, separation)
        ):
            rows.append(
                {
                    "frame": frame,
                    "time_s": time_value,
                    "status": "ok",
                    "small_cx_px": 0.0,
                    "small_cy_px": 0.0,
                    "large_cx_px": distance * np.cos(psi_value),
                    "large_cy_px": -distance * np.sin(psi_value),
                    "cm_dx_from_trap_px": 10.0 * np.cos(theta_value),
                    "cm_dy_from_trap_px": -10.0 * np.sin(theta_value),
                }
            )

        derived = derive_plot_kinematics(rows)
        np.testing.assert_allclose(derived["d_px"], separation, atol=1e-10)
        np.testing.assert_allclose(derived["psi_unwrapped_rad"], psi, atol=1e-10)
        expected_phi = (psi - theta + np.pi) % (2.0 * np.pi) - np.pi
        phi_difference = (derived["phi_rad"] - expected_phi + np.pi) % (2.0 * np.pi) - np.pi
        np.testing.assert_allclose(phi_difference, 0.0, atol=1e-10)
        finite_omega = derived["omega_rad_s"][np.isfinite(derived["omega_rad_s"])]
        self.assertAlmostEqual(float(np.median(finite_omega)), 4.0 * np.pi, places=8)

        frequency, power, metadata = normalized_power_spectrum(
            derived["time_s"], derived["d_px"], derived["pair_valid"], derived["frame"]
        )
        dominant_hz = float(frequency[int(np.argmax(power))])
        self.assertAlmostEqual(dominant_hz, 5.0, delta=metadata["resolution_hz"])
        self.assertAlmostEqual(float(power.sum()), 1.0, places=12)

        display_frequency, display_power = spectrum_display_curve(
            frequency, power, FAST_SPECTRUM_MIN_HZ, FAST_SPECTRUM_MAX_HZ
        )
        self.assertTrue(np.all((display_frequency >= 2.0) & (display_frequency <= 15.0)))
        self.assertAlmostEqual(float(display_power.max()), 1.0, places=12)
        peak_hz = dominant_peak_frequency(
            frequency, power, FAST_SPECTRUM_MIN_HZ, FAST_SPECTRUM_MAX_HZ
        )
        self.assertAlmostEqual(peak_hz, 5.0, delta=metadata["resolution_hz"])


if __name__ == "__main__":
    unittest.main()
