from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANALYSIS_ROOT))

from dynamic_pair_analysis.pipeline import (  # noqa: E402
    AnalysisConfig,
    apply_manual_annotations,
    classify_states,
    clean_tracking,
    derive_geometry,
    theoretical_rlc,
    unwrap_with_gaps,
)
from dynamic_pair_analysis.phenomenological_center_report import fit_phase_locked_orbit  # noqa: E402


def synthetic_tracking(kind: str, fps: float = 100.0, duration: float = 6.0) -> pd.DataFrame:
    time = np.arange(0, duration, 1 / fps)
    if kind == "static":
        psi = np.zeros_like(time)
    elif kind == "libration":
        psi = 0.35 * np.sin(2 * np.pi * 2.0 * time)
    elif kind == "spinning":
        psi = 2 * np.pi * 1.0 * time
    elif kind == "reversal":
        psi = np.where(time < duration / 2, 2 * np.pi * time, 2 * np.pi * (duration - time))
    elif kind == "rapid_excursion":
        psi = 1.2 * np.sin(2 * np.pi * 2.0 * time)
    else:
        raise ValueError(kind)
    center_x, center_y = 618.6307, 407.6506
    separation = 240.0
    small_x = center_x - separation / 2 * np.cos(psi)
    small_y = center_y + separation / 2 * np.sin(psi)
    large_x = center_x + separation / 2 * np.cos(psi)
    large_y = center_y - separation / 2 * np.sin(psi)
    return pd.DataFrame(
        {
            "frame": np.arange(len(time)), "time_s": time, "status": "ok", "selected_count": 2,
            "particle_1_cx_px": small_x, "particle_1_cy_px": small_y, "particle_1_radius_px": 110.0, "particle_1_conf": 0.99,
            "particle_2_cx_px": large_x, "particle_2_cy_px": large_y, "particle_2_radius_px": 130.0, "particle_2_conf": 0.99,
        }
    )


class DynamicPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AnalysisConfig(
            trap_x_px=618.6307, trap_y_px=407.6506, smoothing_window_s=0.15,
            trap_se_x_px=2.20071636210073, trap_se_y_px=3.3308262604128203,
            classification_window_s=2.0, minimum_state_duration_s=1.0,
        )

    def classify(self, kind: str) -> pd.DataFrame:
        clean, _ = clean_tracking(synthetic_tracking(kind), self.config)
        return classify_states(derive_geometry(clean, self.config), self.config)

    def test_static_classification(self) -> None:
        result = self.classify("static")
        self.assertGreater((result["state"] == "static").mean(), 0.7)

    def test_static_classification_does_not_depend_on_calibration_radius_stability(self) -> None:
        clean, _ = clean_tracking(synthetic_tracking("static"), self.config)
        derived = derive_geometry(clean, self.config)
        derived["r_px"] = np.linspace(0.0, 100.0, len(derived))
        result = classify_states(derived, self.config)
        self.assertGreater((result["state"] == "static").mean(), 0.7)

    def test_libration_classification(self) -> None:
        result = self.classify("libration")
        self.assertGreater((result["state"] == "libration").mean(), 0.5)

    def test_spinning_classification(self) -> None:
        result = self.classify("spinning")
        self.assertGreater((result["state"] == "pair_axis_spinning").mean(), 0.5)

    def test_large_pair_axis_excursion_is_candidate_without_stable_direction(self) -> None:
        result = self.classify("rapid_excursion")
        self.assertGreater((result["state"] == "pair_axis_spinning_candidate").mean(), 0.5)
        candidate = result[result["state"].eq("pair_axis_spinning_candidate")]
        self.assertGreater((candidate["psi_dot"] > 0).mean(), 0.2)
        self.assertGreater((candidate["psi_dot"] < 0).mean(), 0.2)

    def test_reversal_contains_opposite_spin_directions(self) -> None:
        result = self.classify("reversal")
        spin = result[result["state"].eq("pair_axis_spinning")]
        self.assertGreater(len(spin), 0)
        directions = np.sign(spin.groupby("segment_id")["psi_dot"].median())
        self.assertIn(-1.0, directions.to_numpy())
        self.assertIn(1.0, directions.to_numpy())

    def test_identity_swap_is_repaired(self) -> None:
        raw = synthetic_tracking("spinning")
        cut = len(raw) // 2
        first = ["cx_px", "cy_px", "radius_px", "conf"]
        for suffix in first:
            one = f"particle_1_{suffix}"
            two = f"particle_2_{suffix}"
            original_one = raw.loc[cut:, one].to_numpy(copy=True)
            original_two = raw.loc[cut:, two].to_numpy(copy=True)
            raw.loc[cut:, one] = original_two
            raw.loc[cut:, two] = original_one
        clean, _ = clean_tracking(raw, self.config)
        self.assertTrue((clean["small_radius_px"] == 110.0).all())
        self.assertGreater(clean["assignment_swapped"].sum(), 0)

    def test_coherent_fast_rotation_is_not_a_center_jump(self) -> None:
        raw = synthetic_tracking("static")
        time = raw["time_s"].to_numpy()
        fast = (time >= 2.0) & (time <= 4.0)
        psi = 2 * np.pi * 8.0 * (time[fast] - 2.0)
        center_x, center_y, half = 618.6307, 407.6506, 120.0
        raw.loc[fast, "particle_1_cx_px"] = center_x - half * np.cos(psi)
        raw.loc[fast, "particle_1_cy_px"] = center_y + half * np.sin(psi)
        raw.loc[fast, "particle_2_cx_px"] = center_x + half * np.cos(psi)
        raw.loc[fast, "particle_2_cy_px"] = center_y - half * np.sin(psi)
        clean, metrics = clean_tracking(raw, self.config)
        interval = clean.loc[fast]
        self.assertGreater(interval["coherent_fast_motion"].sum(), 50)
        self.assertLess(interval["center_jump"].mean(), 0.05)
        self.assertGreater(metrics["coherent_fast_motion_frames"], 50)
        classified = classify_states(derive_geometry(clean, self.config), self.config)
        self.assertGreater((classified.loc[fast, "state"] == "pair_axis_spinning").mean(), 0.4)

    def test_missing_gap_breaks_unwrap(self) -> None:
        angle = np.linspace(0, 4 * np.pi, 20)
        valid = np.ones(20, dtype=bool); valid[8:12] = False
        result = unwrap_with_gaps((angle + np.pi) % (2 * np.pi) - np.pi, valid)
        self.assertTrue(np.isnan(result[8:12]).all())

    def test_rlc_formula(self) -> None:
        self.assertAlmostEqual(theoretical_rlc(1.0, 2.0), 33 / 9)

    def test_phase_locked_center_fit(self) -> None:
        psi = np.linspace(-4 * np.pi, 4 * np.pi, 500)
        expected_center = 12.5 - 7.25j
        expected_amplitude = 4.2 * np.exp(0.3j)
        z = expected_center + expected_amplitude * np.exp(1j * psi)
        center, amplitude = fit_phase_locked_orbit(z, psi)
        self.assertAlmostEqual(center.real, expected_center.real, places=10)
        self.assertAlmostEqual(center.imag, expected_center.imag, places=10)
        self.assertAlmostEqual(amplitude.real, expected_amplitude.real, places=10)
        self.assertAlmostEqual(amplitude.imag, expected_amplitude.imag, places=10)

    def test_legacy_spinning_annotation_maps_to_pair_axis_and_preserves_invalid(self) -> None:
        import tempfile

        result = self.classify("spinning")
        result.loc[10, "valid"] = False
        result.loc[10, "state"] = "invalid"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manual_annotations.csv"
            pd.DataFrame(
                [{"video_id": "test", "start_frame": 0, "end_frame": 20, "override_state": "spinning", "reviewer": "test", "notes": "legacy"}]
            ).to_csv(path, index=False)
            annotated = apply_manual_annotations(result, path)
        self.assertEqual(annotated.loc[0, "state"], "pair_axis_spinning")
        self.assertEqual(annotated.loc[10, "state"], "invalid")


if __name__ == "__main__":
    unittest.main()
