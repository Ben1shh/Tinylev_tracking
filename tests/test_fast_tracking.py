from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1] / "analysis" / "particle_tracking_app_v0_1"
sys.path.insert(0, str(APP_ROOT))

from particle_tracking_app.core import TrackingConfig, track_video  # noqa: E402


class FastTrackingTests(unittest.TestCase):
    def test_fast_mode_keeps_all_rows_and_throttles_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_path = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for value in range(5):
                writer.write(np.full((48, 64, 3), value * 20, dtype=np.uint8))
            writer.release()

            output_dir = root / "output"
            config = TrackingConfig(
                video_path=str(video_path),
                model_path=str(root / "unused.pt"),
                output_dir=str(output_dir),
                trap_x=0, trap_y=0, pixels_per_mm=1,
                fast_mode=True,
                preview_stride=2,
                save_annotated_video=True,
                save_debug_frames=True,
            )
            previews: list[int] = []

            def fake_process(_model, _frame, frame_idx, fps, _config, _prev_pair):
                return [], [], {
                    "frame": frame_idx,
                    "time_s": frame_idx / fps,
                    "status": "missing_particles",
                    "num_candidates": 0,
                    "requested_count": 2,
                    "selected_count": 0,
                }

            with (
                patch("particle_tracking_app.core.load_model", return_value=object()),
                patch("particle_tracking_app.core.process_frame", side_effect=fake_process),
                patch("particle_tracking_app.core.draw_overlay", side_effect=lambda frame, *_args, **_kwargs: frame),
            ):
                rows = track_video(
                    config,
                    frame_cb=lambda _frame, row: previews.append(int(row["frame"])),
                )

            self.assertEqual(len(rows), 5)
            self.assertEqual(previews, [0, 2, 4])
            self.assertTrue((output_dir / "tracking.csv").is_file())
            self.assertTrue((output_dir / "config.json").is_file())
            self.assertFalse(any(output_dir.glob("*.mp4")))
            self.assertFalse((output_dir / "debug_frames").exists())


if __name__ == "__main__":
    unittest.main()
