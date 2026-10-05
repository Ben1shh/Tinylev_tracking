from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from tinylev_tracker.manual_annotations import (  # noqa: E402
    ANNOTATION_FORMAT_VERSION,
    DIRECTION_CONVENTION,
    ManualAnnotationStore,
    read_segments_csv,
    sha256_file,
    validate_segments,
)


def segment(
    annotation_id: str,
    start_frame: int,
    end_frame: int,
    state: str,
) -> dict:
    return {
        "annotation_id": annotation_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "manual_state": state,
        "reviewer": "test",
        "notes": "",
    }


class ManualAnnotationTests(unittest.TestCase):
    def test_revision_round_trip_preserves_every_prior_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "sample.avi"
            tracking = root / "tracking.csv"
            config = root / "config.json"
            video.write_bytes(b"immutable-video-bytes")
            tracking.write_text("frame,time_s\n0,0.0\n", encoding="utf-8")
            config.write_text('{"video_path": "sample.avi"}\n', encoding="utf-8")

            store = ManualAnnotationStore.create(
                root / "annotations",
                video,
                video_info={"frames": 31, "fps": 10.0, "width": 64, "height": 48},
                tracking_csv_path=tracking,
                config_path=config,
            )
            first_csv, first_manifest, _ = store.save_revision(
                [segment("ann_0001", 0, 10, "static")],
                action="add ann_0001",
            )
            first_hash = sha256_file(first_csv)
            second_csv, second_manifest, _ = store.save_revision(
                [
                    segment("ann_0001", 0, 10, "static"),
                    segment("ann_0002", 11, 20, "spinning_CW"),
                ],
                action="add ann_0002",
            )

            self.assertNotEqual(first_csv, second_csv)
            self.assertEqual(sha256_file(first_csv), first_hash)
            self.assertTrue(first_manifest.is_file())
            self.assertTrue(second_manifest.is_file())

            loaded_store, loaded_segments, manifest = ManualAnnotationStore.load(second_csv)
            self.assertEqual(loaded_store.current_revision, 2)
            self.assertEqual([item["manual_state"] for item in loaded_segments], ["static", "spinning_CW"])
            self.assertEqual(loaded_segments[1]["start_time_s"], 1.1)
            self.assertEqual(manifest["format_version"], ANNOTATION_FORMAT_VERSION)
            self.assertEqual(manifest["direction_convention"], DIRECTION_CONVENTION)
            self.assertEqual(
                manifest["source_provenance"]["video"]["sha256"],
                sha256_file(video),
            )
            self.assertEqual(
                manifest["source_provenance"]["tracking_csv"]["sha256"],
                sha256_file(tracking),
            )

    def test_overlap_is_rejected_but_adjacent_inclusive_ranges_are_allowed(self) -> None:
        accepted = validate_segments(
            [
                segment("ann_0001", 0, 10, "static"),
                segment("ann_0002", 11, 20, "jiggling_libration"),
            ],
            fps=10.0,
            maximum_frame=20,
        )
        self.assertEqual(len(accepted), 2)
        with self.assertRaisesRegex(ValueError, "may not overlap"):
            validate_segments(
                [
                    segment("ann_0001", 0, 10, "static"),
                    segment("ann_0002", 10, 20, "transition"),
                ]
            )

    def test_legacy_manager_csv_maps_to_conservative_manual_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "video_id",
                        "start_frame",
                        "end_frame",
                        "override_state",
                        "reviewer",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video_id": "sample",
                        "start_frame": 2,
                        "end_frame": 5,
                        "override_state": "pair_axis_spinning",
                        "reviewer": "legacy",
                        "notes": "",
                    }
                )
            loaded = read_segments_csv(path, fps=10.0, maximum_frame=10)
            self.assertEqual(loaded[0]["manual_state"], "continuous_rotation_unspecified")
            self.assertEqual(loaded[0]["start_time_s"], 0.2)
            self.assertEqual(loaded[0]["end_time_s"], 0.5)

    def test_manifest_hash_mismatch_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "sample.avi"
            video.write_bytes(b"video")
            store = ManualAnnotationStore.create(
                root / "annotations",
                video,
                video_info={"frames": 5, "fps": 1.0},
            )
            csv_path, manifest_path, _ = store.save_revision(
                [segment("ann_0001", 0, 1, "static")],
                action="test",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["annotation_csv"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                ManualAnnotationStore.load(csv_path)


if __name__ == "__main__":
    unittest.main()
