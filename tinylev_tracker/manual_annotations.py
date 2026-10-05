from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ANNOTATION_FORMAT_VERSION = "tinylev-manual-state-annotations-v1"
FRAME_INTERVAL_CONVENTION = "zero-based, inclusive start_frame and end_frame"
DIRECTION_CONVENTION = (
    "CW/CCW refer to the small-to-large pair-axis angle in Cartesian y-up coordinates; "
    "the displayed image y axis points down."
)


@dataclass(frozen=True)
class StateSpec:
    key: str
    label: str
    color: str
    description: str


STATE_SPECS = (
    StateSpec("static", "static", "#4C78A8", "Visually stationary pair."),
    StateSpec(
        "jiggling_libration",
        "jiggling / libration",
        "#7C3AED",
        "Bounded rocking or repeated direction changes without sustained rotation.",
    ),
    StateSpec(
        "spinning_CW",
        "spinning CW (Cartesian y-up)",
        "#DC2626",
        "Sustained clockwise pair-axis rotation in Cartesian y-up coordinates.",
    ),
    StateSpec(
        "spinning_CCW",
        "spinning CCW (Cartesian y-up)",
        "#059669",
        "Sustained counter-clockwise pair-axis rotation in Cartesian y-up coordinates.",
    ),
    StateSpec(
        "continuous_rotation_unspecified",
        "continuous rotation (direction unspecified)",
        "#F97316",
        "Sustained rotation when direction is intentionally left unspecified.",
    ),
    StateSpec(
        "spinning_candidate",
        "spinning candidate",
        "#F59E0B",
        "Short or ambiguous rotation-like interval that needs later review.",
    ),
    StateSpec(
        "irregular_candidate",
        "irregular motion candidate (not chaos)",
        "#EC4899",
        "Visually irregular motion; this label is not evidence of deterministic chaos.",
    ),
    StateSpec(
        "transition",
        "transition / mixed boundary",
        "#14B8A6",
        "A boundary or mixed interval that should not be assigned to a stable-looking state.",
    ),
    StateSpec(
        "invalid_ambiguous",
        "invalid / ambiguous",
        "#64748B",
        "Occlusion, tracking failure, identity ambiguity, or an interval that cannot be classified.",
    ),
)
STATE_SPEC_BY_KEY = {item.key: item for item in STATE_SPECS}
STATE_COLORS = {item.key: item.color for item in STATE_SPECS}
MANUAL_STATE_COLUMNS = (
    "annotation_id",
    "revision",
    "video_id",
    "video_path",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "manual_state",
    "reviewer",
    "notes",
    "created_at_utc",
    "updated_at_utc",
)

LEGACY_STATE_MAP = {
    "libration": "jiggling_libration",
    "pair_axis_spinning": "continuous_rotation_unspecified",
    "spinning": "continuous_rotation_unspecified",
    "user_reviewed_continuous_rotation": "continuous_rotation_unspecified",
    "irregular_motion_candidate": "irregular_candidate",
    "invalid": "invalid_ambiguous",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stat = source.stat()
    return {
        "path": str(source),
        "bytes": int(stat.st_size),
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "sha256": sha256_file(source),
    }


def build_source_provenance(
    video_path: str | Path,
    video_info: dict[str, Any] | None = None,
    tracking_csv_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    video = _file_identity(video_path)
    info = video_info or {}
    video.update(
        {
            "video_id": Path(video["path"]).stem,
            "frame_count": _optional_int(info.get("frames")),
            "fps": _optional_float(info.get("fps")),
            "width_px": _optional_int(info.get("width")),
            "height_px": _optional_int(info.get("height")),
        }
    )
    return {
        "video": video,
        "tracking_csv": _optional_file_identity(tracking_csv_path),
        "config_json": _optional_file_identity(config_path),
    }


def _optional_file_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    return _file_identity(path)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def canonical_state(value: Any) -> str:
    state = str(value or "").strip()
    state = LEGACY_STATE_MAP.get(state, state)
    if state not in STATE_SPEC_BY_KEY:
        raise ValueError(
            f"Unsupported manual state {state!r}. Allowed states: "
            f"{', '.join(STATE_SPEC_BY_KEY)}"
        )
    return state


def normalize_segment(
    record: dict[str, Any],
    *,
    fps: float | None = None,
    maximum_frame: int | None = None,
) -> dict[str, Any]:
    try:
        start_frame = int(float(record["start_frame"]))
        end_frame = int(float(record["end_frame"]))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Every annotation needs integer start_frame and end_frame.") from error
    if start_frame < 0:
        raise ValueError("start_frame must be zero or greater.")
    if end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame.")
    if maximum_frame is not None and end_frame > int(maximum_frame):
        raise ValueError(
            f"end_frame {end_frame} exceeds the final video frame {int(maximum_frame)}."
        )

    manual_state = canonical_state(
        record.get("manual_state", record.get("state", record.get("override_state")))
    )
    start_time_s = _optional_float(record.get("start_time_s", record.get("start_s")))
    end_time_s = _optional_float(
        record.get(
            "end_time_s",
            record.get("end_s", record.get("stop_s")),
        )
    )
    if start_time_s is None and fps is not None and fps > 0:
        start_time_s = start_frame / fps
    if end_time_s is None and fps is not None and fps > 0:
        end_time_s = end_frame / fps
    created_at = str(record.get("created_at_utc") or utc_now())
    updated_at = str(record.get("updated_at_utc") or created_at)
    return {
        "annotation_id": str(record.get("annotation_id") or "").strip(),
        "revision": _optional_int(record.get("revision")),
        "video_id": str(record.get("video_id") or "").strip(),
        "video_path": str(record.get("video_path") or "").strip(),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "manual_state": manual_state,
        "reviewer": str(record.get("reviewer") or "").strip(),
        "notes": str(record.get("notes") or "").strip(),
        "created_at_utc": created_at,
        "updated_at_utc": updated_at,
    }


def validate_segments(
    segments: Iterable[dict[str, Any]],
    *,
    fps: float | None = None,
    maximum_frame: int | None = None,
) -> list[dict[str, Any]]:
    normalized = [
        normalize_segment(item, fps=fps, maximum_frame=maximum_frame) for item in segments
    ]
    normalized.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
    used_ids: set[str] = set()
    next_id = 1
    previous: dict[str, Any] | None = None
    for item in normalized:
        if not item["annotation_id"]:
            while f"ann_{next_id:04d}" in used_ids:
                next_id += 1
            item["annotation_id"] = f"ann_{next_id:04d}"
            next_id += 1
        if item["annotation_id"] in used_ids:
            raise ValueError(f"Duplicate annotation_id: {item['annotation_id']}")
        used_ids.add(item["annotation_id"])
        if previous is not None and item["start_frame"] <= previous["end_frame"]:
            raise ValueError(
                "Manual state intervals may not overlap: "
                f"{previous['annotation_id']}={previous['start_frame']}-{previous['end_frame']} "
                f"and {item['annotation_id']}={item['start_frame']}-{item['end_frame']}."
            )
        previous = item
    return normalized


def read_segments_csv(
    path: str | Path,
    *,
    fps: float | None = None,
    maximum_frame: int | None = None,
) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return validate_segments(rows, fps=fps, maximum_frame=maximum_frame)


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "video"


def _allocate_session_dir(output_root: Path, video_id: str) -> Path:
    parent = output_root / _safe_stem(video_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / timestamp
    for index in range(1, 10000):
        selected = candidate if index == 1 else parent / f"{timestamp}_{index:03d}"
        try:
            selected.mkdir(parents=True, exist_ok=False)
            return selected
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate an annotation session directory under {parent}")


class ManualAnnotationStore:
    def __init__(
        self,
        session_dir: str | Path,
        provenance: dict[str, Any],
        *,
        session_id: str | None = None,
        current_revision: int = 0,
        current_csv_path: str | Path | None = None,
    ):
        self.session_dir = Path(session_dir)
        self.provenance = provenance
        self.session_id = session_id or self.session_dir.name
        self.current_revision = int(current_revision)
        self.current_csv_path = Path(current_csv_path) if current_csv_path else None

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        video_path: str | Path,
        *,
        video_info: dict[str, Any] | None = None,
        tracking_csv_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> "ManualAnnotationStore":
        provenance = build_source_provenance(
            video_path,
            video_info=video_info,
            tracking_csv_path=tracking_csv_path,
            config_path=config_path,
        )
        video_id = str(provenance["video"]["video_id"])
        session_dir = _allocate_session_dir(Path(output_root), video_id)
        return cls(
            session_dir,
            provenance,
            session_id=f"{video_id}__{session_dir.name}",
        )

    @classmethod
    def load(
        cls,
        revision_csv_path: str | Path,
    ) -> tuple["ManualAnnotationStore", list[dict[str, Any]], dict[str, Any]]:
        csv_path = Path(revision_csv_path).resolve()
        match = re.fullmatch(r"revision_(\d+)_manual_state_segments\.csv", csv_path.name)
        revision = int(match.group(1)) if match else 0
        manifest_path = csv_path.with_name(f"revision_{revision:04d}_manifest.json")
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_hash = (
                manifest.get("annotation_csv", {}).get("sha256")
                if isinstance(manifest.get("annotation_csv"), dict)
                else None
            )
            if expected_hash and sha256_file(csv_path) != str(expected_hash).upper():
                raise ValueError(f"Annotation CSV hash does not match its manifest: {csv_path}")
        provenance = manifest.get("source_provenance") or {}
        fps = _optional_float(provenance.get("video", {}).get("fps"))
        frame_count = _optional_int(provenance.get("video", {}).get("frame_count"))
        maximum_frame = frame_count - 1 if frame_count and frame_count > 0 else None
        segments = read_segments_csv(csv_path, fps=fps, maximum_frame=maximum_frame)
        store = cls(
            csv_path.parent,
            provenance,
            session_id=manifest.get("session_id") or csv_path.parent.name,
            current_revision=revision,
            current_csv_path=csv_path,
        )
        return store, segments, manifest

    def save_revision(
        self,
        segments: Iterable[dict[str, Any]],
        *,
        action: str,
    ) -> tuple[Path, Path, list[dict[str, Any]]]:
        video = self.provenance.get("video", {})
        fps = _optional_float(video.get("fps"))
        frame_count = _optional_int(video.get("frame_count"))
        maximum_frame = frame_count - 1 if frame_count and frame_count > 0 else None
        rows = validate_segments(segments, fps=fps, maximum_frame=maximum_frame)
        existing_revisions = []
        for path in self.session_dir.glob("revision_*_manual_state_segments.csv"):
            match = re.fullmatch(r"revision_(\d+)_manual_state_segments\.csv", path.name)
            if match:
                existing_revisions.append(int(match.group(1)))
        revision = max([self.current_revision, *existing_revisions], default=0) + 1
        saved_at = utc_now()
        video_id = str(video.get("video_id") or Path(str(video.get("path", "video"))).stem)
        video_path = str(video.get("path") or "")
        serialized_rows = []
        for row in rows:
            serialized = dict(row)
            serialized["revision"] = revision
            serialized["video_id"] = serialized["video_id"] or video_id
            serialized["video_path"] = serialized["video_path"] or video_path
            serialized_rows.append(serialized)

        csv_path = self.session_dir / f"revision_{revision:04d}_manual_state_segments.csv"
        with csv_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANUAL_STATE_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(serialized_rows)

        manifest_path = self.session_dir / f"revision_{revision:04d}_manifest.json"
        manifest = {
            "format_version": ANNOTATION_FORMAT_VERSION,
            "session_id": self.session_id,
            "revision": revision,
            "parent_revision": self.current_revision or None,
            "saved_at_utc": saved_at,
            "action": str(action),
            "frame_interval_convention": FRAME_INTERVAL_CONVENTION,
            "direction_convention": DIRECTION_CONVENTION,
            "allowed_manual_states": [
                {
                    "key": item.key,
                    "label": item.label,
                    "description": item.description,
                }
                for item in STATE_SPECS
            ],
            "scientific_boundary": (
                "Manual visual annotations describe observed motion only. "
                "irregular_candidate is not a torus or chaos conclusion."
            ),
            "source_provenance": self.provenance,
            "segment_count": len(serialized_rows),
            "annotation_csv": {
                "path": csv_path.name,
                "sha256": sha256_file(csv_path),
                "columns": list(MANUAL_STATE_COLUMNS),
            },
        }
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")

        self.current_revision = revision
        self.current_csv_path = csv_path
        return csv_path, manifest_path, serialized_rows
