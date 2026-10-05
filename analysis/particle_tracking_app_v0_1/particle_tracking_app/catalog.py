from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .core import PROJECT_ROOT


CATALOG_PATH = PROJECT_ROOT / "data" / "experiment_catalog.json"


@dataclass
class AnalysisRunRecord:
    run_id: str
    output_dir: str
    created_at: str
    status: str = "complete"
    input_tracking_sha256: str = ""
    config_sha256: str = ""


@dataclass
class VideoRecord:
    video_id: str
    video_path: str
    tracking_csv: str = ""
    model_path: str = ""
    fps: float | None = None
    fps_source: str = "pending_confirmation"
    expected_video_sha256: str = ""
    expected_tracking_sha256: str = ""
    preserved_derived_tracks: str = ""
    notes: str = ""
    analysis_runs: list[AnalysisRunRecord] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    experiment_id: str
    display_name: str
    experiment_date: str = ""
    date_status: str = "pending_confirmation"
    raw_dir: str = ""
    calibration_images: list[str] = field(default_factory=list)
    calibration_output: str = ""
    trap_x_px: float | None = None
    trap_y_px: float | None = None
    trap_se_x_px: float | None = None
    trap_se_y_px: float | None = None
    trap_center_source: str = "pending_confirmation"
    pixels_per_mm: float | None = None
    pixel_scale_status: str = "pending_confirmation"
    notes: str = ""
    videos: list[VideoRecord] = field(default_factory=list)


@dataclass
class ExperimentCatalog:
    schema_version: int = 1
    experiments: list[ExperimentRecord] = field(default_factory=list)

    def experiment(self, experiment_id: str) -> ExperimentRecord:
        for record in self.experiments:
            if record.experiment_id == experiment_id:
                return record
        raise KeyError(experiment_id)

    def video(self, experiment_id: str, video_id: str) -> VideoRecord:
        for record in self.experiment(experiment_id).videos:
            if record.video_id == video_id:
                return record
        raise KeyError(f"{experiment_id}/{video_id}")

    def validate(self) -> None:
        experiment_ids: set[str] = set()
        for experiment in self.experiments:
            if not experiment.experiment_id.strip():
                raise ValueError("Experiment ID is required")
            if experiment.experiment_id in experiment_ids:
                raise ValueError(f"Duplicate experiment ID: {experiment.experiment_id}")
            experiment_ids.add(experiment.experiment_id)
            video_ids: set[str] = set()
            for video in experiment.videos:
                if not video.video_id.strip():
                    raise ValueError(f"Video ID is required in {experiment.experiment_id}")
                if video.video_id in video_ids:
                    raise ValueError(f"Duplicate video ID in {experiment.experiment_id}: {video.video_id}")
                video_ids.add(video.video_id)


def _run_from_dict(data: dict) -> AnalysisRunRecord:
    return AnalysisRunRecord(**data)


def _video_from_dict(data: dict) -> VideoRecord:
    values = dict(data)
    values["analysis_runs"] = [_run_from_dict(item) for item in values.get("analysis_runs", [])]
    return VideoRecord(**values)


def _experiment_from_dict(data: dict) -> ExperimentRecord:
    values = dict(data)
    values["videos"] = [_video_from_dict(item) for item in values.get("videos", [])]
    return ExperimentRecord(**values)


def load_catalog(path: Path = CATALOG_PATH) -> ExperimentCatalog:
    if not path.exists():
        return ExperimentCatalog()
    data = json.loads(path.read_text(encoding="utf-8"))
    catalog = ExperimentCatalog(
        schema_version=int(data.get("schema_version", 1)),
        experiments=[_experiment_from_dict(item) for item in data.get("experiments", [])],
    )
    catalog.validate()
    return catalog


def save_catalog(catalog: ExperimentCatalog, path: Path = CATALOG_PATH) -> None:
    catalog.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(catalog), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
