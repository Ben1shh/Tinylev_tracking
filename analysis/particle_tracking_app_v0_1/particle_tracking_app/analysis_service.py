from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .catalog import ExperimentRecord, VideoRecord
from .core import PROJECT_ROOT

try:
    from analysis.dynamic_pair_analysis.pipeline import (
        AnalysisConfig,
        apply_manual_annotations,
        classify_states,
        clean_tracking,
        derive_geometry,
        summarize_segments,
        wrap_angle,
    )
    from analysis.dynamic_pair_analysis.phenomenological_center_report import (
        build_tracks,
        fit_phase_locked_orbit,
    )
except ModuleNotFoundError:
    import sys

    analysis_root = PROJECT_ROOT / "analysis"
    if str(analysis_root) not in sys.path:
        sys.path.insert(0, str(analysis_root))
    from dynamic_pair_analysis.pipeline import (  # type: ignore[no-redef]
        AnalysisConfig,
        apply_manual_annotations,
        classify_states,
        clean_tracking,
        derive_geometry,
        summarize_segments,
        wrap_angle,
    )
    from dynamic_pair_analysis.phenomenological_center_report import (  # type: ignore[no-redef]
        build_tracks,
        fit_phase_locked_orbit,
    )


REQUIRED_TRACKING_COLUMNS = {
    "frame",
    "time_s",
    "status",
    "selected_count",
    "particle_1_cx_px",
    "particle_1_cy_px",
    "particle_1_radius_px",
    "particle_1_conf",
    "particle_2_cx_px",
    "particle_2_cy_px",
    "particle_2_radius_px",
    "particle_2_conf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def annotation_path(experiment_id: str, video_id: str) -> Path:
    return PROJECT_ROOT / "outputs" / "experiment_analysis" / experiment_id / video_id / "manual_annotations.csv"


def ensure_annotation_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=["video_id", "start_frame", "end_frame", "override_state", "reviewer", "notes"]
    ).to_csv(path, index=False)


def audit_video(experiment: ExperimentRecord, video: VideoRecord) -> dict:
    video_path = Path(video.video_path)
    tracking_path = Path(video.tracking_csv) if video.tracking_csv else None
    result: dict = {
        "experiment_id": experiment.experiment_id,
        "video_id": video.video_id,
        "video_path": str(video_path),
        "video_exists": video_path.is_file(),
        "tracking_csv": str(tracking_path) if tracking_path else "",
        "tracking_exists": bool(tracking_path and tracking_path.is_file()),
        "phase_a_status": "unavailable",
        "issues": [],
        "pixel_scale_status": experiment.pixel_scale_status,
        "trap_center_source": experiment.trap_center_source,
    }
    if video_path.is_file():
        result["video_sha256"] = sha256(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            result["video_frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result["video_fps_metadata"] = float(cap.get(cv2.CAP_PROP_FPS))
        else:
            result["issues"].append("video_cannot_be_opened")
        cap.release()
        if video.expected_video_sha256 and result["video_sha256"] != video.expected_video_sha256.upper():
            result["issues"].append("video_hash_mismatch")
    else:
        result["issues"].append("video_missing")

    if tracking_path and tracking_path.is_file():
        result["tracking_sha256"] = sha256(tracking_path)
        if video.expected_tracking_sha256 and result["tracking_sha256"] != video.expected_tracking_sha256.upper():
            result["issues"].append("tracking_provenance_mismatch")
        try:
            table = pd.read_csv(tracking_path)
            result["tracking_rows"] = int(len(table))
            missing = sorted(REQUIRED_TRACKING_COLUMNS - set(table.columns))
            result["missing_columns"] = missing
            if missing:
                result["issues"].append("tracking_missing_required_columns")
            if len(table) <= 1:
                result["issues"].append("tracking_truncated_or_empty")
            if "frame" in table and len(table) > 1:
                frames = pd.to_numeric(table["frame"], errors="coerce").to_numpy(float)
                if not np.all(np.diff(frames[np.isfinite(frames)]) > 0):
                    result["issues"].append("frame_index_not_strictly_increasing")
            if result.get("video_frame_count") and len(table) not in {
                int(result["video_frame_count"]),
                max(0, int(result["video_frame_count"]) - 1),
            }:
                result["issues"].append("tracking_video_row_count_mismatch")
        except Exception as error:
            result["issues"].append(f"tracking_read_error:{error}")
    else:
        result["issues"].append("tracking_missing")

    blocking = {
        "video_missing",
        "tracking_missing",
        "tracking_missing_required_columns",
        "tracking_truncated_or_empty",
        "tracking_provenance_mismatch",
        "tracking_video_row_count_mismatch",
    }
    result["analysis_allowed"] = not any(issue.split(":", 1)[0] in blocking for issue in result["issues"])
    result["phase_a_status"] = "ready" if result["analysis_allowed"] else "blocked"
    return result


def make_analysis_config(experiment: ExperimentRecord) -> AnalysisConfig:
    values = (experiment.trap_x_px, experiment.trap_y_px, experiment.trap_se_x_px, experiment.trap_se_y_px)
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("Calibration trap center and its x/y standard errors are required for analysis.")
    if experiment.trap_se_x_px < 0 or experiment.trap_se_y_px < 0:
        raise ValueError("Calibration standard errors cannot be negative.")
    return AnalysisConfig(
        trap_x_px=float(experiment.trap_x_px),
        trap_y_px=float(experiment.trap_y_px),
        trap_se_x_px=float(experiment.trap_se_x_px),
        trap_se_y_px=float(experiment.trap_se_y_px),
    )


def reclassify_preserved_derived(
    derived: pd.DataFrame,
    experiment: ExperimentRecord,
    video: VideoRecord,
) -> pd.DataFrame:
    """Apply current state rules to a preserved table without rewriting it."""
    refreshed = classify_states(derived, make_analysis_config(experiment))
    annotations = annotation_path(experiment.experiment_id, video.video_id)
    ensure_annotation_file(annotations)
    return apply_manual_annotations(refreshed, annotations)


def create_run_dir(experiment_id: str, video_id: str) -> tuple[str, Path]:
    root = PROJECT_ROOT / "outputs" / "experiment_analysis" / experiment_id / video_id
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = base
    counter = 2
    while (root / run_id).exists():
        run_id = f"{base}_{counter:02d}"
        counter += 1
    output = root / run_id
    output.mkdir()
    return run_id, output


def run_phase_ab(experiment: ExperimentRecord, video: VideoRecord) -> dict:
    audit = audit_video(experiment, video)
    if not audit["analysis_allowed"]:
        raise RuntimeError("Phase A blocked: " + "; ".join(audit["issues"]))
    config = make_analysis_config(experiment)
    raw = pd.read_csv(video.tracking_csv)
    clean, clean_metrics = clean_tracking(raw, config)
    derived = classify_states(derive_geometry(clean, config), config)
    annotations = annotation_path(experiment.experiment_id, video.video_id)
    ensure_annotation_file(annotations)
    derived = apply_manual_annotations(derived, annotations)
    segments = summarize_segments(derived, config)
    # Phase D validation is intentionally out of scope for the managed app v0.4.
    # Never carry the legacy pipeline's preliminary Rlc eligibility into these runs.
    if "spinning_qualified_for_rlc" in segments:
        segments["spinning_qualified_for_rlc"] = False
    if "spinning_exclusion_reasons" in segments:
        segments["spinning_exclusion_reasons"] = segments["spinning_exclusion_reasons"].fillna("").map(
            lambda value: ";".join(filter(None, [str(value), "phase_d_validation_not_performed"]))
        )
    run_id, output_dir = create_run_dir(experiment.experiment_id, video.video_id)
    clean.to_csv(output_dir / "clean_tracks.csv", index=False)
    derived.to_csv(output_dir / "derived_tracks.csv", index=False)
    segments.to_csv(output_dir / "segments.csv", index=False)
    shutil.copy2(annotations, output_dir / "manual_annotations_snapshot.csv")
    (output_dir / "phase_a_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output_dir / "analysis_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    provenance = {
        "experiment_id": experiment.experiment_id,
        "video_id": video.video_id,
        "video_path": str(Path(video.video_path).resolve()),
        "video_sha256": audit.get("video_sha256", ""),
        "tracking_csv": str(Path(video.tracking_csv).resolve()),
        "tracking_csv_sha256": audit.get("tracking_sha256", ""),
        "model_path": video.model_path,
        "model_sha256": sha256(Path(video.model_path)) if video.model_path and Path(video.model_path).is_file() else "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pixel_scale_status": experiment.pixel_scale_status,
        "formula_scope": "Phase A-B candidate analysis; no Rlc qualification",
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "audit": audit,
        "clean_metrics": clean_metrics,
        "derived": derived,
        "segments": segments,
        "tracking_sha256": audit.get("tracking_sha256", ""),
    }


def run_phase_c(
    derived: pd.DataFrame,
    experiment: ExperimentRecord,
    video: VideoRecord,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
) -> dict:
    segment = derived[derived["frame"].between(int(start_frame), int(end_frame)) & derived["valid"]].copy()
    if len(segment) < 20:
        raise ValueError("At least 20 valid frames are required for Phase C")
    psi_turns = abs(float(segment["psi_unwrapped_rad"].iloc[-1] - segment["psi_unwrapped_rad"].iloc[0])) / (2 * math.pi)
    if psi_turns < 0.5:
        raise ValueError("Selected interval contains less than half a pair-axis turn")
    z = segment["cm_x_px"].to_numpy(float) - 1j * segment["cm_y_px_image"].to_numpy(float)
    psi = segment["psi_rad"].to_numpy(float)
    center, amplitude = fit_phase_locked_orbit(z, psi)
    tracks = build_tracks(segment, center, amplitude)
    residual = tracks["phase_locked_residual_px"].to_numpy(float)
    baseline = np.abs(z - np.mean(z))
    denominator = float(np.sum(baseline**2))
    fit_r2 = float(1 - np.sum(residual**2) / denominator) if denominator > 0 else math.nan
    phase_dir = output_dir / f"phase_c_{start_frame}_{end_frame}"
    phase_dir.mkdir(parents=True, exist_ok=False)
    tracks.to_csv(phase_dir / "phenomenological_tracks.csv", index=False)
    calibration_center = complex(float(experiment.trap_x_px), -float(experiment.trap_y_px))
    summary = {
        "experiment_id": experiment.experiment_id,
        "video_id": video.video_id,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "valid_frame_count": int(len(segment)),
        "psi_turns": psi_turns,
        "calibration_center_x_px": calibration_center.real,
        "calibration_center_y_image_px": -calibration_center.imag,
        "phenomenological_center_x_px": center.real,
        "phenomenological_center_y_image_px": -center.imag,
        "center_offset_from_calibration_px": abs(center - calibration_center),
        "phase_orbit_radius_px": abs(amplitude),
        "phase_offset_deg": float(np.degrees(np.angle(amplitude))),
        "median_residual_px": float(np.median(residual)),
        "rmse_px": float(np.sqrt(np.mean(residual**2))),
        "fit_r2": fit_r2,
        "median_phen_r_px": float(tracks["phen_r_px"].median()),
        "median_phen_r_cos_phi_px": float(tracks["phen_r_cos_phi_px"].median()),
        "interpretation": "phenomenological fit only; center is not independently validated and cannot qualify Rlc",
        "phase_d_validation": "not_performed",
    }
    (phase_dir / "phase_c_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot(z.real, -z.imag, ".", ms=2, alpha=0.55, label="CM")
    angle = np.linspace(0, 2 * np.pi, 361)
    circle = center + amplitude * np.exp(1j * angle)
    axis.plot(circle.real, -circle.imag, color="#D62728", lw=1.5, label="phase-locked fit")
    axis.scatter([calibration_center.real], [-calibration_center.imag], marker="+", s=160, c="black", label="calibration center")
    axis.scatter([center.real], [-center.imag], marker="x", s=100, c="#D62728", label="phenomenological center")
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Phenomenological fit only - not an Rlc validation")
    figure.tight_layout()
    figure.savefig(phase_dir / "phase_c_geometry.png", dpi=180)
    plt.close(figure)
    return {"summary": summary, "tracks": tracks, "output_dir": str(phase_dir)}


def _phase_d_segment(derived: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    required = {
        "frame", "time_s", "valid", "cm_x_px", "cm_y_px_image", "psi_rad", "psi_unwrapped_rad"
    }
    missing = sorted(required - set(derived.columns))
    if missing:
        raise ValueError(f"Phase D input is missing required columns: {missing}")
    selected = derived[derived["frame"].between(int(start_frame), int(end_frame))].copy()
    if selected.empty:
        raise ValueError("Selected Phase D interval contains no frames")
    selected = selected.sort_values("frame").reset_index().rename(columns={"index": "source_row_index"})
    valid_values = selected["valid"]
    if valid_values.dtype == bool:
        valid_mask = valid_values.to_numpy(bool)
    else:
        valid_mask = valid_values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).to_numpy(bool)
    selected["phase_d_valid"] = (
        valid_mask
        & selected[["frame", "time_s", "cm_x_px", "cm_y_px_image", "psi_rad", "psi_unwrapped_rad"]]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )
    return selected


def _phase_d_arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    z = table["cm_x_px"].to_numpy(float) - 1j * table["cm_y_px_image"].to_numpy(float)
    psi = table["psi_rad"].to_numpy(float)
    return z, psi


def _circular_metrics(angle: np.ndarray) -> tuple[float, float, float]:
    finite = angle[np.isfinite(angle)]
    if finite.size == 0:
        return math.nan, math.nan, math.nan
    resultant = complex(np.mean(np.exp(1j * finite)))
    length = abs(resultant)
    spread = math.sqrt(max(0.0, -2.0 * math.log(max(length, 1e-12))))
    return float(np.angle(resultant)), float(length), float(spread)


def _phase_d_metrics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    center: complex,
    amplitude: complex,
) -> tuple[dict, pd.DataFrame]:
    train_z, train_psi = _phase_d_arrays(train)
    validation_z, validation_psi = _phase_d_arrays(validation)
    prediction = center + amplitude * np.exp(1j * validation_psi)
    residual = np.abs(validation_z - prediction)
    train_prediction = center + amplitude * np.exp(1j * train_psi)
    train_residual = np.abs(train_z - train_prediction)
    baseline = np.abs(validation_z - np.mean(train_z)) ** 2
    squared_error = residual**2
    denominator = float(np.sum(baseline))
    r2 = float(1 - np.sum(squared_error) / denominator) if denominator > 0 else math.nan
    orbit_radius = abs(amplitude)
    validation_radius = np.abs(validation_z - center)
    train_phase = wrap_angle(np.angle(train_z - center) - train_psi)
    validation_phase = wrap_angle(np.angle(validation_z - center) - validation_psi)
    train_phase_mean, train_phase_r, train_phase_spread = _circular_metrics(train_phase)
    validation_phase_mean, validation_phase_r, validation_phase_spread = _circular_metrics(validation_phase)
    metrics = {
        "train_frame_count": int(len(train)),
        "validation_frame_count": int(len(validation)),
        "train_start_frame": int(train["frame"].min()),
        "train_end_frame": int(train["frame"].max()),
        "validation_start_frame": int(validation["frame"].min()),
        "validation_end_frame": int(validation["frame"].max()),
        "train_center_x_px": float(center.real),
        "train_center_y_image_px": float(-center.imag),
        "train_amplitude_real_px": float(amplitude.real),
        "train_amplitude_imag_cart_px": float(amplitude.imag),
        "train_orbit_radius_px": float(orbit_radius),
        "train_phase_offset_deg": float(np.degrees(np.angle(amplitude))),
        "train_rmse_px": float(np.sqrt(np.mean(train_residual**2))),
        "validation_r2_against_train_mean": r2,
        "validation_rmse_px": float(np.sqrt(np.mean(squared_error))),
        "validation_median_residual_px": float(np.median(residual)),
        "validation_p95_residual_px": float(np.quantile(residual, 0.95)),
        "validation_median_residual_over_orbit_radius": float(np.median(residual) / orbit_radius) if orbit_radius > 0 else math.nan,
        "validation_p95_residual_over_orbit_radius": float(np.quantile(residual, 0.95) / orbit_radius) if orbit_radius > 0 else math.nan,
        "validation_median_radius_px": float(np.median(validation_radius)),
        "validation_p05_radius_px": float(np.quantile(validation_radius, 0.05)),
        "validation_p95_radius_px": float(np.quantile(validation_radius, 0.95)),
        "train_phase_circular_mean_deg": float(np.degrees(train_phase_mean)),
        "train_phase_resultant_length": train_phase_r,
        "train_phase_circular_spread_deg": float(np.degrees(train_phase_spread)),
        "validation_phase_circular_mean_deg": float(np.degrees(validation_phase_mean)),
        "validation_phase_resultant_length": validation_phase_r,
        "validation_phase_circular_spread_deg": float(np.degrees(validation_phase_spread)),
        "validation_phase_shift_from_train_deg": float(np.degrees(wrap_angle(np.array([validation_phase_mean - train_phase_mean]))[0])),
        "validation_refit_performed": False,
    }
    tracks = pd.DataFrame(
        {
            "frame": validation["frame"].to_numpy(int),
            "time_s": validation["time_s"].to_numpy(float),
            "cm_x_px": validation_z.real,
            "cm_y_image_px": -validation_z.imag,
            "psi_rad": validation_psi,
            "fixed_center_x_px": center.real,
            "fixed_center_y_image_px": -center.imag,
            "fixed_amplitude_real_px": amplitude.real,
            "fixed_amplitude_imag_cart_px": amplitude.imag,
            "prediction_x_px": prediction.real,
            "prediction_y_image_px": -prediction.imag,
            "residual_px": residual,
            "radius_from_fixed_center_px": validation_radius,
            "phase_offset_from_fixed_center_rad": validation_phase,
            "validation_refit_performed": False,
        }
    )
    return metrics, tracks


def _valid_turns(table: pd.DataFrame) -> float:
    values = table["psi_unwrapped_rad"].to_numpy(float)
    return abs(float(values[-1] - values[0])) / (2 * math.pi) if len(values) >= 2 else 0.0


def _complete_cycles(selected: pd.DataFrame) -> list[pd.DataFrame]:
    """Return complete monotonic 2-pi excursions without crossing gaps or reversals."""
    valid_positions = np.flatnonzero(selected["phase_d_valid"].to_numpy(bool))
    if valid_positions.size < 2:
        return []
    blocks: list[list[int]] = []
    current = [int(valid_positions[0])]
    for position in valid_positions[1:]:
        previous = current[-1]
        consecutive = (
            int(position) == previous + 1
            and int(selected.loc[position, "frame"]) == int(selected.loc[previous, "frame"]) + 1
            and float(selected.loc[position, "time_s"]) > float(selected.loc[previous, "time_s"])
        )
        if consecutive:
            current.append(int(position))
        else:
            if len(current) >= 2:
                blocks.append(current)
            current = [int(position)]
    if len(current) >= 2:
        blocks.append(current)

    monotonic_runs: list[list[int]] = []
    for block in blocks:
        psi = selected.loc[block, "psi_unwrapped_rad"].to_numpy(float)
        increments = np.diff(psi)
        nonzero = increments[np.nonzero(increments)]
        if nonzero.size == 0:
            continue
        run = [block[0]]
        direction = int(np.sign(nonzero[0]))
        for offset, increment in enumerate(increments, start=1):
            sign = int(np.sign(increment))
            if sign == 0 or sign != direction:
                if len(run) >= 2:
                    monotonic_runs.append(run)
                run = [block[offset - 1], block[offset]] if sign != 0 else [block[offset]]
                if sign != 0:
                    direction = sign
            else:
                run.append(block[offset])
        if len(run) >= 2:
            monotonic_runs.append(run)

    cycles: list[pd.DataFrame] = []
    cycle_number = 0
    for run_number, positions in enumerate(monotonic_runs):
        run = selected.loc[positions].copy()
        psi = run["psi_unwrapped_rad"].to_numpy(float)
        direction = 1.0 if psi[-1] > psi[0] else -1.0
        progress = direction * (psi - psi[0]) / (2 * math.pi)
        complete_count = int(math.floor(progress[-1] + 1e-12))
        for local_cycle in range(complete_count):
            start_candidates = np.flatnonzero(progress >= local_cycle)
            end_candidates = np.flatnonzero(progress >= local_cycle + 1)
            if start_candidates.size == 0 or end_candidates.size == 0:
                continue
            start = int(start_candidates[0])
            end = int(end_candidates[0])
            cycle = run.iloc[start : end + 1].copy()
            if len(cycle) < 5:
                continue
            cycle["phase_d_cycle_index"] = cycle_number
            cycle["phase_d_monotonic_run"] = run_number
            cycle["phase_d_cycle_direction"] = int(direction)
            cycles.append(cycle)
            cycle_number += 1
    return cycles


def _phase_d_provenance(video: VideoRecord, derived_source_path: Path | None) -> dict:
    tracking_path = Path(video.tracking_csv) if video.tracking_csv else None
    current_tracking_hash = sha256(tracking_path) if tracking_path and tracking_path.is_file() else ""
    expected_tracking_hash = video.expected_tracking_sha256.upper() if video.expected_tracking_sha256 else ""
    tracking_rows: int | None = None
    video_frame_count: int | None = None
    if tracking_path and tracking_path.is_file():
        try:
            tracking_rows = int(len(pd.read_csv(tracking_path, usecols=["frame"])))
        except Exception:
            tracking_rows = None
    video_path = Path(video.video_path) if video.video_path else None
    if video_path and video_path.is_file():
        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    row_count_matches = bool(
        tracking_rows is not None
        and video_frame_count is not None
        and tracking_rows in {video_frame_count, max(0, video_frame_count - 1)}
    )
    if not tracking_path or not tracking_path.is_file():
        tracking_status = "missing"
    elif expected_tracking_hash and current_tracking_hash != expected_tracking_hash:
        tracking_status = "provenance_mismatch"
    elif not row_count_matches:
        tracking_status = "row_count_mismatch"
    else:
        tracking_status = "matches_recorded_input"
    source_path = derived_source_path.resolve() if derived_source_path else None
    preserved_path = Path(video.preserved_derived_tracks).resolve() if video.preserved_derived_tracks else None
    source_kind = "preserved_derived" if source_path and preserved_path and source_path == preserved_path else "analysis_run_derived"
    return {
        "derived_source_kind": source_kind,
        "derived_source_path": str(source_path) if source_path else "in_memory_unspecified",
        "derived_source_sha256": sha256(source_path) if source_path and source_path.is_file() else "",
        "current_tracking_csv": str(tracking_path.resolve()) if tracking_path else "",
        "expected_tracking_sha256": expected_tracking_hash,
        "current_tracking_sha256": current_tracking_hash,
        "current_tracking_rows": tracking_rows,
        "video_frame_count": video_frame_count,
        "tracking_video_row_count_matches": row_count_matches,
        "upstream_tracking_status": tracking_status,
    }


def run_phase_d_core(
    derived: pd.DataFrame,
    experiment: ExperimentRecord,
    video: VideoRecord,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
    derived_source_path: Path | None = None,
) -> dict:
    """Run within-segment holdouts without fitting on any validation frames."""
    selected = _phase_d_segment(derived, start_frame, end_frame)
    valid = selected[selected["phase_d_valid"]].copy()
    phase_dir = output_dir / f"phase_d_{start_frame}_{end_frame}"
    phase_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []
    validation_tracks: list[pd.DataFrame] = []
    half_available = False
    cycle_available = False

    if len(valid) >= 40:
        midpoint = len(valid) // 2
        first = valid.iloc[:midpoint].copy()
        second = valid.iloc[midpoint:].copy()
        if min(len(first), len(second)) >= 20 and min(_valid_turns(first), _valid_turns(second)) >= 0.5:
            half_available = True
            for fold_id, train, validation in (
                ("first_to_second", first, second),
                ("second_to_first", second, first),
            ):
                train_z, train_psi = _phase_d_arrays(train)
                center, amplitude = fit_phase_locked_orbit(train_z, train_psi)
                metrics, tracks = _phase_d_metrics(train, validation, center, amplitude)
                records.append({"method": "temporal_half_holdout", "fold_id": fold_id, **metrics})
                tracks.insert(0, "fold_id", fold_id)
                tracks.insert(0, "method", "temporal_half_holdout")
                validation_tracks.append(tracks)

    cycles = _complete_cycles(selected)
    if len(cycles) >= 2:
        for held_out, validation in enumerate(cycles):
            train_parts = [cycle for index, cycle in enumerate(cycles) if index != held_out]
            train = pd.concat(train_parts, ignore_index=True)
            if len(train) < 20 or len(validation) < 5:
                continue
            train_z, train_psi = _phase_d_arrays(train)
            center, amplitude = fit_phase_locked_orbit(train_z, train_psi)
            metrics, tracks = _phase_d_metrics(train, validation, center, amplitude)
            fold_id = f"leave_cycle_{held_out}_out"
            records.append(
                {
                    "method": "leave_one_cycle_out",
                    "fold_id": fold_id,
                    "held_out_cycle_index": held_out,
                    "held_out_cycle_direction": int(validation["phase_d_cycle_direction"].iloc[0]),
                    "held_out_monotonic_run": int(validation["phase_d_monotonic_run"].iloc[0]),
                    **metrics,
                }
            )
            tracks.insert(0, "held_out_cycle_index", held_out)
            tracks.insert(0, "fold_id", fold_id)
            tracks.insert(0, "method", "leave_one_cycle_out")
            validation_tracks.append(tracks)
        cycle_available = any(record["method"] == "leave_one_cycle_out" for record in records)

    validation = pd.DataFrame.from_records(records)
    if validation_tracks:
        tracks_output = pd.concat(validation_tracks, ignore_index=True)
    else:
        tracks_output = pd.DataFrame(
            columns=["method", "fold_id", "frame", "time_s", "validation_refit_performed"]
        )
    validation.to_csv(phase_dir / "center_cross_validation.csv", index=False)
    tracks_output.to_csv(phase_dir / "phase_d_validation_tracks.csv", index=False)

    if not half_available:
        status = "insufficient_for_half_holdout"
    elif not cycle_available:
        status = "half_holdout_complete_cycle_holdout_unavailable"
    else:
        status = "core_cross_validation_complete"
    provenance = _phase_d_provenance(video, derived_source_path)
    summary = {
        "experiment_id": experiment.experiment_id,
        "video_id": video.video_id,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "selected_frame_count": int(len(selected)),
        "valid_frame_count": int(len(valid)),
        "invalid_or_nonfinite_frame_count": int(len(selected) - len(valid)),
        "psi_turns_over_valid_endpoints": _valid_turns(valid) if len(valid) >= 2 else 0.0,
        "complete_cycle_count": int(len(cycles)),
        "temporal_half_holdout_available": half_available,
        "cycle_holdout_status": "complete" if cycle_available else "cycle_holdout_unavailable",
        "phase_d_status": status,
        "validation_refit_performed": False,
        "pixel_scale_status": experiment.pixel_scale_status,
        "interpretation": "within-segment predictive validation only; no trapping-center or Rlc qualification",
        "promotion_status": "manual_review_and_later_cross_burst_or_slow_static_validation_required",
        **provenance,
    }
    (phase_dir / "phase_d_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    z, _psi = _phase_d_arrays(valid) if len(valid) else (np.array([], complex), np.array([], float))
    axes[0, 0].plot(z.real, -z.imag, ".", ms=2, alpha=0.5, label="selected valid CM")
    half_rows = validation[validation.get("method", pd.Series(dtype=str)).eq("temporal_half_holdout")] if not validation.empty else validation
    for row in half_rows.itertuples():
        axes[0, 0].scatter(row.train_center_x_px, row.train_center_y_image_px, marker="x", s=70, label=row.fold_id)
    axes[0, 0].set_aspect("equal"); axes[0, 0].invert_yaxis(); axes[0, 0].set_title("CM and training-only centers")
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.2)
    if not validation.empty:
        labels = validation["fold_id"].astype(str)
        axes[0, 1].bar(labels, validation["validation_r2_against_train_mean"])
        axes[1, 0].bar(labels, validation["validation_median_residual_over_orbit_radius"])
        axes[1, 1].bar(labels, validation["validation_phase_resultant_length"])
        for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
            axis.tick_params(axis="x", rotation=70, labelsize=7); axis.grid(axis="y", alpha=0.2)
    else:
        for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
            axis.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=axis.transAxes)
    axes[0, 1].set_title("Held-out R-squared")
    axes[1, 0].set_title("Median residual / training orbit radius")
    axes[1, 1].set_title("Held-out phase resultant length")
    figure.suptitle(f"Phase D core validation: {video.video_id} {start_frame}-{end_frame}\n{status}")
    figure.tight_layout()
    figure.savefig(phase_dir / "phase_d_validation.png", dpi=180)
    plt.close(figure)
    return {
        "summary": summary,
        "cross_validation": validation,
        "tracks": tracks_output,
        "output_dir": str(phase_dir),
    }
