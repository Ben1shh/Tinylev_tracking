from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORMULA_VERSION = "2026-07-05 manuscript Eq. 32 and exact projection relation"
STANDARD_MAPPING = {
    "frame": "frame",
    "time_s": "time_s",
    "particle_1_cx_px": "raw_particle_1_x_px",
    "particle_1_cy_px": "raw_particle_1_y_px",
    "particle_1_radius_px": "raw_particle_1_radius_px",
    "particle_1_conf": "raw_particle_1_confidence",
    "particle_2_cx_px": "raw_particle_2_x_px",
    "particle_2_cy_px": "raw_particle_2_y_px",
    "particle_2_radius_px": "raw_particle_2_radius_px",
    "particle_2_conf": "raw_particle_2_confidence",
    "status": "raw_detection_status",
    "selected_count": "raw_selected_count",
}


@dataclass
class AnalysisConfig:
    trap_x_px: float | None = None
    trap_y_px: float | None = None
    trap_se_x_px: float | None = None
    trap_se_y_px: float | None = None
    confidence_floor: float = 0.5
    roi_x_min: float = 300.0
    roi_x_max: float = 900.0
    roi_y_min: float = 250.0
    roi_y_max: float = 590.0
    roi_edge_margin_px: float = 5.0
    coherent_motion_max_pair_angle_step_deg: float = 90.0
    smoothing_window_s: float = 0.25
    classification_window_s: float = 2.0
    spinning_window_s: float = 0.5
    minimum_state_duration_s: float = 1.0
    spinning_sign_consistency: float = 0.8
    spinning_min_window_turns: float = 0.4
    spinning_candidate_min_range_turns: float = 0.15
    static_angle_std_deg: float = 10.0
    libration_min_range_deg: float = 10.0
    irregular_min_net_turns: float = 1.0
    theta_reliability_sigma: float = 3.0
    reversal_max_gap_s: float = 1.5
    spinning_omega_relative_tolerance: float = 0.2
    spinning_max_radial_drift_fraction: float = 0.1
    systematic_radius_uncertainty_px: float = 0.5
    monte_carlo_samples: int = 1000
    bootstrap_block_s: float = 0.5
    max_review_frames: int = 30
    formula_version: str = FORMULA_VERSION

    @classmethod
    def from_json(cls, path: Path) -> "AnalysisConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        roi = raw.pop("roi", None)
        allowed = set(cls.__dataclass_fields__)
        values = {key: value for key, value in raw.items() if key in allowed}
        if roi:
            values.update(
                roi_x_min=roi[0], roi_x_max=roi[1], roi_y_min=roi[2], roi_y_max=roi[3]
            )
        return cls(**values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def robust_mad(values: np.ndarray, floor: float = 1e-9) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return floor
    median = np.median(finite)
    return max(1.4826 * float(np.median(np.abs(finite - median))), floor)


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else math.nan


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def unwrap_with_gaps(angle: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full(angle.shape, np.nan, dtype=float)
    for start, stop in contiguous_runs(valid & np.isfinite(angle)):
        output[start:stop] = np.unwrap(angle[start:stop])
    return output


def local_smooth_derivative(
    values: np.ndarray, time: np.ndarray, valid: np.ndarray, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    smooth = np.full(values.shape, np.nan, dtype=float)
    derivative = np.full(values.shape, np.nan, dtype=float)
    if len(time) < 3:
        return smooth, derivative
    dt = float(np.nanmedian(np.diff(time)))
    width = max(5, int(round(window_s / dt)))
    if width % 2 == 0:
        width += 1
    half = width // 2
    for start, stop in contiguous_runs(valid & np.isfinite(values)):
        if stop - start < width:
            continue
        segment = values[start:stop]
        for local in range(half, len(segment) - half):
            lo, hi = local - half, local + half + 1
            x = time[start + lo : start + hi]
            y = segment[lo:hi]
            xc = x - x.mean()
            slope = float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))
            smooth[start + local] = float(y.mean())
            derivative[start + local] = slope
    return smooth, derivative


def _identity_viterbi(raw: pd.DataFrame, valid: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    fields = {
        "x1": raw["particle_1_cx_px"].to_numpy(float),
        "y1": raw["particle_1_cy_px"].to_numpy(float),
        "r1": raw["particle_1_radius_px"].to_numpy(float),
        "c1": raw["particle_1_conf"].to_numpy(float),
        "x2": raw["particle_2_cx_px"].to_numpy(float),
        "y2": raw["particle_2_cy_px"].to_numpy(float),
        "r2": raw["particle_2_radius_px"].to_numpy(float),
        "c2": raw["particle_2_conf"].to_numpy(float),
    }
    all_r = np.concatenate([fields["r1"][valid], fields["r2"][valid]])
    small_proto = float(np.median(np.minimum(fields["r1"][valid], fields["r2"][valid])))
    large_proto = float(np.median(np.maximum(fields["r1"][valid], fields["r2"][valid])))
    scale = max(robust_mad(all_r), 0.5)
    assignment = np.full(len(raw), -1, dtype=int)
    ambiguous = np.zeros(len(raw), dtype=bool)

    for start, stop in contiguous_runs(valid):
        length = stop - start
        costs = np.full((length, 2), np.inf)
        back = np.zeros((length, 2), dtype=np.int8)
        emissions = np.zeros((length, 2))
        for local, idx in enumerate(range(start, stop)):
            emissions[local, 0] = (
                abs(fields["r1"][idx] - small_proto) + abs(fields["r2"][idx] - large_proto)
            ) / scale
            emissions[local, 1] = (
                abs(fields["r2"][idx] - small_proto) + abs(fields["r1"][idx] - large_proto)
            ) / scale
        costs[0] = emissions[0]
        for local in range(1, length):
            idx = start + local
            prev = idx - 1
            coordinates = [
                ((fields["x1"][idx], fields["y1"][idx]), (fields["x2"][idx], fields["y2"][idx])),
                ((fields["x2"][idx], fields["y2"][idx]), (fields["x1"][idx], fields["y1"][idx])),
            ]
            previous = [
                ((fields["x1"][prev], fields["y1"][prev]), (fields["x2"][prev], fields["y2"][prev])),
                ((fields["x2"][prev], fields["y2"][prev]), (fields["x1"][prev], fields["y1"][prev])),
            ]
            for state in (0, 1):
                transitions = []
                for old_state in (0, 1):
                    move = sum(
                        math.hypot(coordinates[state][j][0] - previous[old_state][j][0], coordinates[state][j][1] - previous[old_state][j][1])
                        for j in (0, 1)
                    )
                    transitions.append(costs[local - 1, old_state] + move / 10.0)
                back[local, state] = int(np.argmin(transitions))
                costs[local, state] = emissions[local, state] + min(transitions)
        state = int(np.argmin(costs[-1]))
        for local in range(length - 1, -1, -1):
            idx = start + local
            assignment[idx] = state
            ambiguous[idx] = abs(emissions[local, 0] - emissions[local, 1]) < 1.0
            if local:
                state = int(back[local, state])

    output: dict[str, np.ndarray] = {}
    state_one = assignment == 1
    for suffix, first, second in (
        ("x_px", "x1", "x2"),
        ("y_px", "y1", "y2"),
        ("radius_px", "r1", "r2"),
        ("confidence", "c1", "c2"),
    ):
        output[f"small_{suffix}"] = np.where(state_one, fields[second], fields[first])
        output[f"large_{suffix}"] = np.where(state_one, fields[first], fields[second])
    output["assignment_swapped"] = state_one
    output["small_radius_prototype_px"] = np.full(len(raw), small_proto)
    output["large_radius_prototype_px"] = np.full(len(raw), large_proto)
    return output, ambiguous


def clean_tracking(raw: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, dict]:
    required = list(STANDARD_MAPPING)
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"Tracking CSV is missing required columns: {missing}")
    numeric = [name for name in required if name not in {"status"}]
    frame = raw.copy()
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    base_valid = (
        frame["status"].eq("ok").to_numpy()
        & frame["selected_count"].eq(2).to_numpy()
        & frame[["particle_1_cx_px", "particle_1_cy_px", "particle_1_radius_px", "particle_2_cx_px", "particle_2_cy_px", "particle_2_radius_px"]]
        .notna()
        .all(axis=1)
        .to_numpy()
    )
    assigned, ambiguous = _identity_viterbi(frame, base_valid)
    clean = pd.DataFrame({"frame": frame["frame"].astype(int), "time_s": frame["time_s"]})
    for key, value in assigned.items():
        clean[key] = value
    clean["raw_status"] = frame["status"].astype(str)
    clean["missing_detection"] = ~base_valid
    clean["low_confidence"] = (
        (clean["small_confidence"] < config.confidence_floor)
        | (clean["large_confidence"] < config.confidence_floor)
    ) & base_valid
    edge = config.roi_edge_margin_px
    clean["roi_boundary"] = base_valid & (
        (clean["small_x_px"] <= config.roi_x_min + edge)
        | (clean["small_x_px"] >= config.roi_x_max - edge)
        | (clean["large_x_px"] <= config.roi_x_min + edge)
        | (clean["large_x_px"] >= config.roi_x_max - edge)
        | (clean["small_y_px"] <= config.roi_y_min + edge)
        | (clean["small_y_px"] >= config.roi_y_max - edge)
        | (clean["large_y_px"] <= config.roi_y_min + edge)
        | (clean["large_y_px"] >= config.roi_y_max - edge)
    )
    clean["identity_ambiguous"] = ambiguous & base_valid

    ds = np.hypot(np.diff(clean["small_x_px"], prepend=np.nan), np.diff(clean["small_y_px"], prepend=np.nan))
    dl = np.hypot(np.diff(clean["large_x_px"], prepend=np.nan), np.diff(clean["large_y_px"], prepend=np.nan))
    drs = np.abs(np.diff(clean["small_radius_px"], prepend=np.nan))
    drl = np.abs(np.diff(clean["large_radius_px"], prepend=np.nan))
    center_threshold = max(float(np.nanmedian(np.r_[ds, dl])) + 8 * robust_mad(np.r_[ds, dl]), 5.0)
    radius_threshold = max(float(np.nanmedian(np.r_[drs, drl])) + 8 * robust_mad(np.r_[drs, drl]), 2.0)
    separation = np.hypot(
        clean["large_x_px"] - clean["small_x_px"],
        clean["large_y_px"] - clean["small_y_px"],
    ).to_numpy(float)
    separation_step = np.abs(np.diff(separation, prepend=np.nan))
    separation_step_threshold = max(
        float(np.nanmedian(separation_step)) + 8 * robust_mad(separation_step), 2.0
    )
    a = float(clean["small_radius_prototype_px"].iloc[0])
    b = float(clean["large_radius_prototype_px"].iloc[0])
    wa, wb = a**3, b**3
    cmx = (wa * clean["small_x_px"] + wb * clean["large_x_px"]) / (wa + wb)
    cmy = (wa * clean["small_y_px"] + wb * clean["large_y_px"]) / (wa + wb)
    cm_step = np.hypot(np.diff(cmx, prepend=np.nan), np.diff(cmy, prepend=np.nan))
    cm_step_threshold = max(
        float(np.nanmedian(cm_step)) + 8 * robust_mad(cm_step), 5.0
    )
    pair_angle = np.arctan2(
        -(clean["large_y_px"] - clean["small_y_px"]),
        clean["large_x_px"] - clean["small_x_px"],
    ).to_numpy(float)
    pair_angle_step = np.abs(wrap_angle(np.diff(pair_angle, prepend=np.nan)))
    consecutive_detection = np.r_[False, base_valid[1:] & base_valid[:-1]]
    large_individual_step = (ds > center_threshold) | (dl > center_threshold)
    coherent_fast_motion = (
        consecutive_detection
        & large_individual_step
        & (separation_step <= separation_step_threshold)
        & (pair_angle_step <= math.radians(config.coherent_motion_max_pair_angle_step_deg))
    )
    clean["coherent_fast_motion"] = coherent_fast_motion
    clean["center_jump"] = base_valid & large_individual_step & ~coherent_fast_motion
    clean["radius_jump"] = base_valid & ((drs > radius_threshold) | (drl > radius_threshold))
    clean["valid"] = ~clean[["missing_detection", "low_confidence", "identity_ambiguous", "center_jump", "radius_jump", "roi_boundary"]].any(axis=1)
    metrics = {
        "small_radius_prototype_px": float(clean["small_radius_prototype_px"].iloc[0]),
        "large_radius_prototype_px": float(clean["large_radius_prototype_px"].iloc[0]),
        "center_jump_threshold_px": center_threshold,
        "separation_step_threshold_px": separation_step_threshold,
        "cm_step_threshold_px": cm_step_threshold,
        "radius_jump_threshold_px": radius_threshold,
        "raw_ok_fraction": float(base_valid.mean()),
        "clean_valid_fraction": float(clean["valid"].mean()),
        "identity_swapped_frames": int(clean["assignment_swapped"].sum()),
        "identity_ambiguous_frames": int(clean["identity_ambiguous"].sum()),
        "coherent_fast_motion_frames": int(clean["coherent_fast_motion"].sum()),
    }
    return clean, metrics


def derive_geometry(clean: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    values = (config.trap_x_px, config.trap_y_px, config.trap_se_x_px, config.trap_se_y_px)
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("Calibration center and its x/y standard errors must be explicitly supplied.")
    if config.trap_se_x_px < 0 or config.trap_se_y_px < 0:
        raise ValueError("Calibration standard errors cannot be negative.")
    out = clean.copy()
    a = float(clean["small_radius_prototype_px"].iloc[0])
    b = float(clean["large_radius_prototype_px"].iloc[0])
    wa, wb = a**3, b**3
    out["cm_x_px"] = (wa * out["small_x_px"] + wb * out["large_x_px"]) / (wa + wb)
    out["cm_y_px_image"] = (wa * out["small_y_px"] + wb * out["large_y_px"]) / (wa + wb)
    out["r_x_px"] = out["cm_x_px"] - config.trap_x_px
    out["r_y_px"] = -(out["cm_y_px_image"] - config.trap_y_px)
    out["r_px"] = np.hypot(out["r_x_px"], out["r_y_px"])
    out["theta_rad"] = np.arctan2(out["r_y_px"], out["r_x_px"])
    nx = out["large_x_px"] - out["small_x_px"]
    ny = -(out["large_y_px"] - out["small_y_px"])
    out["pair_separation_px"] = np.hypot(nx, ny)
    out["n_x"] = nx / np.hypot(nx, ny)
    out["n_y"] = ny / np.hypot(nx, ny)
    out["psi_rad"] = np.arctan2(ny, nx)
    valid = out["valid"].to_numpy(bool)
    theta_min_r = config.theta_reliability_sigma * math.hypot(
        config.trap_se_x_px, config.trap_se_y_px
    )
    out["theta_reliable"] = valid & (out["r_px"].to_numpy(float) > theta_min_r)
    out["theta_unwrapped_rad"] = unwrap_with_gaps(
        out["theta_rad"].to_numpy(float), out["theta_reliable"].to_numpy(bool)
    )
    out["psi_unwrapped_rad"] = unwrap_with_gaps(out["psi_rad"].to_numpy(float), valid)
    out["phi_rad"] = wrap_angle(out["psi_rad"].to_numpy(float) - out["theta_rad"].to_numpy(float))
    time = out["time_s"].to_numpy(float)
    for source, prefix in (("psi_unwrapped_rad", "psi"), ("theta_unwrapped_rad", "theta"), ("r_px", "r")):
        derivative_valid = out["theta_reliable"].to_numpy(bool) if prefix == "theta" else valid
        smoothed, derivative = local_smooth_derivative(
            out[source].to_numpy(float), time, derivative_valid, config.smoothing_window_s
        )
        out[f"{prefix}_smooth"] = smoothed
        out[f"{prefix}_dot"] = derivative
    out["r_cos_phi_px"] = out["r_px"] * np.cos(out["phi_rad"])
    return out


def _rolling_metrics(derived: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    time = derived["time_s"].to_numpy(float)
    dt = float(np.nanmedian(np.diff(time)))
    width = max(5, int(round(config.classification_window_s / dt)))
    if width % 2 == 0:
        width += 1
    min_periods = max(3, width // 2)
    spin_width = max(5, int(round(config.spinning_window_s / dt)))
    if spin_width % 2 == 0:
        spin_width += 1
    spin_min_periods = max(3, spin_width // 2)
    result = pd.DataFrame(index=derived.index)
    psi = pd.Series(derived["psi_smooth"], index=derived.index)
    omega = pd.Series(derived["psi_dot"], index=derived.index)
    result["psi_range"] = psi.rolling(width, center=True, min_periods=min_periods).max() - psi.rolling(width, center=True, min_periods=min_periods).min()
    result["psi_std"] = psi.rolling(width, center=True, min_periods=min_periods).std()
    result["r_std"] = derived["r_px"].rolling(width, center=True, min_periods=min_periods).std()
    result["omega_abs_median"] = omega.abs().rolling(width, center=True, min_periods=min_periods).median()
    positive = (omega > 0).astype(float).rolling(spin_width, center=True, min_periods=spin_min_periods).mean()
    negative = (omega < 0).astype(float).rolling(spin_width, center=True, min_periods=spin_min_periods).mean()
    result["sign_consistency"] = np.maximum(positive, negative)
    result["net_turns"] = (psi.shift(-(spin_width // 2)) - psi.shift(spin_width // 2)).abs() / (2 * np.pi)
    result["spin_range_turns"] = (
        psi.rolling(spin_width, center=True, min_periods=spin_min_periods).max()
        - psi.rolling(spin_width, center=True, min_periods=spin_min_periods).min()
    ) / (2 * np.pi)
    sign = np.sign(omega.fillna(0).to_numpy())
    changes = np.r_[0, (sign[1:] * sign[:-1] < 0).astype(int)]
    result["sign_changes"] = pd.Series(changes).rolling(width, center=True, min_periods=min_periods).sum()
    return result


def _run_length_labels(labels: np.ndarray) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    if len(labels) == 0:
        return runs
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            runs.append((start, index, str(labels[start])))
            start = index
    return runs


def classify_states(derived: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    metrics = _rolling_metrics(derived, config)
    valid = derived["valid"].to_numpy(bool)
    labels = np.full(len(derived), "transition", dtype=object)
    labels[~valid] = "invalid"
    spinning = valid & (metrics["net_turns"].to_numpy() >= config.spinning_min_window_turns) & (metrics["sign_consistency"].to_numpy() >= config.spinning_sign_consistency)
    labels[spinning] = "pair_axis_spinning"
    spinning_candidate = (
        valid
        & ~spinning
        & (metrics["spin_range_turns"].to_numpy() >= config.spinning_candidate_min_range_turns)
    )
    labels[spinning_candidate] = "pair_axis_spinning_candidate"
    # Static is a pair-axis state. Do not make it depend on calibration-center
    # radius stability because an offset center can create artificial r motion.
    static = valid & ~spinning & ~spinning_candidate & (metrics["psi_std"].to_numpy() <= math.radians(config.static_angle_std_deg))
    labels[static] = "static"
    libration = valid & ~spinning & ~spinning_candidate & ~static & (metrics["psi_range"].to_numpy() >= math.radians(config.libration_min_range_deg)) & (metrics["sign_changes"].to_numpy() >= 2)
    labels[libration] = "libration"

    time = derived["time_s"].to_numpy(float)
    dt = float(np.nanmedian(np.diff(time)))
    minimum = max(1, int(round(config.minimum_state_duration_s / dt)))
    for start, stop, label in _run_length_labels(labels.copy()):
        if label in {"static", "libration", "pair_axis_spinning"} and stop - start < minimum:
            labels[start:stop] = "transition"

    # A bounded oscillation is required for the experimental libration label.
    # Long multi-turn, direction-changing runs remain useful candidates, but are
    # deliberately not promoted to spinning or chaotic motion.
    psi = derived["psi_unwrapped_rad"].to_numpy(float)
    for start, stop, label in _run_length_labels(labels.copy()):
        if label != "libration":
            continue
        finite = psi[start:stop][np.isfinite(psi[start:stop])]
        if finite.size < 2:
            labels[start:stop] = "transition"
            continue
        net_turns = abs(float(finite[-1] - finite[0])) / (2 * np.pi)
        span_turns = float(np.ptp(finite)) / (2 * np.pi)
        if max(net_turns, span_turns) > config.irregular_min_net_turns:
            labels[start:stop] = "irregular_candidate"

    runs = _run_length_labels(labels.copy())
    max_gap = int(round(config.reversal_max_gap_s / dt))
    spin_runs = [(s, e, np.sign(np.nanmedian(derived["psi_dot"].iloc[s:e]))) for s, e, lab in runs if lab == "pair_axis_spinning"]
    for left, right in zip(spin_runs, spin_runs[1:]):
        if left[2] * right[2] < 0 and right[0] - left[1] <= max_gap:
            labels[left[1] : right[0]] = "reversal"
    output = derived.copy()
    output["auto_state"] = labels
    output["state"] = labels
    output["segment_id"] = ""
    for number, (start, stop, label) in enumerate(_run_length_labels(labels), start=1):
        output.loc[start : stop - 1, "segment_id"] = f"seg_{number:03d}_{label}"
    return output


def dominant_frequency_phase(time: np.ndarray, signal: np.ndarray, companion: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(time) & np.isfinite(signal) & np.isfinite(companion)
    if valid.sum() < 8:
        return math.nan, math.nan, math.nan
    t = time[valid]
    x = signal[valid]
    y = companion[valid]
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    window = np.hanning(len(x))
    fx = np.fft.rfft(x * window)
    fy = np.fft.rfft(y * window)
    frequencies = np.fft.rfftfreq(len(x), d=float(np.median(np.diff(t))))
    if len(frequencies) < 3:
        return math.nan, math.nan, math.nan
    index = 1 + int(np.argmax(np.abs(fx[1:]) ** 2))
    power = np.abs(fx) ** 2
    peak_fraction = float(power[index] / max(power[1:].sum(), 1e-12))
    phase = float(np.angle(fy[index] * np.conj(fx[index])))
    return float(frequencies[index]), peak_fraction, phase


def theoretical_rlc(a: float, b: float) -> float:
    if not (0 < a < b):
        return math.nan
    return (a**5 + b**5) / ((a**3 + b**3) * (b - a))


def summarize_segments(derived: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    records: list[dict] = []
    a = float(derived["small_radius_prototype_px"].iloc[0])
    b = float(derived["large_radius_prototype_px"].iloc[0])
    rlc = theoretical_rlc(a, b)
    for segment_id, segment in derived.groupby("segment_id", sort=False):
        state = str(segment["state"].iloc[0])
        duration = float(segment["time_s"].iloc[-1] - segment["time_s"].iloc[0])
        psi = segment["psi_unwrapped_rad"].to_numpy(float)
        theta = segment["theta_unwrapped_rad"].to_numpy(float)
        psi_turns = float((psi[np.isfinite(psi)][-1] - psi[np.isfinite(psi)][0]) / (2 * np.pi)) if np.isfinite(psi).sum() >= 2 else math.nan
        theta_turns = float((theta[np.isfinite(theta)][-1] - theta[np.isfinite(theta)][0]) / (2 * np.pi)) if np.isfinite(theta).sum() >= 2 else math.nan
        omega_psi = finite_median(segment["psi_dot"])
        omega_theta = finite_median(segment["theta_dot"])
        denom = max(abs(omega_psi), abs(omega_theta), 1e-12)
        omega_relative_difference = abs(omega_psi - omega_theta) / denom
        r_values = segment["r_px"].to_numpy(float)
        if np.isfinite(r_values).sum() >= 3 and duration > 0:
            drift = float(np.polyfit(segment["time_s"], r_values, 1)[0])
        else:
            drift = math.nan
        frequency, peak_fraction, phase = dominant_frequency_phase(
            segment["time_s"].to_numpy(float), segment["psi_unwrapped_rad"].to_numpy(float), segment["r_px"].to_numpy(float)
        )
        invalid_fraction = float((~segment["valid"]).mean())
        theta_reliable_fraction = float(segment["theta_reliable"].mean())
        complete_orbit = abs(psi_turns) >= 1.0
        omega_consistent = (
            np.isfinite(omega_psi)
            and np.isfinite(omega_theta)
            and np.sign(omega_psi) == np.sign(omega_theta)
            and omega_relative_difference <= config.spinning_omega_relative_tolerance
        )
        median_r = finite_median(r_values)
        radial_stable = bool(
            np.isfinite(drift)
            and np.isfinite(median_r)
            and abs(drift) * max(duration, 1e-12)
            <= config.spinning_max_radial_drift_fraction * max(median_r, 1e-12)
        )
        qualified = state == "pair_axis_spinning" and complete_orbit and omega_consistent and radial_stable and invalid_fraction < 0.05
        reasons = []
        if state == "pair_axis_spinning":
            if not complete_orbit: reasons.append("less_than_one_pair_axis_turn")
            if theta_reliable_fraction < 0.95: reasons.append("theta_unreliable_near_trap_center")
            if not omega_consistent: reasons.append("omega_psi_theta_inconsistent_or_missing")
            if not radial_stable: reasons.append("radial_drift")
            if invalid_fraction >= 0.05: reasons.append("invalid_fraction_ge_5pct")
        records.append(
            {
                "segment_id": segment_id,
                "state": state,
                "start_frame": int(segment["frame"].iloc[0]),
                "end_frame": int(segment["frame"].iloc[-1]),
                "start_s": float(segment["time_s"].iloc[0]),
                "end_s": float(segment["time_s"].iloc[-1]),
                "duration_s": duration,
                "frame_count": len(segment),
                "invalid_fraction": invalid_fraction,
                "theta_reliable_fraction": theta_reliable_fraction,
                "psi_net_turns": psi_turns,
                "theta_net_turns": theta_turns,
                "omega_psi_rad_s": omega_psi,
                "omega_theta_rad_s": omega_theta,
                "omega_relative_difference": omega_relative_difference,
                "median_r_px": median_r,
                "median_r_cos_phi_px": finite_median(segment["r_cos_phi_px"]),
                "radial_drift_px_s": drift,
                "libration_frequency_hz": frequency if state == "libration" else math.nan,
                "libration_peak_power_fraction": peak_fraction if state == "libration" else math.nan,
                "libration_r_phase_rad": phase if state == "libration" else math.nan,
                "libration_angle_amplitude_deg": math.degrees((float(np.nanmax(psi)) - float(np.nanmin(psi))) / 2) if state == "libration" and np.isfinite(psi).any() else math.nan,
                "small_radius_px": a,
                "large_radius_px": b,
                "a_over_b": a / b,
                "rlc_theory_px": rlc,
                "spinning_qualified_for_rlc": qualified,
                "spinning_exclusion_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame.from_records(records)


def monte_carlo_spinning(
    derived: pd.DataFrame, segments: pd.DataFrame, config: AnalysisConfig, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    fps = 1.0 / float(np.nanmedian(np.diff(derived["time_s"])))
    block = max(1, int(round(config.bootstrap_block_s * fps)))
    for _, metadata in segments[segments["state"].eq("pair_axis_spinning")].iterrows():
        segment = derived[derived["segment_id"].eq(metadata["segment_id"])]
        n = len(segment)
        if n < 3:
            continue
        a_values = segment["small_radius_px"].to_numpy(float)
        b_values = segment["large_radius_px"].to_numpy(float)
        samples = []
        for _sample in range(config.monte_carlo_samples):
            starts = rng.integers(0, max(1, n - block + 1), size=math.ceil(n / block))
            indices = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
            a = float(np.median(a_values[indices]) + rng.normal(0, config.systematic_radius_uncertainty_px))
            b = float(np.median(b_values[indices]) + rng.normal(0, config.systematic_radius_uncertainty_px))
            if a > b:
                a, b = b, a
            tx = config.trap_x_px + rng.normal(0, config.trap_se_x_px)
            ty = config.trap_y_px + rng.normal(0, config.trap_se_y_px)
            wa, wb = a**3, b**3
            cmx = (wa * segment["small_x_px"].to_numpy() + wb * segment["large_x_px"].to_numpy()) / (wa + wb)
            cmy = (wa * segment["small_y_px"].to_numpy() + wb * segment["large_y_px"].to_numpy()) / (wa + wb)
            rx, ry = cmx - tx, -(cmy - ty)
            r = np.hypot(rx, ry)
            theta = np.arctan2(ry, rx)
            psi = segment["psi_rad"].to_numpy(float)
            q = r * np.cos(wrap_angle(psi - theta))
            samples.append((a, b, theoretical_rlc(a, b), float(np.median(r)), float(np.median(q))))
        array = np.asarray(samples)
        q05, q50, q95 = np.nanquantile(array, [0.05, 0.5, 0.95], axis=0)
        sigma_delta = math.sqrt(2) * config.systematic_radius_uncertainty_px
        observable = min(
            config.trap_x_px - config.roi_x_min,
            config.roi_x_max - config.trap_x_px,
            config.trap_y_px - config.roi_y_min,
            config.roi_y_max - config.trap_y_px,
        )
        records.append(
            {
                "segment_id": metadata["segment_id"],
                "qualified_for_rlc": bool(metadata["spinning_qualified_for_rlc"]),
                "a_px_mc_median": q50[0], "a_px_mc_p05": q05[0], "a_px_mc_p95": q95[0],
                "b_px_mc_median": q50[1], "b_px_mc_p05": q05[1], "b_px_mc_p95": q95[1],
                "rlc_theory_px_mc_median": q50[2], "rlc_theory_px_mc_p05": q05[2], "rlc_theory_px_mc_p95": q95[2],
                "measured_r_px_mc_median": q50[3], "measured_r_px_mc_p05": q05[3], "measured_r_px_mc_p95": q95[3],
                "measured_r_cos_phi_px_mc_median": q50[4], "measured_r_cos_phi_px_mc_p05": q05[4], "measured_r_cos_phi_px_mc_p95": q95[4],
                "ill_conditioned": bool((q50[1] - q50[0]) <= 3 * sigma_delta),
                "outside_observable_range": bool(q50[2] > observable),
                "observable_radius_px": observable,
                "uncertainty_method": f"{config.bootstrap_block_s}s block bootstrap + {config.systematic_radius_uncertainty_px}px radius systematic + trap-center Gaussian MC",
                "formula_version": config.formula_version,
            }
        )
    return pd.DataFrame.from_records(records)


def apply_manual_annotations(derived: pd.DataFrame, annotation_path: Path) -> pd.DataFrame:
    if not annotation_path.exists():
        pd.DataFrame(columns=["video_id", "start_frame", "end_frame", "override_state", "reviewer", "notes"]).to_csv(annotation_path, index=False)
        return derived
    annotations = pd.read_csv(annotation_path)
    if annotations.empty:
        return derived
    output = derived.copy()
    for _, row in annotations.iterrows():
        override = str(row["override_state"])
        if override == "spinning":
            override = "pair_axis_spinning"
        mask = output["frame"].between(int(row["start_frame"]), int(row["end_frame"]))
        if override != "invalid":
            mask &= output["valid"]
        output.loc[mask, "state"] = override
    output["segment_id"] = ""
    for number, (start, stop, label) in enumerate(_run_length_labels(output["state"].to_numpy()), start=1):
        output.loc[start : stop - 1, "segment_id"] = f"seg_{number:03d}_{label}"
    return output


def state_sensitivity(clean: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    variants = {
        "baseline": config,
        "smooth_0.75x": replace(config, smoothing_window_s=0.75 * config.smoothing_window_s),
        "smooth_1.5x": replace(config, smoothing_window_s=1.5 * config.smoothing_window_s),
        "state_window_0.75x": replace(config, classification_window_s=0.75 * config.classification_window_s),
        "state_window_1.5x": replace(config, classification_window_s=1.5 * config.classification_window_s),
        "libration_range_0.75x": replace(config, libration_min_range_deg=0.75 * config.libration_min_range_deg),
        "libration_range_1.5x": replace(config, libration_min_range_deg=1.5 * config.libration_min_range_deg),
    }
    records = []
    for name, variant in variants.items():
        classified = classify_states(derive_geometry(clean, variant), variant)
        fractions = classified["state"].value_counts(normalize=True).to_dict()
        records.append(
            {
                "variant": name,
                "smoothing_window_s": variant.smoothing_window_s,
                "classification_window_s": variant.classification_window_s,
                "libration_min_range_deg": variant.libration_min_range_deg,
                **{f"{state}_fraction": fractions.get(state, 0.0) for state in (
                    "static", "libration", "pair_axis_spinning_candidate", "pair_axis_spinning", "reversal",
                    "irregular_candidate", "transition", "invalid"
                )},
            }
        )
    return pd.DataFrame.from_records(records)


def plot_video_report(derived: pd.DataFrame, segments: pd.DataFrame, output_dir: Path, title: str) -> list[str]:
    state_colors = {"invalid": "#777777", "static": "#4C78A8", "libration": "#F2CF5B", "pair_axis_spinning_candidate": "#F28E2B", "pair_axis_spinning": "#E45756", "reversal": "#B279A2", "transition": "#72B7B2", "irregular_candidate": "#FF9DA6"}
    time = derived["time_s"]
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time, np.degrees(derived["psi_rad"]), lw=0.7, label=r"wrapped $\psi$")
    theta_plot = derived["theta_rad"].where(derived["theta_reliable"])
    axes[0].plot(time, np.degrees(theta_plot), lw=0.7, label=r"wrapped $\theta$ (reliable only)")
    axes[0].set_ylabel("wrapped angle (deg)"); axes[0].legend(loc="upper left", ncol=2)
    axes[1].plot(time, np.degrees(derived["phi_rad"]), lw=0.7); axes[1].set_ylabel(r"$\phi$ (deg)")
    axes[2].plot(time, derived["r_px"], lw=0.7, label="r")
    axes[2].plot(time, derived["r_cos_phi_px"], lw=0.7, label=r"$r\cos\phi$")
    axes[2].set_ylabel("distance (px)"); axes[2].legend(loc="upper left", ncol=2)
    axes[3].plot(time, derived["psi_dot"], lw=0.7, label=r"$\dot\psi$")
    axes[3].plot(time, derived["theta_dot"], lw=0.7, label=r"$\dot\theta$")
    axes[3].set_ylabel("rad/s"); axes[3].set_xlabel("time (s)"); axes[3].legend(loc="upper left", ncol=2)
    for axis in axes:
        for _, segment in segments.iterrows():
            axis.axvspan(segment["start_s"], segment["end_s"], color=state_colors.get(segment["state"], "#cccccc"), alpha=0.10)
        axis.grid(alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    timeseries = output_dir / "timeseries_states.png"
    figure.savefig(timeseries, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 7))
    for state, group in derived.groupby("state"):
        axis.plot(group["r_x_px"], group["r_y_px"], ".", ms=1.5, alpha=0.55, label=state, color=state_colors.get(state))
    axis.scatter([0], [0], marker="+", s=120, color="black", label="trap center")
    axis.set_aspect("equal"); axis.set_xlabel("CM x from trap (px)"); axis.set_ylabel("CM y from trap (px)")
    axis.grid(alpha=0.2); axis.legend(markerscale=4, fontsize=8); axis.set_title(title)
    figure.tight_layout()
    trajectory = output_dir / "cm_trajectory.png"
    figure.savefig(trajectory, dpi=160)
    plt.close(figure)
    return [timeseries.name, trajectory.name]


def write_html_report(output_dir: Path, title: str, metrics: dict, segments: pd.DataFrame, images: list[str]) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(str(row.segment_id))}</td><td>{html.escape(str(row.state))}</td><td>{row.start_s:.3f}</td><td>{row.end_s:.3f}</td><td>{row.duration_s:.3f}</td><td>{html.escape(str(row.spinning_qualified_for_rlc))}</td><td>{html.escape(str(row.spinning_exclusion_reasons))}</td></tr>"
        for row in segments.itertuples()
    )
    metric_list = "".join(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>" for k, v in metrics.items())
    image_html = "".join(f'<h2>{html.escape(name)}</h2><img src="{html.escape(name)}" style="max-width:100%">' for name in images)
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
    <style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:auto;padding:24px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:5px;font-size:12px}}th{{background:#eee}}</style></head>
    <body><h1>{html.escape(title)}</h1><p>Exploratory pixel-scale analysis. Physical pixel calibration remains [pending confirmation].</p>
    <ul>{metric_list}</ul>{image_html}<h2>Segments</h2><table><tr><th>ID</th><th>state</th><th>start s</th><th>end s</th><th>duration s</th><th>Rlc qualified</th><th>exclusion reasons</th></tr>{rows}</table></body></html>"""
    (output_dir / "video_report.html").write_text(document, encoding="utf-8")


def extract_review_frames(
    video_path: Path, derived: pd.DataFrame, output_dir: Path, config: AnalysisConfig
) -> None:
    review_dir = output_dir / "review_frames"
    review_dir.mkdir(exist_ok=True)
    for stale in review_dir.glob("frame_*.jpg"):
        stale.unlink()
    candidates = {0, len(derived) // 2, len(derived) - 1}
    labels = derived["state"].to_numpy()
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if len(boundaries) > max(0, config.max_review_frames - len(candidates)):
        slots = max(0, config.max_review_frames - len(candidates))
        if slots:
            boundaries = boundaries[np.linspace(0, len(boundaries) - 1, slots, dtype=int)]
        else:
            boundaries = np.array([], dtype=int)
    candidates.update(boundaries.tolist())
    candidates = sorted(
        {max(0, min(int(index), len(derived) - 1)) for index in candidates}
    )[: config.max_review_frames]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    thumbnails = []
    for index in candidates:
        frame_number = int(derived["frame"].iloc[index])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, image = cap.read()
        if not ok:
            continue
        row = derived.iloc[index]
        cv2.putText(image, f"frame {frame_number} state={row['state']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        coordinates = [row["small_x_px"], row["small_y_px"], row["large_x_px"], row["large_y_px"]]
        if all(np.isfinite(float(value)) for value in coordinates):
            cv2.arrowedLine(
                image,
                (round(row["small_x_px"]), round(row["small_y_px"])),
                (round(row["large_x_px"]), round(row["large_y_px"])),
                (0, 255, 255),
                2,
            )
        cv2.drawMarker(image, (round(config.trap_x_px), round(config.trap_y_px)), (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.imwrite(str(review_dir / f"frame_{frame_number:07d}_{row['state']}.jpg"), image)
        thumb = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        thumbnails.append(thumb)
    cap.release()
    if thumbnails:
        columns = 3
        blank = np.zeros_like(thumbnails[0])
        rows = []
        for start in range(0, len(thumbnails), columns):
            row = thumbnails[start : start + columns]
            rows.append(np.hstack(row + [blank] * (columns - len(row))))
        cv2.imwrite(str(output_dir / "review_contact_sheet.jpg"), np.vstack(rows))


def analyze_video(
    tracking_path: Path,
    video_path: Path,
    output_root: Path,
    config: AnalysisConfig,
    model_path: Path | None = None,
    seed: int = 20260714,
) -> dict:
    video_id = tracking_path.parent.name.replace("_particle_tracking_app", "")
    output_dir = output_root / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(tracking_path)
    mapping = pd.DataFrame([{"source_column": source, "standard_field": target, "status": "mapped"} for source, target in STANDARD_MAPPING.items()] + [{"source_column": name, "standard_field": "preserved_source_only", "status": "unmapped"} for name in raw.columns if name not in STANDARD_MAPPING])
    mapping.to_csv(output_dir / "raw_mapping.csv", index=False)
    clean, metrics = clean_tracking(raw, config)
    clean.to_csv(output_dir / "clean_tracks.csv", index=False)
    derived = derive_geometry(clean, config)
    if {"cm_x_px", "cm_y_px"}.issubset(raw.columns):
        derived["legacy_instantaneous_cm_x_px"] = pd.to_numeric(raw["cm_x_px"], errors="coerce")
        derived["legacy_instantaneous_cm_y_px"] = pd.to_numeric(raw["cm_y_px"], errors="coerce")
        derived["fixed_vs_legacy_cm_difference_px"] = np.hypot(
            derived["cm_x_px"] - derived["legacy_instantaneous_cm_x_px"],
            derived["cm_y_px_image"] - derived["legacy_instantaneous_cm_y_px"],
        )
    derived = classify_states(derived, config)
    annotations = output_dir / "manual_annotations.csv"
    derived = apply_manual_annotations(derived, annotations)
    segments = summarize_segments(derived, config)
    mc = monte_carlo_spinning(derived, segments, config, seed)
    if not mc.empty:
        segments = segments.merge(mc, on="segment_id", how="left")
    derived.to_csv(output_dir / "derived_tracks.csv", index=False)
    segments.to_csv(output_dir / "segments.csv", index=False)
    sensitivity = state_sensitivity(clean, config)
    sensitivity.to_csv(output_dir / "state_sensitivity.csv", index=False)
    images = plot_video_report(derived, segments, output_dir, video_id)
    extract_review_frames(video_path, derived, output_dir, config)
    state_fraction = derived["state"].value_counts(normalize=True).to_dict()
    metrics.update(
        video_id=video_id,
        row_count=len(raw),
        fps=float((raw["frame"].iloc[-1] - raw["frame"].iloc[0]) / (raw["time_s"].iloc[-1] - raw["time_s"].iloc[0])),
        duration_s=float(raw["time_s"].iloc[-1] - raw["time_s"].iloc[0]),
        a_over_b=metrics["small_radius_prototype_px"] / metrics["large_radius_prototype_px"],
        rlc_theory_px=theoretical_rlc(metrics["small_radius_prototype_px"], metrics["large_radius_prototype_px"]),
        state_fractions=state_fraction,
    )
    if "fixed_vs_legacy_cm_difference_px" in derived:
        difference = derived["fixed_vs_legacy_cm_difference_px"].to_numpy(float)
        metrics.update(
            fixed_vs_legacy_cm_median_difference_px=float(np.nanmedian(difference)),
            fixed_vs_legacy_cm_p95_difference_px=float(np.nanquantile(difference, 0.95)),
            fixed_vs_legacy_cm_max_difference_px=float(np.nanmax(difference)),
        )
    provenance = {
        "video_id": video_id,
        "tracking_csv": str(tracking_path.resolve()),
        "tracking_csv_sha256": sha256(tracking_path),
        "raw_video": str(video_path.resolve()),
        "raw_video_sha256": sha256(video_path),
        "model_path": str(model_path.resolve()) if model_path else None,
        "model_sha256": sha256(model_path) if model_path and model_path.exists() else None,
        "analysis_config": asdict(config),
        "formula_version": config.formula_version,
        "pixel_scale_status": "pending_confirmation; outputs remain in pixels",
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_html_report(output_dir, video_id, metrics, segments, images)
    return {"metrics": metrics, "segments": segments, "derived": derived, "output_dir": output_dir}


def plot_global(results: list[dict], output_root: Path) -> pd.DataFrame:
    records = []
    for result in results:
        metrics = result["metrics"]
        derived = result["derived"]
        segments = result["segments"]
        fractions = derived["state"].value_counts(normalize=True).to_dict()
        qualified = segments[segments["spinning_qualified_for_rlc"].eq(True)]
        libration = segments[segments["state"].eq("libration")]
        records.append(
            {
                "video_id": metrics["video_id"], "row_count": metrics["row_count"], "fps": metrics["fps"],
                "raw_ok_fraction": metrics["raw_ok_fraction"], "clean_valid_fraction": metrics["clean_valid_fraction"],
                "small_radius_px": metrics["small_radius_prototype_px"], "large_radius_px": metrics["large_radius_prototype_px"],
                "a_over_b": metrics["a_over_b"], "rlc_theory_px": metrics["rlc_theory_px"],
                "static_fraction": fractions.get("static", 0), "libration_fraction": fractions.get("libration", 0),
                "pair_axis_spinning_candidate_fraction": fractions.get("pair_axis_spinning_candidate", 0),
                "pair_axis_spinning_fraction": fractions.get("pair_axis_spinning", 0), "reversal_fraction": fractions.get("reversal", 0),
                "irregular_candidate_fraction": fractions.get("irregular_candidate", 0),
                "transition_fraction": fractions.get("transition", 0), "invalid_fraction": fractions.get("invalid", 0),
                "qualified_spinning_segments": len(qualified),
                "median_qualified_r_px": float(qualified["median_r_px"].median()) if len(qualified) else math.nan,
                "median_qualified_r_cos_phi_px": float(qualified["median_r_cos_phi_px"].median()) if len(qualified) else math.nan,
                "median_libration_frequency_hz": float(libration["libration_frequency_hz"].median()) if len(libration) else math.nan,
                "median_libration_amplitude_deg": float(libration["libration_angle_amplitude_deg"].median()) if len(libration) else math.nan,
            }
        )
    summary = pd.DataFrame(records).sort_values("a_over_b")
    summary.to_csv(output_root / "global_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(summary["video_id"], summary["raw_ok_fraction"], label="raw complete")
    axes[0].bar(summary["video_id"], summary["clean_valid_fraction"], label="clean valid", alpha=0.8)
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("fraction of frames"); axes[0].legend()
    axes[1].bar(summary["video_id"], summary["invalid_fraction"], label="invalid")
    axes[1].bar(summary["video_id"], summary["transition_fraction"], bottom=summary["invalid_fraction"], label="transition")
    axes[1].set_ylabel("fraction of frames"); axes[1].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Tracking and classification QC")
    figure.tight_layout(); figure.savefig(output_root / "tracking_qc_panel.png", dpi=180); plt.close(figure)

    states = ["static", "libration", "pair_axis_spinning_candidate", "pair_axis_spinning", "reversal", "irregular_candidate", "transition", "invalid"]
    figure, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(summary))
    colors = ["#4C78A8", "#F2CF5B", "#F28E2B", "#E45756", "#B279A2", "#FF9DA6", "#72B7B2", "#777777"]
    for state, color in zip(states, colors):
        values = summary[f"{state}_fraction"].to_numpy()
        axis.bar(summary["a_over_b"].astype(str), values, bottom=bottom, label=state, color=color)
        bottom += values
    axis.set_xlabel("a/b"); axis.set_ylabel("fraction of frames"); axis.legend(ncol=3); axis.set_title("Exploratory state fractions")
    figure.tight_layout(); figure.savefig(output_root / "state_fraction_vs_size_ratio.png", dpi=180); plt.close(figure)

    all_segments = pd.concat([result["segments"].assign(video_id=result["metrics"]["video_id"]) for result in results], ignore_index=True)
    all_segments.to_csv(output_root / "all_segments.csv", index=False)
    spin = all_segments[all_segments["state"].eq("pair_axis_spinning")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(spin["rlc_theory_px"], spin["median_r_px"], c=spin["spinning_qualified_for_rlc"].map({True: "#E45756", False: "#999999"}))
    axes[0].set_xlabel("theory Rlc (px)"); axes[0].set_ylabel("measured median r (px)")
    axes[1].scatter(spin["rlc_theory_px"], spin["median_r_cos_phi_px"], c=spin["spinning_qualified_for_rlc"].map({True: "#E45756", False: "#999999"}))
    axes[1].set_xlabel(r"theory $\Gamma_I/\Xi$ (px)"); axes[1].set_ylabel(r"median $r\cos\phi$ (px)")
    axes[2].scatter(spin["omega_psi_rad_s"], spin["omega_theta_rad_s"], c=spin["spinning_qualified_for_rlc"].map({True: "#E45756", False: "#999999"}))
    if len(spin):
        limit = max(abs(spin["omega_psi_rad_s"]).max(), abs(spin["omega_theta_rad_s"]).max())
        axes[2].plot([-limit, limit], [-limit, limit], "k--", lw=1)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No spinning segments detected", ha="center", va="center", transform=axis.transAxes)
    axes[2].set_xlabel(r"$\Omega_\psi$ (rad/s)"); axes[2].set_ylabel(r"$\Omega_\theta$ (rad/s)")
    for axis in axes: axis.grid(alpha=0.2)
    figure.tight_layout(); figure.savefig(output_root / "rlc_and_omega_comparisons.png", dpi=180); plt.close(figure)

    lib = all_segments[all_segments["state"].eq("libration")]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(lib["a_over_b"], lib["libration_frequency_hz"]); axes[0].set_xlabel("a/b"); axes[0].set_ylabel("libration frequency (Hz)")
    axes[1].scatter(lib["a_over_b"], lib["libration_angle_amplitude_deg"]); axes[1].set_xlabel("a/b"); axes[1].set_ylabel("libration amplitude (deg)")
    for axis in axes: axis.grid(alpha=0.2)
    figure.tight_layout(); figure.savefig(output_root / "libration_vs_size_ratio.png", dpi=180); plt.close(figure)
    return summary


def analyze_all(project_root: Path, config: AnalysisConfig) -> pd.DataFrame:
    tracking_root = project_root / "outputs" / "particle_tracking"
    raw_root = project_root / "raw_data" / "2026-07-10_spining_pairs"
    output_root = project_root / "outputs" / "dynamic_analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = project_root / "models" / "particle_tracking" / "best.pt"
    results = []
    for tracking_path in sorted(tracking_root.glob("*_particle_tracking_app/tracking.csv")):
        stem = tracking_path.parent.name.replace("_particle_tracking_app", "")
        video_path = raw_root / f"{stem}.avi"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        results.append(analyze_video(tracking_path, video_path, output_root, config, model_path=model_path))
    if len(results) != 5:
        raise RuntimeError(f"Expected five tracking inputs, found {len(results)}")
    summary = plot_global(results, output_root)
    pd.DataFrame([{"source_column": source, "standard_field": target} for source, target in STANDARD_MAPPING.items()]).to_csv(output_root / "raw_mapping.csv", index=False)
    (output_root / "analysis_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args(argv)
    config = AnalysisConfig.from_json(args.config)
    summary = analyze_all(args.project_root.resolve(), config)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
