from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import AnalysisConfig, theoretical_rlc, wrap_angle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def fit_phase_locked_orbit(z: np.ndarray, psi: np.ndarray) -> tuple[complex, complex]:
    design = np.column_stack([np.ones(len(z)), np.exp(1j * psi)])
    center, amplitude = np.linalg.lstsq(design, z, rcond=None)[0]
    return complex(center), complex(amplitude)


def fit_projection_center(
    z: np.ndarray, psi: np.ndarray
) -> tuple[complex, float]:
    x, y = z.real, z.imag
    cosine, sine = np.cos(psi), np.sin(psi)
    response = x * cosine + y * sine
    design = np.column_stack([cosine, sine, np.ones(len(z))])
    center_x, center_y, projection = np.linalg.lstsq(design, response, rcond=None)[0]
    return complex(center_x, center_y), float(projection)


def cross_validate(z: np.ndarray, psi: np.ndarray) -> pd.DataFrame:
    indices = np.arange(len(z))
    first = indices < len(z) // 2
    splits = [
        ("first_to_second", first, ~first),
        ("second_to_first", ~first, first),
        ("full_fit", np.ones(len(z), bool), np.ones(len(z), bool)),
    ]
    records = []
    for name, train, test in splits:
        center, amplitude = fit_phase_locked_orbit(z[train], psi[train])
        prediction = center + amplitude * np.exp(1j * psi[test])
        baseline = np.full(test.sum(), np.mean(z[train]))
        squared_error = np.abs(z[test] - prediction) ** 2
        baseline_error = np.abs(z[test] - baseline) ** 2
        projection_center, fitted_projection = fit_projection_center(z[train], psi[train])
        projection_test = np.real((z[test] - projection_center) * np.exp(-1j * psi[test]))
        records.append(
            {
                "fit": name,
                "phase_center_x_px": center.real,
                "phase_center_y_image_px": -center.imag,
                "phase_orbit_radius_px": abs(amplitude),
                "phase_inclination_deg": -np.degrees(np.angle(amplitude)),
                "test_rmse_px": np.sqrt(np.mean(squared_error)),
                "constant_center_baseline_rmse_px": np.sqrt(np.mean(baseline_error)),
                "test_r2": 1 - squared_error.sum() / baseline_error.sum(),
                "projection_center_x_px": projection_center.real,
                "projection_center_y_image_px": -projection_center.imag,
                "fitted_projection_px": fitted_projection,
                "test_projection_mean_px": np.mean(projection_test),
                "test_projection_std_px": np.std(projection_test),
            }
        )
    return pd.DataFrame.from_records(records)


def build_tracks(segment: pd.DataFrame, center: complex, amplitude: complex) -> pd.DataFrame:
    z = segment["cm_x_px"].to_numpy(float) - 1j * segment["cm_y_px_image"].to_numpy(float)
    psi = segment["psi_rad"].to_numpy(float)
    relative = z - center
    theta = np.angle(relative)
    theta_unwrapped = np.unwrap(theta)
    psi_unwrapped = np.unwrap(psi)
    phi = wrap_angle(psi - theta)
    prediction = center + amplitude * np.exp(1j * psi)
    output = pd.DataFrame(
        {
            "frame": segment["frame"].to_numpy(int),
            "time_s": segment["time_s"].to_numpy(float),
            "cm_x_px": z.real,
            "cm_y_image_px": -z.imag,
            "psi_rad": psi,
            "psi_unwrapped_rad": psi_unwrapped,
            "phen_center_x_px": center.real,
            "phen_center_y_image_px": -center.imag,
            "phen_r_x_px": relative.real,
            "phen_r_y_cart_px": relative.imag,
            "phen_r_px": np.abs(relative),
            "phen_theta_rad": theta,
            "phen_theta_unwrapped_rad": theta_unwrapped,
            "phen_phi_rad": phi,
            "phen_r_cos_phi_px": np.abs(relative) * np.cos(phi),
            "phase_locked_prediction_x_px": prediction.real,
            "phase_locked_prediction_y_image_px": -prediction.imag,
            "phase_locked_residual_px": np.abs(z - prediction),
        }
    )
    direction = -1 if psi_unwrapped[-1] < psi_unwrapped[0] else 1
    progress = direction * (psi_unwrapped - psi_unwrapped[0]) / (2 * np.pi)
    output["cycle_index"] = np.floor(np.maximum(progress, 0)).astype(int)
    output["validation_half"] = np.where(np.arange(len(output)) < len(output) // 2, "first", "second")
    return output


def cycle_summary(tracks: pd.DataFrame) -> pd.DataFrame:
    records = []
    for cycle, group in tracks.groupby("cycle_index"):
        if len(group) < 5:
            continue
        phi = group["phen_phi_rad"].to_numpy(float)
        records.append(
            {
                "cycle_index": int(cycle),
                "start_s": group["time_s"].iloc[0],
                "end_s": group["time_s"].iloc[-1],
                "frame_count": len(group),
                "validation_half": group["validation_half"].mode().iloc[0],
                "mean_r_px": group["phen_r_px"].mean(),
                "std_r_px": group["phen_r_px"].std(),
                "mean_r_cos_phi_px": group["phen_r_cos_phi_px"].mean(),
                "std_r_cos_phi_px": group["phen_r_cos_phi_px"].std(),
                "phi_resultant_length": abs(np.mean(np.exp(1j * phi))),
                "mean_residual_px": group["phase_locked_residual_px"].mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def make_plots(
    tracks: pd.DataFrame,
    cross_validation: pd.DataFrame,
    cycles: pd.DataFrame,
    global_center: complex,
    phase_center: complex,
    amplitude: complex,
    projection_center: complex,
    output_dir: Path,
) -> list[str]:
    z = tracks["cm_x_px"].to_numpy() - 1j * tracks["cm_y_image_px"].to_numpy()
    figure, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(z.real, -z.imag, c=tracks["time_s"], s=8, cmap="viridis", label="measured CM")
    circle_angle = np.linspace(0, 2 * np.pi, 361)
    circle = phase_center + abs(amplitude) * np.exp(1j * circle_angle)
    axis.plot(circle.real, -circle.imag, "r-", lw=2, label="phase-locked fitted circle")
    axis.scatter([global_center.real], [-global_center.imag], marker="+", s=160, c="black", label="calibration center")
    axis.scatter([phase_center.real], [-phase_center.imag], marker="x", s=110, c="red", label="phase-fit center")
    axis.scatter([projection_center.real], [-projection_center.imag], marker="D", s=55, c="orange", label="projection-fit center")
    axis.set_aspect("equal"); axis.set_xlabel("image x (px)"); axis.set_ylabel("image y (px)")
    axis.invert_yaxis(); axis.grid(alpha=0.2); axis.legend(fontsize=8)
    figure.colorbar(scatter, ax=axis, label="time (s)")
    axis.set_title("0006 phenomenological orbit center")
    figure.tight_layout(); path = output_dir / "phenomenological_center_geometry.png"; figure.savefig(path, dpi=180); plt.close(figure)

    time = tracks["time_s"]
    psi = np.degrees(tracks["psi_unwrapped_rad"] - tracks["psi_unwrapped_rad"].iloc[0])
    theta = np.degrees(tracks["phen_theta_unwrapped_rad"] - tracks["phen_theta_unwrapped_rad"].iloc[0])
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time, psi, label=r"$\psi$", lw=1); axes[0].plot(time, theta, label=r"$\theta_{phen}$", lw=1)
    axes[0].set_ylabel("unwrapped angle (deg)"); axes[0].legend()
    axes[1].plot(time, np.degrees(tracks["phen_phi_rad"]), lw=0.9); axes[1].set_ylabel(r"$\phi_{phen}$ (deg)")
    axes[2].plot(time, tracks["phen_r_px"], label=r"$r_{phen}$"); axes[2].plot(time, tracks["phen_r_cos_phi_px"], label=r"$r\cos\phi$")
    axes[2].set_ylabel("distance (px)"); axes[2].legend()
    axes[3].plot(time, tracks["phase_locked_residual_px"], lw=0.9); axes[3].set_ylabel("fit residual (px)"); axes[3].set_xlabel("time (s)")
    for axis in axes: axis.grid(alpha=0.2)
    figure.suptitle("0006 geometry relative to phenomenological center")
    figure.tight_layout(); path = output_dir / "phenomenological_center_timeseries.png"; figure.savefig(path, dpi=180); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    holdout = cross_validation[cross_validation["fit"] != "full_fit"]
    axes[0].bar(holdout["fit"], holdout["test_r2"], color=["#4C78A8", "#F58518"])
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("held-out R-squared"); axes[0].grid(axis="y", alpha=0.2)
    axes[1].errorbar(cycles["cycle_index"], cycles["mean_r_cos_phi_px"], yerr=cycles["std_r_cos_phi_px"], fmt="o", capsize=2)
    axes[1].axhline(cross_validation.loc[cross_validation["fit"] == "full_fit", "fitted_projection_px"].iloc[0], color="red", ls="--", label="full-fit projection")
    axes[1].set_xlabel("pair-axis cycle"); axes[1].set_ylabel(r"cycle $r\cos\phi$ (px)"); axes[1].grid(alpha=0.2); axes[1].legend()
    figure.tight_layout(); path = output_dir / "phenomenological_center_validation.png"; figure.savefig(path, dpi=180); plt.close(figure)
    return [
        "phenomenological_center_geometry.png",
        "phenomenological_center_timeseries.png",
        "phenomenological_center_validation.png",
    ]


def generate_report(project_root: Path) -> Path:
    video_id = "18_36_59MJPG-0006"
    video_dir = project_root / "outputs" / "dynamic_analysis" / video_id
    output_dir = video_dir / "phenomenological_center_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AnalysisConfig.from_json(project_root / "analysis" / "dynamic_pair_analysis" / "config.json")
    derived_path = video_dir / "derived_tracks.csv"
    provenance_path = video_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    upstream_tracking_path = Path(provenance["tracking_csv"])
    expected_tracking_hash = provenance["tracking_csv_sha256"]
    current_tracking_hash = sha256(upstream_tracking_path) if upstream_tracking_path.exists() else None
    upstream_tracking_status = (
        "matches recorded analysis input"
        if current_tracking_hash == expected_tracking_hash
        else "current upstream file differs from the recorded analysis input; report uses the preserved derived table"
    )
    source = pd.read_csv(derived_path)
    priority = pd.read_csv(video_dir / "priority_spinning_summary.csv").iloc[0]
    segment_id = str(priority["main_continuous_segment_id"])
    segment = source[source["segment_id"].eq(segment_id)].copy()
    if len(segment) < 20:
        raise RuntimeError(f"Segment {segment_id} is missing or too short")
    z = segment["cm_x_px"].to_numpy(float) - 1j * segment["cm_y_px_image"].to_numpy(float)
    psi = segment["psi_rad"].to_numpy(float)
    phase_center, amplitude = fit_phase_locked_orbit(z, psi)
    projection_center, fitted_projection = fit_projection_center(z, psi)
    validation = cross_validate(z, psi)
    tracks = build_tracks(segment, phase_center, amplitude)
    cycles = cycle_summary(tracks)
    tracks.to_csv(output_dir / "phenomenological_tracks.csv", index=False)
    validation.to_csv(output_dir / "cross_validation.csv", index=False)
    cycles.to_csv(output_dir / "cycle_validation.csv", index=False)

    global_center = complex(config.trap_x_px, -config.trap_y_px)
    a = float(segment["small_radius_prototype_px"].iloc[0])
    b = float(segment["large_radius_prototype_px"].iloc[0])
    rlc = theoretical_rlc(a, b)
    summary = {
        "video_id": video_id,
        "segment_id": segment_id,
        "start_s": float(segment["time_s"].iloc[0]),
        "end_s": float(segment["time_s"].iloc[-1]),
        "phase_center_x_px": phase_center.real,
        "phase_center_y_image_px": -phase_center.imag,
        "phase_center_offset_from_calibration_px": abs(phase_center - global_center),
        "projection_center_x_px": projection_center.real,
        "projection_center_y_image_px": -projection_center.imag,
        "phase_orbit_radius_px": abs(amplitude),
        "phase_inclination_deg": -np.degrees(np.angle(amplitude)),
        "fitted_projection_px": fitted_projection,
        "median_phen_r_px": float(tracks["phen_r_px"].median()),
        "median_phen_r_cos_phi_px": float(tracks["phen_r_cos_phi_px"].median()),
        "median_fit_residual_px": float(tracks["phase_locked_residual_px"].median()),
        "theory_rlc_px": rlc,
        "pixel_scale_status": "pending confirmation; no final mm values",
        "interpretation": "exploratory phenomenological orbit center; not an independently calibrated trapping center",
        "source_derived_tracks": str(derived_path.resolve()),
        "source_derived_tracks_sha256": sha256(derived_path),
        "recorded_upstream_tracking_csv": str(upstream_tracking_path),
        "recorded_upstream_tracking_sha256": expected_tracking_hash,
        "current_upstream_tracking_sha256": current_tracking_hash,
        "upstream_tracking_status": upstream_tracking_status,
    }
    (output_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(output_dir / "fit_summary.csv", index=False)
    images = make_plots(tracks, validation, cycles, global_center, phase_center, amplitude, projection_center, output_dir)

    validation_html = validation.to_html(index=False, float_format=lambda value: f"{value:.4g}")
    cycle_html = cycles.to_html(index=False, float_format=lambda value: f"{value:.4g}")
    metrics_html = "".join(f"<li><b>{html.escape(str(key))}</b>: {html.escape(str(value))}</li>" for key, value in summary.items())
    images_html = "".join(f'<h2>{name}</h2><img src="{name}" style="max-width:100%">' for name in images)
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>0006 phenomenological center report</title>
    <style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:auto;padding:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%;overflow:auto;display:block}}td,th{{border:1px solid #ccc;padding:5px;font-size:12px}}th{{background:#eee}}.warning{{background:#fff4d6;border-left:5px solid #e39d16;padding:12px}}</style></head>
    <body><h1>0006 phenomenological center report</h1>
    <p class='warning'><b>Exploratory fit:</b> the fitted point is a phenomenological orbit center, not an independently calibrated trapping center. The fit relationship is not reused as independent evidence for the theoretical Rlc.</p>
    <p class='warning'><b>Input provenance:</b> {html.escape(upstream_tracking_status)}.</p>
    <p>The phase-locked model is z_CM(t) = c + A exp(i psi(t)) + residual. A separate projection fit checks whether (r_CM-c) dot n is approximately constant. First-half/second-half cross-validation tests transfer without refitting.</p>
    <ul>{metrics_html}</ul>{images_html}<h2>Cross-validation</h2>{validation_html}<h2>Cycle validation</h2>{cycle_html}</body></html>"""
    report_path = output_dir / "video_report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = generate_report(args.project_root.resolve())
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
