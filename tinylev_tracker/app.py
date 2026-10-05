from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.widgets import SpanSelector
from PyQt5 import QtCore, QtGui, QtWidgets

from particle_tracking_app.app import MainWindow as BaselineMainWindow
from particle_tracking_app.app import PlotPanel as BaselinePlotPanel
from particle_tracking_app.core import (
    DEFAULT_LAYERS,
    RAW_DATA_ROOT,
    TrackingConfig,
    read_config,
    read_tracking_csv,
)

from .manual_annotations import (
    DIRECTION_CONVENTION,
    STATE_COLORS,
    STATE_SPECS,
    ManualAnnotationStore,
    utc_now,
    validate_segments,
)


APP_NAME = "Tinylev Tracker"
APP_VERSION = "v0.5"
APP_DISPLAY_NAME = f"{APP_NAME} {APP_VERSION}"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tinylev_tracker"
DEFAULT_ANNOTATION_ROOT = PROJECT_ROOT / "outputs" / "manual_state_annotations"
DEFAULT_MIN_RADIUS_PX = 45.0
DEFAULT_MAX_RADIUS_PX = 220.0
OMEGA_SMOOTHING_WINDOW_S = 0.25
FAST_SPECTRUM_MIN_HZ = 2.0
FAST_SPECTRUM_MAX_HZ = 15.0
SPECTRUM_DISPLAY_FLOOR = 1e-7
PLOT_MODES = (
    "d(t)",
    "psi(t)",
    "phi(t)",
    "Omega(t)",
    "d spectrum",
    "psi + Omega spectrum",
    "phi spectrum",
    "CM trap distance",
    "Confidence",
)


def scaled_limits(
    limits: tuple[float, float], center: float, scale: float, logarithmic: bool = False
) -> tuple[float, float]:
    """Scale an axis interval around the cursor position."""
    lower, upper = limits
    if logarithmic:
        if lower <= 0 or upper <= 0 or center <= 0:
            return lower, upper
        log_lower, log_upper, log_center = np.log10((lower, upper, center))
        return (
            float(10 ** (log_center + (log_lower - log_center) * scale)),
            float(10 ** (log_center + (log_upper - log_center) * scale)),
        )
    return center + (lower - center) * scale, center + (upper - center) * scale


def spectrum_display_curve(
    frequency_hz: np.ndarray,
    power: np.ndarray,
    low_hz: float | None = None,
    high_hz: float | None = None,
    floor: float = SPECTRUM_DISPLAY_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a peak-normalized, positive spectrum suitable for a log display."""
    mask = np.isfinite(frequency_hz) & np.isfinite(power) & (frequency_hz > 0)
    if low_hz is not None:
        mask &= frequency_hz >= low_hz
    if high_hz is not None:
        mask &= frequency_hz <= high_hz
    selected_frequency = frequency_hz[mask]
    selected_power = power[mask]
    if not len(selected_power):
        return np.array([]), np.array([])
    normalized = selected_power / max(float(np.max(selected_power)), 1e-15)
    return selected_frequency, np.maximum(normalized, floor)


def dominant_peak_frequency(
    frequency_hz: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float
) -> float:
    """Select the strongest local spectral peak in a frequency band."""
    if len(power) < 3:
        return math.nan
    candidates = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    candidates = candidates[
        (frequency_hz[candidates] >= low_hz) & (frequency_hz[candidates] <= high_hz)
    ]
    if not len(candidates):
        return math.nan
    return float(frequency_hz[candidates[int(np.argmax(power[candidates]))]])


def _number(row: dict, *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return math.nan


def _continuous_runs(mask: np.ndarray, frame: np.ndarray, time_s: np.ndarray) -> list[tuple[int, int]]:
    """Return valid, time-monotonic runs without bridging missing frames."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, good in enumerate(mask):
        if not good:
            if start is not None:
                runs.append((start, index))
                start = None
            continue
        continuous = index > 0 and mask[index - 1] and time_s[index] > time_s[index - 1]
        if continuous and np.isfinite(frame[index - 1]) and np.isfinite(frame[index]):
            continuous = bool(round(frame[index] - frame[index - 1]) == 1)
        if start is None:
            start = index
        elif not continuous:
            runs.append((start, index))
            start = index
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _unwrap_with_gaps(
    wrapped: np.ndarray, valid: np.ndarray, frame: np.ndarray, time_s: np.ndarray
) -> np.ndarray:
    output = np.full(wrapped.shape, np.nan, dtype=float)
    mask = valid & np.isfinite(wrapped) & np.isfinite(time_s)
    for start, stop in _continuous_runs(mask, frame, time_s):
        output[start:stop] = np.unwrap(wrapped[start:stop])
    return output


def _local_smooth_derivative(
    values: np.ndarray,
    time_s: np.ndarray,
    valid: np.ndarray,
    frame: np.ndarray,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear smoothing and derivative, evaluated within valid runs only."""
    smooth = np.full(values.shape, np.nan, dtype=float)
    derivative = np.full(values.shape, np.nan, dtype=float)
    mask = valid & np.isfinite(values) & np.isfinite(time_s)
    for start, stop in _continuous_runs(mask, frame, time_s):
        segment_time = time_s[start:stop]
        if len(segment_time) < 5:
            continue
        positive_dt = np.diff(segment_time)
        positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
        if not len(positive_dt):
            continue
        width = max(5, int(round(window_s / float(np.median(positive_dt)))))
        if width % 2 == 0:
            width += 1
        if stop - start < width:
            continue
        half = width // 2
        segment = values[start:stop]
        for local_index in range(half, len(segment) - half):
            lo = local_index - half
            hi = local_index + half + 1
            x = segment_time[lo:hi]
            y = segment[lo:hi]
            x_centered = x - x.mean()
            denominator = float(np.dot(x_centered, x_centered))
            if denominator <= 0:
                continue
            smooth[start + local_index] = float(y.mean())
            derivative[start + local_index] = float(np.dot(x_centered, y - y.mean()) / denominator)
    return smooth, derivative


def derive_plot_kinematics(rows: list[dict], omega_window_s: float = OMEGA_SMOOTHING_WINDOW_S) -> dict[str, np.ndarray]:
    """Reconstruct the professor-note kinematics without changing tracking rows."""
    ordered = sorted(rows or [], key=lambda row: (_number(row, "frame"), _number(row, "time_s")))
    count = len(ordered)
    frame = np.asarray([_number(row, "frame") for row in ordered], dtype=float)
    time_s = np.asarray([_number(row, "time_s") for row in ordered], dtype=float)
    status_ok = np.asarray([row.get("status") == "ok" for row in ordered], dtype=bool)

    small_x = np.asarray([_number(row, "small_cx_px", "particle_1_cx_px") for row in ordered])
    small_y = np.asarray([_number(row, "small_cy_px", "particle_1_cy_px") for row in ordered])
    large_x = np.asarray([_number(row, "large_cx_px", "particle_2_cx_px") for row in ordered])
    large_y = np.asarray([_number(row, "large_cy_px", "particle_2_cy_px") for row in ordered])
    pair_valid = status_ok & np.isfinite(time_s) & np.isfinite(small_x + small_y + large_x + large_y)

    pair_dx = large_x - small_x
    pair_dy_cart = -(large_y - small_y)
    separation_px = np.where(pair_valid, np.hypot(pair_dx, pair_dy_cart), np.nan)
    psi_rad = np.where(pair_valid, np.arctan2(pair_dy_cart, pair_dx), np.nan)
    psi_unwrapped_rad = _unwrap_with_gaps(psi_rad, pair_valid, frame, time_s)

    cm_dx = np.asarray(
        [_number(row, "cm_dx_from_trap_px", "centroid_dx_from_trap_px") for row in ordered]
    )
    cm_dy_image = np.asarray(
        [_number(row, "cm_dy_from_trap_px", "centroid_dy_from_trap_px") for row in ordered]
    )
    cm_radius_px = np.hypot(cm_dx, cm_dy_image)
    theta_valid = pair_valid & np.isfinite(cm_dx + cm_dy_image) & (cm_radius_px > 0)
    theta_rad = np.where(theta_valid, np.arctan2(-cm_dy_image, cm_dx), np.nan)
    theta_unwrapped_rad = _unwrap_with_gaps(theta_rad, theta_valid, frame, time_s)
    phi_valid = pair_valid & theta_valid
    phi_rad = np.where(phi_valid, (psi_rad - theta_rad + np.pi) % (2 * np.pi) - np.pi, np.nan)
    phi_unwrapped_rad = _unwrap_with_gaps(phi_rad, phi_valid, frame, time_s)

    psi_smooth_rad, omega_rad_s = _local_smooth_derivative(
        psi_unwrapped_rad, time_s, pair_valid, frame, omega_window_s
    )
    return {
        "rows": np.asarray(ordered, dtype=object),
        "frame": frame,
        "time_s": time_s,
        "pair_valid": pair_valid,
        "phi_valid": phi_valid,
        "d_px": separation_px,
        "psi_rad": psi_rad,
        "psi_unwrapped_rad": psi_unwrapped_rad,
        "theta_rad": theta_rad,
        "theta_unwrapped_rad": theta_unwrapped_rad,
        "phi_rad": phi_rad,
        "phi_unwrapped_rad": phi_unwrapped_rad,
        "psi_smooth_rad": psi_smooth_rad,
        "omega_rad_s": omega_rad_s,
        "cm_radius_px": np.where(np.isfinite(cm_radius_px), cm_radius_px, np.nan),
        "empty": np.empty(count, dtype=float),
    }


def normalized_power_spectrum(
    time_s: np.ndarray, values: np.ndarray, valid: np.ndarray, frame: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Hann-windowed, linearly detrended spectrum of the longest valid run."""
    mask = valid & np.isfinite(time_s) & np.isfinite(values)
    runs = _continuous_runs(mask, frame, time_s)
    if not runs:
        return np.array([]), np.array([]), {}
    start, stop = max(runs, key=lambda bounds: bounds[1] - bounds[0])
    if stop - start < 8:
        return np.array([]), np.array([]), {}
    segment_time = time_s[start:stop]
    segment = values[start:stop]
    dt_values = np.diff(segment_time)
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
    if not len(dt_values):
        return np.array([]), np.array([]), {}
    dt = float(np.median(dt_values))
    centered_time = segment_time - segment_time.mean()
    design = np.column_stack((centered_time, np.ones_like(centered_time)))
    trend = design @ np.linalg.lstsq(design, segment, rcond=None)[0]
    detrended = segment - trend
    transform = np.fft.rfft(detrended * np.hanning(len(detrended)))
    frequency_hz = np.fft.rfftfreq(len(detrended), dt)
    power = np.abs(transform) ** 2
    if len(power):
        power[0] = 0.0
    total = float(power.sum())
    if total > 0:
        power /= total
    return frequency_hz, power, {
        "samples": float(stop - start),
        "duration_s": float(segment_time[-1] - segment_time[0]),
        "resolution_hz": float(1.0 / max(segment_time[-1] - segment_time[0], 1e-15)),
    }


class KinematicsPlotPanel(BaselinePlotPanel):
    """Tracking-QC plots requested from the professor's trajectory note."""

    range_selected = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.navigation_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout().insertWidget(1, self.navigation_toolbar)
        controls = self.layout().itemAt(0).layout()
        controls.addWidget(QtWidgets.QLabel("Wheel zoom | toolbar: pan / box zoom / reset"))
        self.annotation_range_mode = QtWidgets.QCheckBox("Drag to select annotation range")
        self.annotation_range_mode.setToolTip(
            "When enabled on a time-series plot, drag horizontally to fill the manual "
            "annotation start/end frames."
        )
        self.annotation_range_mode.toggled.connect(self._rebuild_span_selector)
        controls.addWidget(self.annotation_range_mode)
        self._scroll_connection_id = self.canvas.mpl_connect("scroll_event", self._zoom_on_scroll)
        self.kinematics = derive_plot_kinematics([])
        self.spectra: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
        self.annotation_segments: list[dict] = []
        self._span_selector: SpanSelector | None = None
        self._showing_spectrum = False
        self.series.blockSignals(True)
        self.series.clear()
        self.series.addItems(list(PLOT_MODES))
        self.series.blockSignals(False)

    def _zoom_on_scroll(self, event) -> None:
        """Zoom both axes around the mouse cursor without changing plotted data."""
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        if event.button == "up":
            scale = 0.8
        elif event.button == "down":
            scale = 1.25
        else:
            return
        self.axes.set_xlim(
            scaled_limits(
                self.axes.get_xlim(), event.xdata, scale, self.axes.get_xscale() == "log"
            )
        )
        self.axes.set_ylim(
            scaled_limits(
                self.axes.get_ylim(), event.ydata, scale, self.axes.get_yscale() == "log"
            )
        )
        self.navigation_toolbar.push_current()
        self.canvas.draw_idle()

    def set_annotation_segments(self, segments: list[dict]) -> None:
        self.annotation_segments = [dict(item) for item in segments]
        self.redraw()

    def _rebuild_span_selector(self) -> None:
        self._span_selector = None
        if (
            self.axes is None
            or self._showing_spectrum
            or not self.annotation_range_mode.isChecked()
            or not len(self.kinematics.get("time_s", []))
        ):
            return
        self._span_selector = SpanSelector(
            self.axes,
            self._on_annotation_span,
            "horizontal",
            useblit=True,
            props={"alpha": 0.18, "facecolor": "#F97316"},
            button=1,
        )

    def _on_annotation_span(self, minimum: float, maximum: float) -> None:
        time_s = np.asarray(self.kinematics.get("time_s", []), dtype=float)
        frame = np.asarray(self.kinematics.get("frame", []), dtype=float)
        if not len(time_s):
            return
        lower, upper = sorted((float(minimum), float(maximum)))
        selected = (
            np.isfinite(time_s)
            & np.isfinite(frame)
            & (time_s >= lower)
            & (time_s <= upper)
        )
        if not np.any(selected):
            finite = np.flatnonzero(np.isfinite(time_s) & np.isfinite(frame))
            if not len(finite):
                return
            start_index = finite[int(np.argmin(np.abs(time_s[finite] - lower)))]
            end_index = finite[int(np.argmin(np.abs(time_s[finite] - upper)))]
            selected[[start_index, end_index]] = True
        selected_frames = frame[selected]
        self.range_selected.emit(
            int(round(float(np.min(selected_frames)))),
            int(round(float(np.max(selected_frames)))),
        )

    def _draw_annotation_spans(self, axis) -> None:
        if self._showing_spectrum:
            return
        for segment in self.annotation_segments:
            start_time = _number(segment, "start_time_s")
            end_time = _number(segment, "end_time_s")
            if not np.isfinite(start_time) or not np.isfinite(end_time):
                continue
            state = str(segment.get("manual_state") or "")
            axis.axvspan(
                start_time,
                end_time,
                color=STATE_COLORS.get(state, "#64748B"),
                alpha=0.14,
                lw=0,
                zorder=0,
            )

    def set_rows(self, rows: list[dict]) -> None:
        self.kinematics = derive_plot_kinematics(rows)
        self.rows = list(self.kinematics["rows"])
        time_s = self.kinematics["time_s"]
        frame = self.kinematics["frame"]
        angular_valid = self.kinematics["pair_valid"] & np.isfinite(
            self.kinematics["omega_rad_s"]
        )
        self.spectra = {
            "d": normalized_power_spectrum(
                time_s, self.kinematics["d_px"], self.kinematics["pair_valid"], frame
            ),
            "psi": normalized_power_spectrum(
                time_s, self.kinematics["psi_unwrapped_rad"], angular_valid, frame
            ),
            "phi": normalized_power_spectrum(
                time_s, self.kinematics["phi_unwrapped_rad"], self.kinematics["phi_valid"], frame
            ),
            "Omega": normalized_power_spectrum(
                time_s, self.kinematics["omega_rad_s"], angular_valid, frame
            ),
        }
        self.redraw()

    def redraw(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        self.axes = axis
        legend_location = "best"
        self.ok_rows = [row for row in self.rows if row.get("status") == "ok"]
        if not self.rows:
            axis.set_title("No tracking data")
            self.playback_cursor.hide()
            self.canvas.draw()
            self.navigation_toolbar.update()
            self.navigation_toolbar.push_current()
            return

        mode = self.series.currentText()
        self._showing_spectrum = mode.endswith(" spectrum")
        time_s = self.kinematics["time_s"]
        if mode == "d(t)":
            axis.plot(time_s, self.kinematics["d_px"], lw=0.8, label=r"$d=|r_b-r_a|$")
            axis.set_ylabel("d (px)")
        elif mode == "psi(t)":
            axis.plot(time_s, self.kinematics["psi_unwrapped_rad"], lw=0.8, label=r"unwrapped $\psi$")
            axis.set_ylabel(r"$\psi$ (rad)")
        elif mode == "phi(t)":
            axis.plot(time_s, self.kinematics["phi_rad"], lw=0.8, label=r"$\phi=\psi-\theta$")
            axis.set_ylabel(r"$\phi$ (rad)")
        elif mode == "Omega(t)":
            axis.plot(time_s, self.kinematics["omega_rad_s"], lw=0.8, label=r"$\Omega=d\psi/dt$")
            axis.set_ylabel(r"$\Omega$ (rad/s)")
            axis.set_title(f"Local-linear derivative; {OMEGA_SMOOTHING_WINDOW_S:.2f} s window")
        elif mode == "psi + Omega spectrum":
            plotted = False
            for spectrum_key, label, color in (
                ("psi", r"$\psi$", "#4C78A8"),
                ("Omega", r"$\Omega$", "#E45756"),
            ):
                frequency_hz, power, _ = self.spectra.get(
                    spectrum_key, (np.array([]), np.array([]), {})
                )
                display_frequency, display_power = spectrum_display_curve(
                    frequency_hz,
                    power,
                    FAST_SPECTRUM_MIN_HZ,
                    FAST_SPECTRUM_MAX_HZ,
                )
                if len(display_frequency):
                    axis.semilogy(
                        display_frequency, display_power, lw=0.85, color=color, label=label
                    )
                    plotted = True

            psi_frequency, psi_power, metadata = self.spectra.get(
                "psi", (np.array([]), np.array([]), {})
            )
            fast_hz = dominant_peak_frequency(
                psi_frequency, psi_power, FAST_SPECTRUM_MIN_HZ, FAST_SPECTRUM_MAX_HZ
            )
            if np.isfinite(fast_hz):
                axis.axvline(fast_hz, color="black", ls="--", lw=0.9)
                axis.text(
                    0.98,
                    0.90,
                    f"{fast_hz:.3f} Hz",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                )
            if not plotted:
                axis.text(
                    0.5,
                    0.5,
                    "Need spectral data in the 2-15 Hz band",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            sample_note = (
                f"; {int(metadata['samples'])} samples; df={metadata['resolution_hz']:.3g} Hz"
                if metadata
                else ""
            )
            axis.set_title(f"Fast angular spectrum (2-15 Hz){sample_note}")
            axis.set_xlabel("frequency (Hz)")
            axis.set_ylabel("band-normalized power")
            axis.set_xlim(FAST_SPECTRUM_MIN_HZ, FAST_SPECTRUM_MAX_HZ)
            axis.set_ylim(SPECTRUM_DISPLAY_FLOOR, 2.0)
            legend_location = "lower right"
        elif self._showing_spectrum:
            spectrum_key = mode.split()[0]
            frequency_hz, power, metadata = self.spectra.get(
                spectrum_key, (np.array([]), np.array([]), {})
            )
            display_frequency, display_power = spectrum_display_curve(frequency_hz, power)
            if len(display_frequency):
                axis.semilogy(display_frequency, display_power, lw=0.85)
                axis.set_title(
                    f"Longest valid run: {int(metadata['samples'])} samples; "
                    f"df={metadata['resolution_hz']:.3g} Hz"
                )
            else:
                axis.text(0.5, 0.5, "Need at least 8 consecutive valid samples", ha="center", va="center", transform=axis.transAxes)
            axis.set_xlabel("frequency (Hz)")
            axis.set_ylabel("peak-normalized power")
            axis.set_xlim(left=0)
            axis.set_ylim(SPECTRUM_DISPLAY_FLOOR, 2.0)
        elif mode == "CM trap distance":
            axis.plot(time_s, self.kinematics["cm_radius_px"], lw=0.8, label="CM-trap distance")
            axis.set_ylabel("px")
        else:
            rows = list(self.kinematics["rows"])
            for index in (1, 2):
                confidence = np.asarray([_number(row, f"particle_{index}_conf") for row in rows])
                axis.plot(time_s, confidence, lw=0.8, label=f"p{index}")
            axis.set_ylabel("confidence")

        if not self._showing_spectrum:
            self._draw_annotation_spans(axis)
            axis.set_xlabel("time (s)")
        axis.grid(True, alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc=legend_location)
        self.canvas.draw()
        self.navigation_toolbar.update()
        self.navigation_toolbar.push_current()
        self._update_playback_cursor()
        self._rebuild_span_selector()

    def _update_playback_cursor(self) -> None:
        if self._showing_spectrum:
            self.playback_cursor.hide()
            return
        super()._update_playback_cursor()


def tracker_defaults(config: TrackingConfig, width: int = 1280, height: int = 1024) -> TrackingConfig:
    """Return safe tracking-only defaults without changing physical calibration."""
    return replace(
        config,
        output_dir=str(DEFAULT_OUTPUT_ROOT / f"{Path(config.video_path).stem}_tinylev_tracker"),
        fast_mode=True,
        save_annotated_video=False,
        save_debug_frames=False,
        min_radius=DEFAULT_MIN_RADIUS_PX,
        max_radius=DEFAULT_MAX_RADIUS_PX,
        roi_x_min=0,
        roi_x_max=max(1, int(width)),
        roi_y_min=0,
        roi_y_max=max(1, int(height)),
    )


def unused_output_dir(requested: str | Path) -> Path:
    """Choose a new derived-output directory rather than overwrite an old run."""
    requested = Path(requested)
    if not requested.exists() or not any(requested.iterdir()):
        return requested
    for index in range(2, 10000):
        candidate = requested.with_name(f"{requested.name}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a new output directory beside: {requested}")


class MainWindow(BaselineMainWindow):
    """The original tracking window with non-tracking features removed."""

    def _build_ui(self) -> None:
        super()._build_ui()
        settings_panel = self.findChild(QtWidgets.QScrollArea)
        if settings_panel is not None:
            settings_panel.setMinimumWidth(420)
            settings_panel.setMaximumWidth(540)
        previous_panel = self.plot_panel
        replacement = KinematicsPlotPanel()
        replaced_item = self.centralWidget().layout().replaceWidget(
            previous_panel, replacement, QtCore.Qt.FindChildrenRecursively
        )
        if replaced_item is None:
            raise RuntimeError("Could not replace the baseline plot panel.")
        previous_panel.hide()
        previous_panel.setParent(None)
        previous_panel.deleteLater()
        self.plot_panel = replacement

    def __init__(self):
        self.annotation_segments: list[dict] = []
        self.annotation_undo_stack: list[list[dict]] = []
        self.annotation_store: ManualAnnotationStore | None = None
        self.annotation_editing_id: str | None = None
        self.loaded_tracking_path: Path | None = None
        self.loaded_config_path: Path | None = None
        super().__init__()
        self.setWindowTitle(APP_DISPLAY_NAME)
        self._install_tracker_controls()
        info = self.video_info or {"width": 1280, "height": 1024}
        self.apply_config(tracker_defaults(self.config, int(info["width"]), int(info["height"])))
        self._refresh_annotation_ui()
        self.log_msg(
            "Tracker: fast tracking only; full CSV, d/psi/phi/Omega plots and spectra, "
            "revisioned manual-state annotations, lightweight preview, no MP4/debug output."
        )

    def _install_tracker_controls(self) -> None:
        for group in self.findChildren(QtWidgets.QGroupBox):
            if group.title() == "Trap-center calibration":
                group.hide()

        self.fast_mode.setChecked(True)
        self.fast_mode.hide()
        label = self._label_for_field(self.fast_mode)
        if label is not None:
            label.hide()
        self.save_video.hide()
        self.save_debug.hide()

        self.min_radius = QtWidgets.QDoubleSpinBox()
        self.min_radius.setRange(0.0, 5000.0)
        self.min_radius.setDecimals(1)
        self.max_radius = QtWidgets.QDoubleSpinBox()
        self.max_radius.setRange(0.0, 5000.0)
        self.max_radius.setDecimals(1)
        self.roi_x_min = QtWidgets.QSpinBox()
        self.roi_x_max = QtWidgets.QSpinBox()
        self.roi_y_min = QtWidgets.QSpinBox()
        self.roi_y_max = QtWidgets.QSpinBox()
        for widget in (self.roi_x_min, self.roi_x_max, self.roi_y_min, self.roi_y_max):
            widget.setRange(0, 100000)

        settings = next(
            group for group in self.findChildren(QtWidgets.QGroupBox) if group.title() == "Settings"
        )
        form = settings.layout()
        form.addRow("Mode", QtWidgets.QLabel("Fast only (all frames -> CSV)"))
        form.addRow("Min radius (px)", self.min_radius)
        form.addRow("Max radius (px)", self.max_radius)
        form.addRow("ROI x min", self.roi_x_min)
        form.addRow("ROI x max", self.roi_x_max)
        form.addRow("ROI y min", self.roi_y_min)
        form.addRow("ROI y max", self.roi_y_max)

        self.annotation_group = QtWidgets.QGroupBox("Manual state annotation")
        annotation_layout = QtWidgets.QVBoxLayout(self.annotation_group)
        explanation = QtWidgets.QLabel(
            "Observed motion only. Frame ranges are inclusive; torus/chaos are not visual labels."
        )
        explanation.setWordWrap(True)
        explanation.setToolTip(DIRECTION_CONVENTION)
        explanation.setStyleSheet("QLabel { color: #555; }")
        annotation_layout.addWidget(explanation)

        annotation_form = QtWidgets.QFormLayout()
        self.annotation_start = QtWidgets.QSpinBox()
        self.annotation_end = QtWidgets.QSpinBox()
        for widget in (self.annotation_start, self.annotation_end):
            widget.setRange(0, 100000000)
        self.annotation_state = QtWidgets.QComboBox()
        for item in STATE_SPECS:
            self.annotation_state.addItem(item.label, item.key)
            index = self.annotation_state.count() - 1
            self.annotation_state.setItemData(index, item.description, QtCore.Qt.ToolTipRole)
        self.annotation_reviewer = QtWidgets.QLineEdit()
        self.annotation_reviewer.setPlaceholderText("reviewer (recommended)")
        self.annotation_notes = QtWidgets.QLineEdit()
        self.annotation_notes.setPlaceholderText("optional boundary/QC note")
        annotation_form.addRow("Start frame", self.annotation_start)
        annotation_form.addRow("End frame", self.annotation_end)
        annotation_form.addRow("State", self.annotation_state)
        annotation_form.addRow("Reviewer", self.annotation_reviewer)
        annotation_form.addRow("Notes", self.annotation_notes)
        annotation_layout.addLayout(annotation_form)

        range_buttons = QtWidgets.QHBoxLayout()
        self.annotation_set_start_btn = QtWidgets.QPushButton("Start = current")
        self.annotation_set_end_btn = QtWidgets.QPushButton("End = current")
        self.annotation_set_start_btn.clicked.connect(
            lambda: self.annotation_start.setValue(self.current_frame)
        )
        self.annotation_set_end_btn.clicked.connect(
            lambda: self.annotation_end.setValue(self.current_frame)
        )
        range_buttons.addWidget(self.annotation_set_start_btn)
        range_buttons.addWidget(self.annotation_set_end_btn)
        annotation_layout.addLayout(range_buttons)

        self.annotation_save_btn = QtWidgets.QPushButton("Add segment")
        self.annotation_save_btn.clicked.connect(self.save_annotation_segment)
        self.annotation_new_btn = QtWidgets.QPushButton("New segment")
        self.annotation_new_btn.clicked.connect(self.new_annotation_segment)
        edit_buttons = QtWidgets.QHBoxLayout()
        edit_buttons.addWidget(self.annotation_save_btn, 1)
        edit_buttons.addWidget(self.annotation_new_btn)
        annotation_layout.addLayout(edit_buttons)

        self.annotation_table = QtWidgets.QTableWidget(0, 5)
        self.annotation_table.setHorizontalHeaderLabels(
            ["Start", "End", "State", "Reviewer", "Notes"]
        )
        self.annotation_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.annotation_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.annotation_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.annotation_table.setMinimumHeight(150)
        self.annotation_table.itemSelectionChanged.connect(self.on_annotation_selected)
        self.annotation_table.cellDoubleClicked.connect(self.go_to_annotation_start)
        annotation_layout.addWidget(self.annotation_table)

        history_buttons = QtWidgets.QHBoxLayout()
        self.annotation_delete_btn = QtWidgets.QPushButton("Delete selected")
        self.annotation_delete_btn.clicked.connect(self.delete_annotation_segment)
        self.annotation_undo_btn = QtWidgets.QPushButton("Undo edit")
        self.annotation_undo_btn.clicked.connect(self.undo_annotation_edit)
        self.annotation_load_btn = QtWidgets.QPushButton("Load revision")
        self.annotation_load_btn.clicked.connect(self.load_annotation_revision)
        self.annotation_new_session_btn = QtWidgets.QPushButton("New session")
        self.annotation_new_session_btn.clicked.connect(self.new_annotation_session)
        for widget in (
            self.annotation_delete_btn,
            self.annotation_undo_btn,
            self.annotation_load_btn,
            self.annotation_new_session_btn,
        ):
            history_buttons.addWidget(widget)
        annotation_layout.addLayout(history_buttons)

        self.annotation_current_label = QtWidgets.QLabel("Current frame: unlabeled")
        self.annotation_revision_label = QtWidgets.QLabel("No annotation revision saved")
        self.annotation_revision_label.setWordWrap(True)
        self.annotation_revision_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        annotation_layout.addWidget(self.annotation_current_label)
        annotation_layout.addWidget(self.annotation_revision_label)

        panel_layout = settings.parentWidget().layout()
        actions = next(
            group for group in self.findChildren(QtWidgets.QGroupBox) if group.title() == "Actions"
        )
        panel_layout.insertWidget(panel_layout.indexOf(actions), self.annotation_group)
        self.plot_panel.range_selected.connect(self.set_annotation_range)

        self.annotation_shortcuts = []
        for sequence, callback in (
            ("Alt+I", lambda: self.annotation_start.setValue(self.current_frame)),
            ("Alt+O", lambda: self.annotation_end.setValue(self.current_frame)),
            ("Ctrl+Return", self.save_annotation_segment),
        ):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self.annotation_shortcuts.append(shortcut)

    def _annotation_fps(self) -> float | None:
        if not self.video_info:
            return None
        try:
            fps = float(self.video_info.get("fps"))
        except (TypeError, ValueError):
            return None
        return fps if math.isfinite(fps) and fps > 0 else None

    def _annotation_maximum_frame(self) -> int | None:
        if self.video_info:
            try:
                frames = int(self.video_info.get("frames"))
            except (TypeError, ValueError):
                frames = 0
            if frames > 0:
                return frames - 1
        if hasattr(self, "frame_slider"):
            return int(self.frame_slider.maximum())
        return None

    def _time_for_frame(self, frame: int) -> float | None:
        row = self.rows_by_frame.get(int(frame))
        if row is not None:
            value = _number(row, "time_s")
            if math.isfinite(value):
                return value
        fps = self._annotation_fps()
        return float(frame / fps) if fps else None

    def set_annotation_range(self, start_frame: int, end_frame: int) -> None:
        start, end = sorted((int(start_frame), int(end_frame)))
        self.annotation_start.setValue(start)
        self.annotation_end.setValue(end)
        self.log_msg(f"Manual annotation range selected: frames {start}-{end} inclusive")

    def _next_annotation_id(self) -> str:
        maximum = 0
        for segment in self.annotation_segments:
            match = None
            value = str(segment.get("annotation_id") or "")
            if value.startswith("ann_"):
                try:
                    match = int(value.split("_", 1)[1])
                except (TypeError, ValueError):
                    match = None
            if match is not None:
                maximum = max(maximum, match)
        return f"ann_{maximum + 1:04d}"

    def _annotation_record_from_controls(self) -> dict:
        start_frame = self.annotation_start.value()
        end_frame = self.annotation_end.value()
        existing = next(
            (
                item
                for item in self.annotation_segments
                if item.get("annotation_id") == self.annotation_editing_id
            ),
            None,
        )
        now = utc_now()
        return {
            "annotation_id": (
                str(existing["annotation_id"]) if existing is not None else self._next_annotation_id()
            ),
            "revision": existing.get("revision") if existing is not None else None,
            "video_id": Path(self.video_edit.text().strip()).stem,
            "video_path": self.video_edit.text().strip(),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": self._time_for_frame(start_frame),
            "end_time_s": self._time_for_frame(end_frame),
            "manual_state": str(self.annotation_state.currentData()),
            "reviewer": self.annotation_reviewer.text().strip(),
            "notes": self.annotation_notes.text().strip(),
            "created_at_utc": existing.get("created_at_utc") if existing is not None else now,
            "updated_at_utc": now,
        }

    @staticmethod
    def _copy_segments(segments: list[dict]) -> list[dict]:
        return [dict(item) for item in segments]

    def _persist_annotation_revision(self, action: str) -> None:
        video_path = Path(self.video_edit.text().strip())
        if not video_path.is_file():
            raise ValueError("Select an existing video before saving manual annotations.")
        if self.annotation_store is None:
            self.log_msg(
                "Creating annotation session and hashing source video/tracking/config for provenance..."
            )
            QtWidgets.QApplication.processEvents()
            tracking_path = (
                self.loaded_tracking_path
                if self.loaded_tracking_path is not None and self.loaded_tracking_path.is_file()
                else None
            )
            config_path = (
                self.loaded_config_path
                if self.loaded_config_path is not None and self.loaded_config_path.is_file()
                else None
            )
            self.annotation_store = ManualAnnotationStore.create(
                DEFAULT_ANNOTATION_ROOT,
                video_path,
                video_info=self.video_info,
                tracking_csv_path=tracking_path,
                config_path=config_path,
            )
        csv_path, manifest_path, rows = self.annotation_store.save_revision(
            self.annotation_segments,
            action=action,
        )
        self.annotation_segments = rows
        self.annotation_revision_label.setText(
            f"Revision {self.annotation_store.current_revision}: {csv_path}"
        )
        self.annotation_revision_label.setToolTip(str(manifest_path))
        self.log_msg(
            f"Manual annotation revision saved ({action}): {csv_path}; manifest: {manifest_path}"
        )

    def _commit_annotation_segments(self, candidate: list[dict], action: str) -> None:
        previous = self._copy_segments(self.annotation_segments)
        validated = validate_segments(
            candidate,
            fps=self._annotation_fps(),
            maximum_frame=self._annotation_maximum_frame(),
        )
        self.annotation_segments = validated
        try:
            self._persist_annotation_revision(action)
        except Exception:
            self.annotation_segments = previous
            raise
        self.annotation_undo_stack.append(previous)
        self._refresh_annotation_ui()

    def save_annotation_segment(self) -> None:
        try:
            record = self._annotation_record_from_controls()
            candidate = self._copy_segments(self.annotation_segments)
            if self.annotation_editing_id is None:
                candidate.append(record)
                action = f"add {record['annotation_id']}"
            else:
                replaced = False
                for index, item in enumerate(candidate):
                    if item.get("annotation_id") == self.annotation_editing_id:
                        candidate[index] = record
                        replaced = True
                        break
                if not replaced:
                    raise ValueError("The selected annotation no longer exists.")
                action = f"update {record['annotation_id']}"
            self._commit_annotation_segments(candidate, action)
            next_frame = min(
                record["end_frame"] + 1,
                self._annotation_maximum_frame() or record["end_frame"] + 1,
            )
            self.annotation_editing_id = None
            self.annotation_table.clearSelection()
            self.annotation_start.setValue(next_frame)
            self.annotation_end.setValue(next_frame)
            self.annotation_notes.clear()
            self.annotation_save_btn.setText("Add segment")
            self._refresh_annotation_ui()
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Manual state annotation", str(error))

    def new_annotation_segment(self) -> None:
        self.annotation_editing_id = None
        self.annotation_table.clearSelection()
        self.annotation_start.setValue(self.current_frame)
        self.annotation_end.setValue(self.current_frame)
        self.annotation_notes.clear()
        self.annotation_save_btn.setText("Add segment")
        self._refresh_annotation_ui()

    def delete_annotation_segment(self) -> None:
        if self.annotation_editing_id is None:
            return
        annotation_id = self.annotation_editing_id
        candidate = [
            dict(item)
            for item in self.annotation_segments
            if item.get("annotation_id") != annotation_id
        ]
        try:
            self._commit_annotation_segments(candidate, f"delete {annotation_id}")
            self.new_annotation_segment()
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Manual state annotation", str(error))

    def undo_annotation_edit(self) -> None:
        if not self.annotation_undo_stack:
            return
        current = self._copy_segments(self.annotation_segments)
        previous = self.annotation_undo_stack.pop()
        self.annotation_segments = self._copy_segments(previous)
        try:
            self._persist_annotation_revision("undo last edit")
        except Exception as error:
            self.annotation_segments = current
            self.annotation_undo_stack.append(previous)
            QtWidgets.QMessageBox.warning(self, "Manual state annotation", str(error))
            return
        self.annotation_editing_id = None
        self._refresh_annotation_ui()

    def on_annotation_selected(self) -> None:
        row = self.annotation_table.currentRow()
        if row < 0:
            return
        start_item = self.annotation_table.item(row, 0)
        annotation_id = start_item.data(QtCore.Qt.UserRole) if start_item is not None else None
        segment = next(
            (
                item
                for item in self.annotation_segments
                if item.get("annotation_id") == annotation_id
            ),
            None,
        )
        if segment is None:
            return
        self.annotation_editing_id = str(segment["annotation_id"])
        self.annotation_start.setValue(int(segment["start_frame"]))
        self.annotation_end.setValue(int(segment["end_frame"]))
        state_index = self.annotation_state.findData(segment["manual_state"])
        if state_index >= 0:
            self.annotation_state.setCurrentIndex(state_index)
        self.annotation_reviewer.setText(str(segment.get("reviewer") or ""))
        self.annotation_notes.setText(str(segment.get("notes") or ""))
        self.annotation_save_btn.setText("Update selected")

    def go_to_annotation_start(self, row: int, _column: int) -> None:
        item = self.annotation_table.item(row, 0)
        if item is not None:
            self.frame_slider.setValue(int(item.text()))

    def _refresh_annotation_ui(self) -> None:
        if not hasattr(self, "annotation_table"):
            return
        self.annotation_table.blockSignals(True)
        self.annotation_table.setRowCount(0)
        for segment in sorted(
            self.annotation_segments,
            key=lambda item: (int(item["start_frame"]), int(item["end_frame"])),
        ):
            row = self.annotation_table.rowCount()
            self.annotation_table.insertRow(row)
            state_key = str(segment["manual_state"])
            values = (
                int(segment["start_frame"]),
                int(segment["end_frame"]),
                next(
                    (item.label for item in STATE_SPECS if item.key == state_key),
                    state_key,
                ),
                str(segment.get("reviewer") or ""),
                str(segment.get("notes") or ""),
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, segment["annotation_id"])
                if column == 2:
                    item.setBackground(QtGui.QColor(STATE_COLORS.get(state_key, "#64748B")))
                    item.setForeground(QtGui.QColor("#FFFFFF"))
                self.annotation_table.setItem(row, column, item)
        self.annotation_table.blockSignals(False)
        self.annotation_table.resizeColumnsToContents()
        self.annotation_delete_btn.setEnabled(self.annotation_editing_id is not None)
        self.annotation_undo_btn.setEnabled(bool(self.annotation_undo_stack))
        self.plot_panel.set_annotation_segments(self.annotation_segments)
        self._refresh_annotation_status()

    def _refresh_annotation_status(self) -> None:
        if not hasattr(self, "annotation_current_label"):
            return
        selected = next(
            (
                item
                for item in self.annotation_segments
                if int(item["start_frame"]) <= self.current_frame <= int(item["end_frame"])
            ),
            None,
        )
        if selected is None:
            self.annotation_current_label.setText(
                f"Current frame {self.current_frame}: unlabeled"
            )
            self.annotation_current_label.setStyleSheet("")
            return
        state = str(selected["manual_state"])
        display = next((item.label for item in STATE_SPECS if item.key == state), state)
        self.annotation_current_label.setText(
            f"Current frame {self.current_frame}: {display} ({selected['annotation_id']})"
        )
        self.annotation_current_label.setStyleSheet(
            f"QLabel {{ color: {STATE_COLORS.get(state, '#64748B')}; font-weight: 600; }}"
        )

    def load_annotation_revision(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load manual-state annotation revision",
            str(DEFAULT_ANNOTATION_ROOT),
            "Annotation CSV (revision_*_manual_state_segments.csv);;CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            store, segments, manifest = ManualAnnotationStore.load(path)
            source_video = (
                manifest.get("source_provenance", {}).get("video", {}).get("path")
                if manifest
                else None
            )
            current_video = self.video_edit.text().strip()
            if source_video and current_video:
                if Path(source_video).resolve() != Path(current_video).resolve():
                    raise ValueError(
                        "This annotation revision belongs to a different video.\n"
                        f"Revision video: {source_video}\nCurrent video: {current_video}"
                    )
            elif source_video and not current_video:
                self.video_edit.setText(str(source_video))
                self.output_edit.setText(
                    str(
                        DEFAULT_OUTPUT_ROOT
                        / f"{Path(source_video).stem}_tinylev_tracker"
                    )
                )
                self.update_video_info()
            segments = validate_segments(
                segments,
                fps=self._annotation_fps(),
                maximum_frame=self._annotation_maximum_frame(),
            )
            self.annotation_segments = segments
            self.annotation_undo_stack.clear()
            self.annotation_editing_id = None
            if manifest:
                self.annotation_store = store
                self.annotation_revision_label.setText(
                    f"Revision {store.current_revision}: {Path(path).resolve()}"
                )
            else:
                self.annotation_store = None
                self.annotation_revision_label.setText(
                    f"Imported legacy annotation: {Path(path).resolve()}; "
                    "the next edit will start a new hashed session."
                )
            self._refresh_annotation_ui()
            self.log_msg(f"Loaded manual annotations: {len(segments)} segments from {path}")
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Manual state annotation", str(error))

    def new_annotation_session(self) -> None:
        if self.annotation_segments:
            choice = QtWidgets.QMessageBox.question(
                self,
                "New annotation session",
                "Start a new empty annotation session? Existing saved revisions will be preserved.",
            )
            if choice != QtWidgets.QMessageBox.Yes:
                return
        self.annotation_segments = []
        self.annotation_undo_stack.clear()
        self.annotation_store = None
        self.annotation_editing_id = None
        self.annotation_revision_label.setText("No annotation revision saved")
        self.new_annotation_segment()
        self._refresh_annotation_ui()
        self.log_msg("Started a new empty manual-annotation session; prior files were preserved.")

    def _reset_annotation_context_for_video_change(self) -> None:
        self.annotation_segments = []
        self.annotation_undo_stack.clear()
        self.annotation_store = None
        self.annotation_editing_id = None
        self.loaded_tracking_path = None
        if hasattr(self, "annotation_revision_label"):
            self.annotation_revision_label.setText("No annotation revision saved")
        self._refresh_annotation_ui()

    def _label_for_field(self, field):
        parent = field.parentWidget()
        layout = parent.layout() if parent is not None else None
        return layout.labelForField(field) if isinstance(layout, QtWidgets.QFormLayout) else None

    def make_config(self) -> TrackingConfig:
        config = super().make_config()
        if not hasattr(self, "max_radius"):
            return replace(config, fast_mode=True, save_annotated_video=False, save_debug_frames=False)
        if self.max_radius.value() < self.min_radius.value():
            raise ValueError("Max radius must be greater than or equal to min radius.")
        if self.roi_x_max.value() <= self.roi_x_min.value() or self.roi_y_max.value() <= self.roi_y_min.value():
            raise ValueError("ROI maximum values must be greater than ROI minimum values.")
        return replace(
            config,
            fast_mode=True,
            save_annotated_video=False,
            save_debug_frames=False,
            min_radius=self.min_radius.value(),
            max_radius=self.max_radius.value(),
            roi_x_min=self.roi_x_min.value(),
            roi_x_max=self.roi_x_max.value(),
            roi_y_min=self.roi_y_min.value(),
            roi_y_max=self.roi_y_max.value(),
        )

    def apply_config(self, config: TrackingConfig) -> None:
        previous_video = self.video_edit.text().strip() if hasattr(self, "video_edit") else ""
        config = replace(config, fast_mode=True, save_annotated_video=False, save_debug_frames=False)
        super().apply_config(config)
        if previous_video and previous_video != config.video_path:
            self._reset_annotation_context_for_video_change()
        if hasattr(self, "max_radius"):
            self.min_radius.setValue(float(config.min_radius))
            self.max_radius.setValue(float(config.max_radius))
            self.roi_x_min.setValue(int(config.roi_x_min))
            self.roi_x_max.setValue(int(config.roi_x_max))
            self.roi_y_min.setValue(int(config.roi_y_min))
            self.roi_y_max.setValue(int(config.roi_y_max))

    def browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(Path(self.video_edit.text()).parent if self.video_edit.text() else RAW_DATA_ROOT),
            "Video Files (*.avi *.mp4 *.mov *.mkv);;All Files (*)",
        )
        if not path:
            return
        if self.video_edit.text().strip() != path:
            self._reset_annotation_context_for_video_change()
            self.loaded_config_path = None
            for field in (self.trap_x, self.trap_y, self.pixels_per_mm):
                field.setValue(field.minimum())
        self.video_edit.setText(path)
        self.output_edit.setText(str(DEFAULT_OUTPUT_ROOT / f"{Path(path).stem}_tinylev_tracker"))
        self.update_video_info()
        if self.video_info:
            self.roi_x_min.setValue(0)
            self.roi_x_max.setValue(int(self.video_info["width"]))
            self.roi_y_min.setValue(0)
            self.roi_y_max.setValue(int(self.video_info["height"]))

    def update_video_info(self) -> None:
        super().update_video_info()
        if hasattr(self, "annotation_start"):
            maximum = self._annotation_maximum_frame()
            if maximum is not None:
                self.annotation_start.setMaximum(maximum)
                self.annotation_end.setMaximum(maximum)
            self._refresh_annotation_status()

    def load_cache(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select tracking CSV",
            self.output_edit.text(),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        rows = read_tracking_csv(path)
        self.rows = rows
        self.rows_by_frame = {int(float(row["frame"])): row for row in rows if "frame" in row}
        self.loaded_tracking_path = Path(path).resolve()
        self.plot_panel.set_rows(rows)
        self.plot_panel.set_annotation_segments(self.annotation_segments)
        self.log_msg(f"Loaded cache rows: {len(rows)} from {path}")
        if rows:
            self.frame_slider.setValue(int(float(rows[0]["frame"])))

    def load_config_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select config.json",
            self.output_edit.text(),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        config = read_config(path)
        self.loaded_config_path = Path(path).resolve()
        self.apply_config(config)
        self.log_msg(f"Loaded config: {path}")

    def run_batch(self) -> None:
        requested = self.output_edit.text().strip()
        if requested:
            selected = unused_output_dir(requested)
            if selected != Path(requested):
                self.output_edit.setText(str(selected))
                self.log_msg(f"Existing output preserved; using new run directory: {selected}")
        super().run_batch()

    def on_batch_done(self, rows) -> None:
        super().on_batch_done(rows)
        tracking_path = self.make_config().resolved_output_dir() / "tracking.csv"
        self.loaded_tracking_path = tracking_path.resolve() if tracking_path.is_file() else None
        self.plot_panel.set_annotation_segments(self.annotation_segments)

    def on_slider_changed(self, value: int) -> None:
        super().on_slider_changed(value)
        self._refresh_annotation_status()

    def set_busy(self, busy: bool) -> None:
        super().set_busy(busy)
        if hasattr(self, "max_radius"):
            for widget in (
                self.min_radius,
                self.max_radius,
                self.roi_x_min,
                self.roi_x_max,
                self.roi_y_min,
                self.roi_y_max,
            ):
                widget.setEnabled(not busy)
        if hasattr(self, "annotation_group"):
            self.annotation_group.setEnabled(not busy)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
