from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .core import (
    DEFAULT_LAYERS,
    TrackingConfig,
    compute_pair_kinematics,
    dependency_report,
    draw_overlay,
    get_video_info,
    load_model,
    process_frame,
    read_config,
    read_tracking_csv,
    read_video_frame,
    track_video,
)

APP_NAME = "Tinylev Tracking"
APP_VERSION = "v0.1"
APP_DISPLAY_NAME = f"{APP_NAME} {APP_VERSION}"


def frame_to_qpixmap(frame_bgr: np.ndarray, max_size: QtCore.QSize | None = None, smooth: bool = True) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    pixmap = QtGui.QPixmap.fromImage(image)
    if max_size is not None and max_size.width() > 0 and max_size.height() > 0:
        mode = QtCore.Qt.SmoothTransformation if smooth else QtCore.Qt.FastTransformation
        pixmap = pixmap.scaled(max_size, QtCore.Qt.KeepAspectRatio, mode)
    return pixmap


class TrackingWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    frame_ready = QtCore.pyqtSignal(object, object)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config: TrackingConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            rows = track_video(
                self.config,
                progress_cb=lambda done, total: self.progress.emit(done, total),
                frame_cb=lambda frame, row: self.frame_ready.emit(frame, row),
                stop_check=lambda: self._stop,
            )
            self.finished_ok.emit(rows)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PreviewWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object, object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config: TrackingConfig, frame_idx: int, layers: dict[str, bool], parent=None):
        super().__init__(parent)
        self.config = config
        self.frame_idx = frame_idx
        self.layers = layers

    def run(self) -> None:
        try:
            frame = read_video_frame(self.config.video_path, self.frame_idx)
            if frame is None:
                raise RuntimeError(f"Could not read frame {self.frame_idx}")
            info = get_video_info(self.config.video_path)
            model = load_model(self.config.model_path)
            detections, selected, row = process_frame(model, frame, self.frame_idx, float(info["fps"]), self.config)
            overlay = draw_overlay(frame, detections, selected, self.config, frame_idx=self.frame_idx, row=row, layers=self.layers)
            self.finished_ok.emit(overlay, row, detections)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PlotPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[dict] = []
        self.current_frame: int | None = None
        self.playhead_line = None
        self.playhead_point = None
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.series = QtWidgets.QComboBox()
        self.series.addItems(
            [
                "CM trap distance",
                "CM XY",
                "CM XY trajectory",
                "Orientation",
                "Angular velocity",
                "Relative angle phi",
                "Spin state timeline",
                "Particle XY",
                "Confidence",
            ]
        )
        self.series.currentIndexChanged.connect(self.redraw)
        controls.addWidget(QtWidgets.QLabel("Plot"))
        controls.addWidget(self.series, 1)
        layout.addLayout(controls)
        self.figure = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

    @staticmethod
    def _num(row: dict, key: str) -> float | None:
        value = row.get(key)
        if value in ("", None):
            return None
        try:
            value = float(value)
        except Exception:
            return None
        return value if np.isfinite(value) else None

    def _num_any(self, row: dict, *keys: str) -> float | None:
        for key in keys:
            value = self._num(row, key)
            if value is not None:
                return value
        return None

    def _time_rows(self) -> list[dict]:
        rows = [r for r in self.rows if self._num(r, "time_s") is not None]
        return sorted(rows, key=lambda r: self._num(r, "time_s") or 0.0)

    def _ok_rows(self) -> list[dict]:
        return [r for r in self._time_rows() if r.get("status") == "ok"]

    def _series_values(self, rows: list[dict], key: str) -> list[float]:
        return [self._num(r, key) if self._num(r, key) is not None else np.nan for r in rows]

    def frame_time(self, rows: list[dict], frame: int | None = None) -> float | None:
        if not rows:
            return None
        target = self.current_frame if frame is None else frame
        if target is None:
            return None
        row = min(rows, key=lambda r: abs((self._num(r, "frame") or 0) - target))
        return self._num(row, "time_s")

    def current_trajectory_point(self) -> tuple[float, float] | None:
        rows = self._ok_rows()
        if not rows or self.current_frame is None:
            return None
        row = min(rows, key=lambda r: abs((self._num(r, "frame") or 0) - self.current_frame))
        x = self._num(row, "cm_dx_from_trap_px")
        y = self._num(row, "cm_dy_from_trap_px")
        if x is None or y is None:
            return None
        return x, y

    def update_playhead(self) -> None:
        if self.playhead_line is not None:
            t = self.frame_time(self._time_rows())
            if t is not None:
                self.playhead_line.set_xdata([t, t])
                self.canvas.draw_idle()
        if self.playhead_point is not None:
            point = self.current_trajectory_point()
            if point is not None:
                self.playhead_point.set_data([point[0]], [point[1]])
                self.canvas.draw_idle()

    def set_rows(self, rows: list[dict]) -> None:
        self.rows = rows or []
        self.redraw()

    def set_current_frame(self, frame: int, redraw: bool = True) -> None:
        self.current_frame = frame
        if redraw:
            self.redraw()
        else:
            self.update_playhead()

    def reversal_times(self, rows: list[dict]) -> list[float]:
        reversals: list[float] = []
        prev_state = 0
        for row in rows:
            state_value = self._num(row, "spin_state")
            state = int(state_value) if state_value is not None else 0
            if state == 0:
                continue
            if prev_state and state != prev_state:
                t = self._num(row, "time_s")
                if t is not None:
                    reversals.append(t)
            prev_state = state
        return reversals

    def redraw(self) -> None:
        self.figure.clear()
        self.playhead_line = None
        self.playhead_point = None
        ax = self.figure.add_subplot(111)
        mode = self.series.currentText()
        time_rows = self._time_rows()
        ok_rows = self._ok_rows()
        if not time_rows:
            ax.set_title("No tracking data")
            self.canvas.draw_idle()
            return

        if mode == "CM XY trajectory":
            rows = [r for r in ok_rows if self._num(r, "cm_dx_from_trap_px") is not None and self._num(r, "cm_dy_from_trap_px") is not None]
            if not rows:
                ax.set_title("No CM trajectory data")
                self.canvas.draw_idle()
                return
            x = self._series_values(rows, "cm_dx_from_trap_px")
            y = self._series_values(rows, "cm_dy_from_trap_px")
            ax.plot(x, y, "-o", markersize=2.5, linewidth=1, label="CM")
            ax.plot([0], [0], marker="x", markersize=9, color="black", label="trap")
            ax.plot([x[0]], [y[0]], marker="o", markersize=7, color="green", linestyle="None", label="start")
            ax.plot([x[-1]], [y[-1]], marker="s", markersize=7, color="red", linestyle="None", label="end")
            point = self.current_trajectory_point()
            if point is not None:
                self.playhead_point = ax.plot([point[0]], [point[1]], marker="o", markersize=8, color="black", linestyle="None", label="current")[0]
            ax.set_xlabel("x_CM - x_trap (px)")
            ax.set_ylabel("y_CM - y_trap (px)")
            ax.set_aspect("equal", adjustable="box")
        else:
            x = [self._num(r, "time_s") or 0.0 for r in time_rows]
            if mode == "CM XY":
                rows = ok_rows
                x = [self._num(r, "time_s") or 0.0 for r in rows]
                ax.plot(x, [self._num_any(r, "cm_x_px", "centroid_x_px") for r in rows], label="CM x")
                ax.plot(x, [self._num_any(r, "cm_y_px", "centroid_y_px") for r in rows], label="CM y")
                ax.set_ylabel("px")
            elif mode == "CM trap distance":
                rows = ok_rows
                x = [self._num(r, "time_s") or 0.0 for r in rows]
                ax.plot(x, [self._num_any(r, "cm_distance_from_trap_px", "centroid_distance_from_trap_px") for r in rows], label="CM-trap distance")
                ax.set_ylabel("px")
            elif mode == "Orientation":
                rows = ok_rows
                x = [self._num(r, "time_s") or 0.0 for r in rows]
                ax.plot(x, [self._num(r, "angle_deg") for r in rows], label="orientation")
                ax.set_ylabel("deg")
            elif mode == "Angular velocity":
                rows = time_rows
                ax.plot(x, self._series_values(rows, "omega_rad_s"), label="omega")
                spin_freq = next((self._num(r, "spin_freq_hz") for r in rows if self._num(r, "spin_freq_hz") is not None), None)
                title_value = "nan" if spin_freq is None else f"{spin_freq:.2f}"
                ax.set_title(f"Angular velocity, mean |f| = {title_value} Hz")
                ax.set_ylabel("angular velocity (rad/s)")
            elif mode == "Relative angle phi":
                rows = time_rows
                ax.plot(x, self._series_values(rows, "phi_rad"), label="phi")
                ax.axhline(0, color="black", linestyle="--", alpha=0.35, linewidth=1)
                ax.axhline(math.pi, color="gray", linestyle="--", alpha=0.35, linewidth=1)
                ax.axhline(-math.pi, color="gray", linestyle="--", alpha=0.35, linewidth=1)
                ax.set_ylabel("phi = psi - theta (rad)")
            elif mode == "Spin state timeline":
                rows = time_rows
                states = [int(self._num(r, "spin_state") or 0) for r in rows]
                ax.step(x, states, where="mid", label="state")
                reversals = self.reversal_times(rows)
                for t in reversals:
                    ax.axvline(t, color="red", linestyle="--", alpha=0.5, linewidth=1)
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels(["CW", "static/rocking", "CCW"])
                ax.set_ylim(-1.4, 1.4)
                ax.set_title(f"Spin state timeline, reversals = {len(reversals)}")
                ax.set_ylabel("state")
            elif mode == "Confidence":
                rows = ok_rows
                x = [self._num(r, "time_s") or 0.0 for r in rows]
                max_count = max((int(self._num(r, "selected_count") or 0) for r in rows), default=0)
                for idx in range(1, max_count + 1):
                    ax.plot(x, [self._num(r, f"particle_{idx}_conf") for r in rows], label=f"p{idx}")
                ax.set_ylabel("confidence")
            else:
                rows = ok_rows
                x = [self._num(r, "time_s") or 0.0 for r in rows]
                max_count = max((int(self._num(r, "selected_count") or 0) for r in rows), default=0)
                for idx in range(1, max_count + 1):
                    ax.plot(x, [self._num(r, f"particle_{idx}_cx_px") for r in rows], label=f"p{idx} x")
                    ax.plot(x, [self._num(r, f"particle_{idx}_cy_px") for r in rows], label=f"p{idx} y")
                ax.set_ylabel("px")

            t = self.frame_time(time_rows)
            if t is not None:
                self.playhead_line = ax.axvline(t, color="black", alpha=0.35, linewidth=1)
            ax.set_xlabel("time (s)")

        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        self.canvas.draw_idle()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1500, 900)
        self.config = TrackingConfig()
        self.video_info: dict | None = None
        self.rows: list[dict] = []
        self.rows_by_frame: dict[int, dict] = {}
        self.current_frame = 0
        self.worker: TrackingWorker | None = None
        self.preview_worker: PreviewWorker | None = None
        self.raw_frame: np.ndarray | None = None
        self.last_overlay: np.ndarray | None = None
        self.layer_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.playback_cap: cv2.VideoCapture | None = None
        self.playback_cap_path: str | None = None
        self.playback_next_frame: int | None = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self.advance_playback)

        self._build_ui()
        self._load_defaults()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel("Load a video or run preview")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(860, 560)
        self.video_label.setStyleSheet("QLabel { background: #111; color: #ddd; }")
        left.addWidget(self.video_label, 1)

        slider_row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_fps = QtWidgets.QSpinBox()
        self.play_fps.setRange(1, 1000)
        self.play_fps.setValue(30)
        self.play_fps.setSuffix(" fps")
        self.play_fps.setMinimumWidth(86)
        self.play_fps.valueChanged.connect(self.update_playback_interval)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        self.frame_label = QtWidgets.QLabel("frame 0")
        slider_row.addWidget(self.play_btn)
        slider_row.addWidget(self.play_fps)
        slider_row.addWidget(self.frame_slider, 1)
        slider_row.addWidget(self.frame_label)
        left.addLayout(slider_row)

        self.plot_panel = PlotPanel()
        left.addWidget(self.plot_panel, 0)
        root.addLayout(left, 1)

        panel = QtWidgets.QScrollArea()
        panel.setWidgetResizable(True)
        panel_widget = QtWidgets.QWidget()
        panel.setWidget(panel_widget)
        form = QtWidgets.QVBoxLayout(panel_widget)
        root.addWidget(panel, 0)

        self.video_edit = self._path_row(form, "Video", self.browse_video)
        self.model_edit = self._path_row(form, "Model", self.browse_model)
        self.output_edit = self._path_row(form, "Output", self.browse_output_dir)

        self.trap_x = QtWidgets.QDoubleSpinBox()
        self.trap_x.setRange(-10000, 10000)
        self.trap_x.setDecimals(3)
        self.trap_y = QtWidgets.QDoubleSpinBox()
        self.trap_y.setRange(-10000, 10000)
        self.trap_y.setDecimals(3)
        self.pixels_per_mm = QtWidgets.QDoubleSpinBox()
        self.pixels_per_mm.setRange(0.001, 100000)
        self.pixels_per_mm.setDecimals(4)
        self.start_frame = QtWidgets.QSpinBox()
        self.start_frame.setRange(0, 100000000)
        self.end_frame = QtWidgets.QSpinBox()
        self.end_frame.setRange(0, 100000000)
        self.end_frame.setSpecialValueText("Full")
        self.particle_count = QtWidgets.QSpinBox()
        self.particle_count.setRange(0, 50)
        self.particle_count.setSpecialValueText("Auto")
        self.conf = QtWidgets.QDoubleSpinBox()
        self.conf.setRange(0.01, 1.0)
        self.conf.setSingleStep(0.01)
        self.conf.setDecimals(3)
        self.mask_open = QtWidgets.QSpinBox()
        self.mask_open.setRange(0, 31)
        self.mask_erode = QtWidgets.QSpinBox()
        self.mask_erode.setRange(0, 31)
        self.radius_scale = QtWidgets.QDoubleSpinBox()
        self.radius_scale.setRange(0.5, 1.5)
        self.radius_scale.setSingleStep(0.01)
        self.radius_scale.setDecimals(3)

        calibration = QtWidgets.QGroupBox("Calibration")
        calibration_grid = QtWidgets.QFormLayout(calibration)
        for label, widget in [
            ("Trap X", self.trap_x),
            ("Trap Y", self.trap_y),
            ("Pixels/mm", self.pixels_per_mm),
        ]:
            calibration_grid.addRow(label, widget)
        form.addWidget(calibration)

        settings = QtWidgets.QGroupBox("Settings")
        settings_grid = QtWidgets.QFormLayout(settings)
        for label, widget in [
            ("Start frame", self.start_frame),
            ("End frame", self.end_frame),
            ("Particle count", self.particle_count),
            ("YOLO conf", self.conf),
            ("Mask open px", self.mask_open),
            ("Mask erode px", self.mask_erode),
            ("Circle scale", self.radius_scale),
        ]:
            settings_grid.addRow(label, widget)
        form.addWidget(settings)

        layers = QtWidgets.QGroupBox("Layers")
        layer_layout = QtWidgets.QGridLayout(layers)
        for idx, name in enumerate(DEFAULT_LAYERS):
            check = QtWidgets.QCheckBox(name)
            check.setChecked(DEFAULT_LAYERS[name])
            check.stateChanged.connect(self.refresh_current_frame)
            self.layer_checks[name] = check
            layer_layout.addWidget(check, idx // 2, idx % 2)
        form.addWidget(layers)

        actions = QtWidgets.QGroupBox("Actions")
        actions_layout = QtWidgets.QVBoxLayout(actions)
        self.preview_btn = QtWidgets.QPushButton("Preview current frame")
        self.preview_btn.clicked.connect(self.run_preview)
        self.batch_btn = QtWidgets.QPushButton("Run batch tracking")
        self.batch_btn.clicked.connect(self.run_batch)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_worker)
        self.load_cache_btn = QtWidgets.QPushButton("Load cache CSV")
        self.load_cache_btn.clicked.connect(self.load_cache)
        self.open_config_btn = QtWidgets.QPushButton("Load config.json")
        self.open_config_btn.clicked.connect(self.load_config_file)
        self.check_deps_btn = QtWidgets.QPushButton("Check Python deps")
        self.check_deps_btn.clicked.connect(self.check_dependencies)
        actions_layout.addWidget(self.preview_btn)
        actions_layout.addWidget(self.batch_btn)
        actions_layout.addWidget(self.stop_btn)
        actions_layout.addWidget(self.load_cache_btn)
        actions_layout.addWidget(self.open_config_btn)
        actions_layout.addWidget(self.check_deps_btn)
        form.addWidget(actions)

        self.save_video = QtWidgets.QCheckBox("Save annotated video")
        self.save_video.setChecked(True)
        self.save_debug = QtWidgets.QCheckBox("Save debug frames")
        self.save_debug.setChecked(True)
        form.addWidget(self.save_video)
        form.addWidget(self.save_debug)

        self.progress = QtWidgets.QProgressBar()
        form.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        form.addWidget(self.log, 1)
        credit = QtWidgets.QLabel("NYU Grier Group")
        credit.setAlignment(QtCore.Qt.AlignRight)
        credit.setStyleSheet("QLabel { color: #666; padding-top: 4px; }")
        form.addWidget(credit)
        self.log_msg(dependency_report())

    def _path_row(self, parent_layout: QtWidgets.QVBoxLayout, label: str, callback) -> QtWidgets.QLineEdit:
        group = QtWidgets.QGroupBox(label)
        layout = QtWidgets.QHBoxLayout(group)
        edit = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("Browse")
        button.clicked.connect(callback)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        parent_layout.addWidget(group)
        return edit

    def _load_defaults(self) -> None:
        self.video_edit.setText(self.config.video_path)
        self.model_edit.setText(self.config.model_path)
        self.output_edit.setText(str(self.config.resolved_output_dir()))
        self.trap_x.setValue(self.config.trap_x)
        self.trap_y.setValue(self.config.trap_y)
        self.pixels_per_mm.setValue(self.config.pixels_per_mm)
        self.start_frame.setValue(self.config.start_frame)
        self.end_frame.setValue(0)
        self.particle_count.setValue(self.config.particle_count)
        self.conf.setValue(self.config.conf)
        self.mask_open.setValue(self.config.mask_open_px)
        self.mask_erode.setValue(self.config.mask_erode_px)
        self.radius_scale.setValue(self.config.circle_radius_scale)
        self.update_video_info()

    def log_msg(self, text: str) -> None:
        self.log.appendPlainText(text)

    def check_dependencies(self) -> None:
        report = dependency_report()
        self.log_msg(report)
        QtWidgets.QMessageBox.information(self, "Python dependency report", report)

    def browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", str(Path(self.video_edit.text()).parent), "Video Files (*.avi *.mp4 *.mov *.mkv);;All Files (*)")
        if path:
            self.video_edit.setText(path)
            self.output_edit.setText(str(Path(path).parent / f"{Path(path).stem}_particle_tracking_app"))
            self.update_video_info()

    def browse_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YOLO model", str(Path(self.model_edit.text()).parent), "Model Files (*.pt);;All Files (*)")
        if path:
            self.model_edit.setText(path)

    def browse_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def update_video_info(self) -> None:
        path = self.video_edit.text().strip()
        if not path or not Path(path).exists():
            return
        try:
            self.close_playback_capture()
            self.video_info = get_video_info(path)
            max_frame = max(0, int(self.video_info["frames"]) - 1)
            self.frame_slider.setMaximum(max_frame)
            self.end_frame.setMaximum(max_frame + 1)
            self.start_frame.setMaximum(max_frame)
            playback_fps = int(round(max(float(self.video_info["fps"]), 1.0)))
            self.play_fps.setValue(playback_fps)
            self.update_playback_interval()
            self.log_msg(f"Video: {self.video_info['width']}x{self.video_info['height']}, {self.video_info['frames']} frames, {self.video_info['fps']:.3f} fps")
            self.show_raw_frame(self.start_frame.value())
        except Exception as exc:
            self.log_msg(f"Video info error: {exc}")

    def make_config(self) -> TrackingConfig:
        end = self.end_frame.value()
        return TrackingConfig(
            video_path=self.video_edit.text().strip(),
            model_path=self.model_edit.text().strip(),
            output_dir=self.output_edit.text().strip(),
            trap_x=self.trap_x.value(),
            trap_y=self.trap_y.value(),
            pixels_per_mm=self.pixels_per_mm.value(),
            start_frame=self.start_frame.value(),
            end_frame=None if end == 0 else end,
            frame_stride=1,
            conf=self.conf.value(),
            particle_count=self.particle_count.value(),
            mask_open_px=self.mask_open.value(),
            mask_erode_px=self.mask_erode.value(),
            circle_radius_scale=self.radius_scale.value(),
            save_annotated_video=self.save_video.isChecked(),
            save_debug_frames=self.save_debug.isChecked(),
        )

    def current_layers(self) -> dict[str, bool]:
        return {name: check.isChecked() for name, check in self.layer_checks.items()}

    def set_busy(self, busy: bool) -> None:
        if busy:
            self.stop_playback()
        self.preview_btn.setEnabled(not busy)
        self.batch_btn.setEnabled(not busy)
        self.load_cache_btn.setEnabled(not busy)
        self.open_config_btn.setEnabled(not busy)
        self.play_btn.setEnabled(not busy)

    def run_preview(self) -> None:
        if self.preview_worker is not None and self.preview_worker.isRunning():
            return
        config = self.make_config()
        self.set_busy(True)
        self.log_msg(f"Preview frame {self.current_frame}")
        self.preview_worker = PreviewWorker(config, self.current_frame, self.current_layers())
        self.preview_worker.finished_ok.connect(self.on_preview_done)
        self.preview_worker.failed.connect(self.on_worker_failed)
        self.preview_worker.start()

    def on_preview_done(self, frame, row, _detections) -> None:
        self.last_overlay = frame
        self.show_frame(frame)
        self.rows_by_frame[int(row.get("frame", self.current_frame))] = row
        config = self.make_config()
        rows = list(self.rows_by_frame.values())
        compute_pair_kinematics(rows, config.trap_x, config.trap_y, config.pixels_per_mm)
        self.rows = rows
        self.rows_by_frame = {int(float(r["frame"])): r for r in rows if "frame" in r}
        self.plot_panel.set_rows(rows)
        self.plot_panel.set_current_frame(self.current_frame)
        self.log_msg(f"Preview status: {row.get('status')}, candidates={row.get('num_candidates')}")
        self.set_busy(False)

    def run_batch(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        config = self.make_config()
        self.progress.setValue(0)
        self.set_busy(True)
        self.log_msg(f"Batch start: {config.video_path}")
        self.worker = TrackingWorker(config)
        self.worker.progress.connect(self.on_progress)
        self.worker.frame_ready.connect(self.on_worker_frame)
        self.worker.finished_ok.connect(self.on_batch_done)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.start()

    def stop_worker(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.log_msg("Stop requested")

    def on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(min(done, max(total, 1)))

    def on_worker_frame(self, frame, row) -> None:
        self.last_overlay = frame
        self.show_frame(frame)
        try:
            self.current_frame = int(float(row.get("frame", self.current_frame)))
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(f"frame {self.current_frame}")
        except Exception:
            pass

    def on_batch_done(self, rows) -> None:
        config = self.make_config()
        compute_pair_kinematics(rows, config.trap_x, config.trap_y, config.pixels_per_mm)
        self.rows = rows
        self.rows_by_frame = {int(float(r["frame"])): r for r in rows if "frame" in r}
        self.plot_panel.set_rows(rows)
        self.plot_panel.set_current_frame(self.current_frame)
        self.log_msg(f"Batch done. Rows: {len(rows)}. Output: {self.make_config().resolved_output_dir()}")
        self.set_busy(False)

    def on_worker_failed(self, text: str) -> None:
        self.log_msg(text)
        QtWidgets.QMessageBox.critical(self, "Tracking error", text)
        self.set_busy(False)

    def load_cache(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select tracking CSV", self.output_edit.text(), "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        rows = read_tracking_csv(path)
        config = self.make_config()
        compute_pair_kinematics(rows, config.trap_x, config.trap_y, config.pixels_per_mm)
        self.rows = rows
        self.rows_by_frame = {int(float(r["frame"])): r for r in rows if "frame" in r}
        self.plot_panel.set_rows(rows)
        self.log_msg(f"Loaded cache rows: {len(rows)} from {path}")
        if rows:
            self.frame_slider.setValue(int(float(rows[0]["frame"])))

    def load_config_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select config.json", self.output_edit.text(), "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        config = read_config(path)
        self.apply_config(config)
        self.log_msg(f"Loaded config: {path}")

    def apply_config(self, config: TrackingConfig) -> None:
        self.config = config
        self.video_edit.setText(config.video_path)
        self.model_edit.setText(config.model_path)
        self.output_edit.setText(config.output_dir or str(config.resolved_output_dir()))
        self.trap_x.setValue(config.trap_x)
        self.trap_y.setValue(config.trap_y)
        self.pixels_per_mm.setValue(config.pixels_per_mm)
        self.start_frame.setValue(config.start_frame)
        self.end_frame.setValue(0 if config.end_frame is None else config.end_frame)
        self.particle_count.setValue(getattr(config, "particle_count", 2))
        self.conf.setValue(config.conf)
        self.mask_open.setValue(config.mask_open_px)
        self.mask_erode.setValue(config.mask_erode_px)
        self.radius_scale.setValue(config.circle_radius_scale)
        self.save_video.setChecked(config.save_annotated_video)
        self.save_debug.setChecked(config.save_debug_frames)
        self.update_video_info()

    def show_raw_frame(self, frame_idx: int) -> None:
        path = self.video_edit.text().strip()
        if not path or not Path(path).exists():
            return
        frame = self.read_video_frame_cached(path, frame_idx)
        if frame is None:
            return
        self.raw_frame = frame
        row = self.rows_by_frame.get(frame_idx)
        if row:
            overlay = draw_overlay(frame, None, None, self.make_config(), frame_idx=frame_idx, row=row, layers=self.current_layers())
            self.show_frame(overlay)
        else:
            self.show_frame(frame)

    def show_frame(self, frame: np.ndarray) -> None:
        self.video_label.setPixmap(frame_to_qpixmap(frame, self.video_label.size(), smooth=not self.play_timer.isActive()))

    def close_playback_capture(self) -> None:
        if self.playback_cap is not None:
            self.playback_cap.release()
        self.playback_cap = None
        self.playback_cap_path = None
        self.playback_next_frame = None

    def read_video_frame_cached(self, path: str, frame_idx: int) -> np.ndarray | None:
        if self.playback_cap is None or self.playback_cap_path != path:
            self.close_playback_capture()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return read_video_frame(path, frame_idx)
            self.playback_cap = cap
            self.playback_cap_path = path
            self.playback_next_frame = None

        if self.playback_next_frame != frame_idx:
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = self.playback_cap.read()
        if not ok:
            self.playback_next_frame = None
            return None
        self.playback_next_frame = int(frame_idx) + 1
        return frame

    def update_playback_interval(self) -> None:
        interval_ms = max(1, int(round(1000 / max(1, self.play_fps.value()))))
        self.play_timer.setInterval(interval_ms)

    def toggle_playback(self, checked: bool) -> None:
        if checked:
            self.start_playback()
        else:
            self.stop_playback()

    def start_playback(self) -> None:
        path = self.video_edit.text().strip()
        if not path or not Path(path).exists():
            self.play_btn.setChecked(False)
            return
        if self.frame_slider.value() >= self.frame_slider.maximum():
            self.frame_slider.setValue(self.frame_slider.minimum())
        self.update_playback_interval()
        self.play_btn.setText("Pause")
        self.play_btn.setChecked(True)
        self.play_timer.start()

    def stop_playback(self) -> None:
        self.play_timer.stop()
        if hasattr(self, "play_btn"):
            self.play_btn.setChecked(False)
            self.play_btn.setText("Play")
        self.plot_panel.set_current_frame(self.current_frame, redraw=True)

    def advance_playback(self) -> None:
        value = self.frame_slider.value()
        if value >= self.frame_slider.maximum():
            self.stop_playback()
            return
        self.frame_slider.setValue(value + 1)

    def refresh_current_frame(self) -> None:
        self.show_raw_frame(self.current_frame)

    def on_slider_changed(self, value: int) -> None:
        self.current_frame = value
        self.frame_label.setText(f"frame {value}")
        self.plot_panel.set_current_frame(value, redraw=not self.play_timer.isActive())
        self.show_raw_frame(value)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.last_overlay is not None:
            self.show_frame(self.last_overlay)
        elif self.raw_frame is not None:
            self.show_frame(self.raw_frame)

    def closeEvent(self, event) -> None:
        self.stop_playback()
        self.close_playback_capture()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
