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
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRAP_CENTER_OUTPUT_ROOT,
    RAW_DATA_ROOT,
    TrackingConfig,
    calibrate_trap_center,
    dependency_report,
    draw_overlay,
    get_video_info,
    load_model,
    process_frame,
    read_image,
    read_config,
    read_tracking_csv,
    read_video_frame,
    track_video,
)

APP_NAME = "Tinylev Tracking"
APP_VERSION = "v0.3"
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


class TrapCenterWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, image_paths: list[str], config: TrackingConfig, output_dir: str, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.config = config
        self.output_dir = output_dir
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            result = calibrate_trap_center(
                self.image_paths,
                self.config,
                self.output_dir,
                progress_cb=lambda done, total: self.progress.emit(done, total),
                stop_check=lambda: self._stop,
            )
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PlotPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[dict] = []
        self.current_frame: int | None = None
        self.axes = None
        self.ok_rows: list[dict] = []
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.series = QtWidgets.QComboBox()
        self.series.addItems(
            [
                "CM XY",
                "CM trap distance",
                "Orientation",
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
        self.playback_cursor = QtWidgets.QFrame(self.canvas)
        self.playback_cursor.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.playback_cursor.setStyleSheet("background-color: rgba(20, 20, 20, 190);")
        self.playback_cursor.setFixedWidth(2)
        self.playback_cursor.hide()

    def set_rows(self, rows: list[dict]) -> None:
        self.rows = rows or []
        self.redraw()

    def set_current_frame(self, frame: int, redraw: bool = False) -> None:
        self.current_frame = frame
        if redraw:
            self.redraw()
        else:
            self._update_playback_cursor()

    @staticmethod
    def _num(row: dict, key: str) -> float | None:
        value = row.get(key)
        if value in ("", None):
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _num_any(self, row: dict, *keys: str) -> float | None:
        for key in keys:
            value = self._num(row, key)
            if value is not None:
                return value
        return None

    def redraw(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.axes = ax
        ok_rows = [r for r in self.rows if r.get("status") == "ok"]
        self.ok_rows = ok_rows
        if not ok_rows:
            ax.set_title("No tracking data")
            self.playback_cursor.hide()
            self.canvas.draw()
            return
        x = [self._num(r, "time_s") or 0.0 for r in ok_rows]
        mode = self.series.currentText()

        if mode == "CM XY":
            ax.plot(x, [self._num_any(r, "cm_x_px", "centroid_x_px") for r in ok_rows], label="CM x")
            ax.plot(x, [self._num_any(r, "cm_y_px", "centroid_y_px") for r in ok_rows], label="CM y")
            ax.set_ylabel("px")
        elif mode == "CM trap distance":
            ax.plot(
                x,
                [self._num_any(r, "cm_distance_from_trap_px", "centroid_distance_from_trap_px") for r in ok_rows],
                label="CM-trap distance",
            )
            ax.set_ylabel("px")
        elif mode == "Orientation":
            ax.plot(x, [self._num(r, "angle_deg") for r in ok_rows], label="orientation")
            ax.set_ylabel("deg")
        elif mode == "Confidence":
            max_count = max(int(self._num(r, "selected_count") or 0) for r in ok_rows)
            for idx in range(1, max_count + 1):
                ax.plot(x, [self._num(r, f"particle_{idx}_conf") for r in ok_rows], label=f"p{idx}")
            ax.set_ylabel("confidence")
        else:
            max_count = max(int(self._num(r, "selected_count") or 0) for r in ok_rows)
            for idx in range(1, max_count + 1):
                ax.plot(x, [self._num(r, f"particle_{idx}_cx_px") for r in ok_rows], label=f"p{idx} x")
                ax.plot(x, [self._num(r, f"particle_{idx}_cy_px") for r in ok_rows], label=f"p{idx} y")
            ax.set_ylabel("px")

        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        self.canvas.draw()
        self._update_playback_cursor()

    def _update_playback_cursor(self) -> None:
        if self.axes is None or self.current_frame is None or not self.ok_rows:
            self.playback_cursor.hide()
            return
        row = min(self.ok_rows, key=lambda r: abs((self._num(r, "frame") or 0) - self.current_frame))
        time_s = self._num(row, "time_s")
        if time_s is None:
            self.playback_cursor.hide()
            return
        x_display = float(self.axes.transData.transform((time_s, 0.0))[0])
        bbox = self.axes.bbox
        if x_display < bbox.x0 or x_display > bbox.x1:
            self.playback_cursor.hide()
            return
        top = round(self.canvas.height() - bbox.y1)
        self.playback_cursor.setGeometry(round(x_display) - 1, top, 2, max(1, round(bbox.height)))
        self.playback_cursor.show()
        self.playback_cursor.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._update_playback_cursor)


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
        self.trap_center_worker: TrapCenterWorker | None = None
        self.calibration_image_paths: list[str] = []
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

        trap_center = QtWidgets.QGroupBox("Trap-center calibration")
        trap_center_layout = QtWidgets.QVBoxLayout(trap_center)
        calibration_image_row = QtWidgets.QHBoxLayout()
        self.calibration_images_edit = QtWidgets.QLineEdit()
        self.calibration_images_edit.setReadOnly(True)
        self.calibration_images_edit.setPlaceholderText("No single-particle images selected")
        self.select_calibration_images_btn = QtWidgets.QPushButton("Select images")
        self.select_calibration_images_btn.clicked.connect(self.browse_calibration_images)
        calibration_image_row.addWidget(self.calibration_images_edit, 1)
        calibration_image_row.addWidget(self.select_calibration_images_btn)
        trap_center_layout.addLayout(calibration_image_row)
        calibration_output_row = QtWidgets.QHBoxLayout()
        self.calibration_output_edit = QtWidgets.QLineEdit(str(DEFAULT_TRAP_CENTER_OUTPUT_ROOT / "trap_center_calibration"))
        self.calibration_output_btn = QtWidgets.QPushButton("Output")
        self.calibration_output_btn.clicked.connect(self.browse_calibration_output_dir)
        calibration_output_row.addWidget(self.calibration_output_edit, 1)
        calibration_output_row.addWidget(self.calibration_output_btn)
        trap_center_layout.addLayout(calibration_output_row)
        self.run_calibration_btn = QtWidgets.QPushButton("Detect images and average trap center")
        self.run_calibration_btn.clicked.connect(self.run_trap_center_calibration)
        trap_center_layout.addWidget(self.run_calibration_btn)
        form.addWidget(trap_center)

        self.trap_x = QtWidgets.QDoubleSpinBox()
        self.trap_x.setRange(-10001, 10000)
        self.trap_x.setSpecialValueText("Not set")
        self.trap_x.setDecimals(3)
        self.trap_y = QtWidgets.QDoubleSpinBox()
        self.trap_y.setRange(-10001, 10000)
        self.trap_y.setSpecialValueText("Not set")
        self.trap_y.setDecimals(3)
        self.pixels_per_mm = QtWidgets.QDoubleSpinBox()
        self.pixels_per_mm.setRange(0, 100000)
        self.pixels_per_mm.setSpecialValueText("Not set")
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
        self.fast_mode = QtWidgets.QCheckBox("Fast tracking (full CSV, lightweight preview)")
        self.fast_mode.setToolTip(
            "Detect every selected frame, but skip annotated-video/debug-frame output and update the GUI only at the preview interval."
        )
        self.preview_stride = QtWidgets.QSpinBox()
        self.preview_stride.setRange(1, 1000)
        self.preview_stride.setSuffix(" frames")

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
            ("Fast mode", self.fast_mode),
            ("Preview every", self.preview_stride),
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
        self.fast_mode.toggled.connect(self.on_fast_mode_toggled)
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
        self.trap_x.setValue(self.trap_x.minimum() if self.config.trap_x is None else self.config.trap_x)
        self.trap_y.setValue(self.trap_y.minimum() if self.config.trap_y is None else self.config.trap_y)
        self.pixels_per_mm.setValue(self.pixels_per_mm.minimum() if self.config.pixels_per_mm is None else self.config.pixels_per_mm)
        self.start_frame.setValue(self.config.start_frame)
        self.end_frame.setValue(0)
        self.particle_count.setValue(self.config.particle_count)
        self.conf.setValue(self.config.conf)
        self.mask_open.setValue(self.config.mask_open_px)
        self.mask_erode.setValue(self.config.mask_erode_px)
        self.radius_scale.setValue(self.config.circle_radius_scale)
        self.fast_mode.setChecked(self.config.fast_mode)
        self.preview_stride.setValue(self.config.preview_stride)
        self.on_fast_mode_toggled(self.config.fast_mode)
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
            self.output_edit.setText(str(DEFAULT_OUTPUT_ROOT / f"{Path(path).stem}_particle_tracking_app"))
            self.update_video_info()

    def browse_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YOLO model", str(Path(self.model_edit.text()).parent), "Model Files (*.pt);;All Files (*)")
        if path:
            self.model_edit.setText(path)

    def browse_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def browse_calibration_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select single-particle calibration images",
            str(RAW_DATA_ROOT),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not paths:
            return
        self.calibration_image_paths = paths
        self.calibration_images_edit.setText(f"{len(paths)} images selected")
        parents = {str(Path(path).parent) for path in paths}
        folder_name = Path(next(iter(parents))).name if len(parents) == 1 else "selected_images"
        self.calibration_output_edit.setText(
            str(DEFAULT_TRAP_CENTER_OUTPUT_ROOT / f"{folder_name}_trap_center_calibration")
        )
        self.log_msg(f"Selected {len(paths)} trap-center calibration images")

    def browse_calibration_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select parent directory for trap-center calibration output",
            str(Path(self.calibration_output_edit.text()).parent),
        )
        if path:
            self.calibration_output_edit.setText(str(Path(path) / "trap_center_calibration"))

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
            trap_x=None if self.trap_x.value() == self.trap_x.minimum() else self.trap_x.value(),
            trap_y=None if self.trap_y.value() == self.trap_y.minimum() else self.trap_y.value(),
            pixels_per_mm=None if self.pixels_per_mm.value() == self.pixels_per_mm.minimum() else self.pixels_per_mm.value(),
            start_frame=self.start_frame.value(),
            end_frame=None if end == 0 else end,
            frame_stride=1,
            conf=self.conf.value(),
            particle_count=self.particle_count.value(),
            mask_open_px=self.mask_open.value(),
            mask_erode_px=self.mask_erode.value(),
            circle_radius_scale=self.radius_scale.value(),
            fast_mode=self.fast_mode.isChecked(),
            preview_stride=self.preview_stride.value(),
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
        self.select_calibration_images_btn.setEnabled(not busy)
        self.calibration_output_btn.setEnabled(not busy)
        self.run_calibration_btn.setEnabled(not busy)
        self.fast_mode.setEnabled(not busy)
        self.preview_stride.setEnabled(not busy)

    def run_preview(self) -> None:
        if self.preview_worker is not None and self.preview_worker.isRunning():
            return
        config = self.make_config()
        try:
            config.validate_calibration()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Calibration required", str(error))
            return
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
        self.plot_panel.set_rows(list(self.rows_by_frame.values()))
        self.plot_panel.set_current_frame(self.current_frame)
        self.log_msg(f"Preview status: {row.get('status')}, candidates={row.get('num_candidates')}")
        self.set_busy(False)

    def run_batch(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        config = self.make_config()
        try:
            config.validate_calibration()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Calibration required", str(error))
            return
        self.progress.setValue(0)
        self.set_busy(True)
        mode = "fast CSV-first" if config.fast_mode else "full annotated"
        self.log_msg(f"Batch start ({mode}): {config.video_path}")
        self.worker = TrackingWorker(config)
        self.worker.progress.connect(self.on_progress)
        self.worker.frame_ready.connect(self.on_worker_frame)
        self.worker.finished_ok.connect(self.on_batch_done)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.start()

    def run_trap_center_calibration(self) -> None:
        if self.trap_center_worker is not None and self.trap_center_worker.isRunning():
            return
        if not self.calibration_image_paths:
            QtWidgets.QMessageBox.information(
                self, "Trap-center calibration", "Select one or more single-particle images first."
            )
            return
        output_dir = self.calibration_output_edit.text().strip()
        if not output_dir:
            QtWidgets.QMessageBox.information(
                self, "Trap-center calibration", "Choose an output directory for the derived calibration files."
            )
            return
        config = self.make_config()
        config.particle_count = 1
        self.progress.setValue(0)
        self.set_busy(True)
        self.log_msg(f"Trap-center calibration start: {len(self.calibration_image_paths)} images")
        self.trap_center_worker = TrapCenterWorker(
            self.calibration_image_paths, config, output_dir, self
        )
        self.trap_center_worker.progress.connect(self.on_progress)
        self.trap_center_worker.finished_ok.connect(self.on_trap_center_done)
        self.trap_center_worker.failed.connect(self.on_worker_failed)
        self.trap_center_worker.start()

    def stop_worker(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.log_msg("Stop requested")
        if self.trap_center_worker is not None and self.trap_center_worker.isRunning():
            self.trap_center_worker.stop()
            self.log_msg("Trap-center calibration stop requested")

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
        self.rows = rows
        self.rows_by_frame = {int(float(r["frame"])): r for r in rows if "frame" in r}
        self.plot_panel.set_rows(rows)
        self.plot_panel.set_current_frame(self.current_frame)
        self.log_msg(f"Batch done. Rows: {len(rows)}. Output: {self.make_config().resolved_output_dir()}")
        self.set_busy(False)

    def on_trap_center_done(self, result: dict) -> None:
        summary = result["summary"]
        valid_count = int(summary["valid_detection_count"])
        if summary.get("status") == "stopped":
            message = (
                f"Calibration stopped after {summary['processed_image_count']}/"
                f"{summary['input_image_count']} images. Partial results were saved but were not "
                f"copied into Trap X/Y.\n\nOutput: {result['output_dir']}"
            )
            self.log_msg(message.replace("\n", " "))
            QtWidgets.QMessageBox.warning(self, "Trap-center calibration stopped", message)
            self.set_busy(False)
            return
        if valid_count:
            trap_x = float(summary["trap_x_px"])
            trap_y = float(summary["trap_y_px"])
            self.trap_x.setValue(trap_x)
            self.trap_y.setValue(trap_y)
            preview_path = result.get("preview_path")
            if preview_path:
                preview = read_image(preview_path)
                if preview is not None:
                    self.last_overlay = preview
                    self.show_frame(preview)
            message = (
                f"Trap center = ({trap_x:.3f}, {trap_y:.3f}) px from "
                f"{valid_count}/{summary['processed_image_count']} processed images.\n\n"
                f"Output: {result['output_dir']}"
            )
            self.log_msg(message.replace("\n", " "))
            QtWidgets.QMessageBox.information(self, "Trap-center calibration complete", message)
        else:
            message = f"No valid particle detections. Output: {result['output_dir']}"
            self.log_msg(message)
            QtWidgets.QMessageBox.warning(self, "Trap-center calibration", message)
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
        self.trap_x.setValue(self.trap_x.minimum() if config.trap_x is None else config.trap_x)
        self.trap_y.setValue(self.trap_y.minimum() if config.trap_y is None else config.trap_y)
        self.pixels_per_mm.setValue(self.pixels_per_mm.minimum() if config.pixels_per_mm is None else config.pixels_per_mm)
        self.start_frame.setValue(config.start_frame)
        self.end_frame.setValue(0 if config.end_frame is None else config.end_frame)
        self.particle_count.setValue(getattr(config, "particle_count", 2))
        self.conf.setValue(config.conf)
        self.mask_open.setValue(config.mask_open_px)
        self.mask_erode.setValue(config.mask_erode_px)
        self.radius_scale.setValue(config.circle_radius_scale)
        self.fast_mode.setChecked(getattr(config, "fast_mode", False))
        self.preview_stride.setValue(getattr(config, "preview_stride", 10))
        self.save_video.setChecked(config.save_annotated_video)
        self.save_debug.setChecked(config.save_debug_frames)
        self.on_fast_mode_toggled(self.fast_mode.isChecked())
        self.update_video_info()

    def on_fast_mode_toggled(self, checked: bool) -> None:
        if not hasattr(self, "save_video"):
            return
        if checked:
            self.save_video.setChecked(False)
            self.save_debug.setChecked(False)
        self.save_video.setEnabled(not checked)
        self.save_debug.setEnabled(not checked)

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
        self.plot_panel.set_current_frame(self.current_frame, redraw=False)

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
        self.plot_panel.set_current_frame(value, redraw=False)
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
