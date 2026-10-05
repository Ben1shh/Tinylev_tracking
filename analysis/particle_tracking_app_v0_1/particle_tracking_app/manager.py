from __future__ import annotations

import json
import math
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from .analysis_service import (
    annotation_path,
    apply_manual_annotations,
    audit_video,
    create_run_dir,
    ensure_annotation_file,
    reclassify_preserved_derived,
    run_phase_ab,
    run_phase_c,
    run_phase_d_core,
    sha256,
)
from .app import MainWindow
from .catalog import (
    AnalysisRunRecord,
    ExperimentCatalog,
    ExperimentRecord,
    VideoRecord,
    load_catalog,
    save_catalog,
)
from .core import PROJECT_ROOT, DEFAULT_MODEL_PATH, read_tracking_csv


APP_DISPLAY_NAME = "Tinylev Experiment Manager v0.5"
ANNOTATION_STATES = ["static", "libration", "pair_axis_spinning_candidate", "pair_axis_spinning", "transition", "invalid"]


class AnalysisWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, experiment: ExperimentRecord, video: VideoRecord, parent=None):
        super().__init__(parent)
        self.experiment = experiment
        self.video = video

    def run(self) -> None:
        try:
            self.finished_ok.emit(run_phase_ab(self.experiment, self.video))
        except Exception:
            self.failed.emit(traceback.format_exc())


class PhaseCWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, derived: pd.DataFrame, experiment: ExperimentRecord, video: VideoRecord, start_frame: int, end_frame: int, output: Path, parent=None):
        super().__init__(parent)
        self.derived = derived
        self.experiment = experiment
        self.video = video
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.output = output

    def run(self) -> None:
        try:
            self.finished_ok.emit(
                run_phase_c(
                    self.derived,
                    self.experiment,
                    self.video,
                    self.start_frame,
                    self.end_frame,
                    self.output,
                )
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class PhaseDWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        derived: pd.DataFrame,
        experiment: ExperimentRecord,
        video: VideoRecord,
        start_frame: int,
        end_frame: int,
        output: Path,
        derived_source_path: Path | None,
        parent=None,
    ):
        super().__init__(parent)
        self.derived = derived
        self.experiment = experiment
        self.video = video
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.output = output
        self.derived_source_path = derived_source_path

    def run(self) -> None:
        try:
            self.finished_ok.emit(
                run_phase_d_core(
                    self.derived,
                    self.experiment,
                    self.video,
                    self.start_frame,
                    self.end_frame,
                    self.output,
                    self.derived_source_path,
                )
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class AnalysisPlotPanel(QtWidgets.QWidget):
    range_selected = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.derived = pd.DataFrame()
        self.axes = None
        self.current_frame: int | None = None
        self.span: SpanSelector | None = None
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Pair axis", "Pair separation", "Calibration geometry", "State overview"])
        self.mode.currentIndexChanged.connect(self.redraw)
        controls.addWidget(QtWidgets.QLabel("Analysis plot"))
        controls.addWidget(self.mode, 1)
        layout.addLayout(controls)
        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

    def set_derived(self, table: pd.DataFrame | None) -> None:
        self.derived = table.copy() if table is not None else pd.DataFrame()
        if "state" in self.derived:
            self.derived["state"] = self.derived["state"].replace({"spinning": "pair_axis_spinning"})
        self.redraw()

    def set_current_frame(self, frame: int) -> None:
        self.current_frame = int(frame)
        self.redraw_cursor()

    def redraw(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        self.axes = axis
        if self.derived.empty or "time_s" not in self.derived:
            axis.set_title("No Phase A-C data loaded")
            self.canvas.draw()
            return
        table = self.derived
        time = table["time_s"].to_numpy(float)
        mode = self.mode.currentText()
        if mode == "Pair axis":
            axis.plot(time, pd.to_numeric(table.get("psi_unwrapped_rad"), errors="coerce"), lw=0.8, label=r"$\psi$ unwrapped")
            axis.plot(time, pd.to_numeric(table.get("psi_dot"), errors="coerce"), lw=0.7, alpha=0.75, label=r"$\dot\psi$")
            axis.set_ylabel("rad or rad/s")
        elif mode == "Pair separation":
            separation = table.get("pair_separation_px")
            if separation is None and {"small_x_px", "small_y_px", "large_x_px", "large_y_px"}.issubset(table.columns):
                separation = ((table["large_x_px"] - table["small_x_px"]) ** 2 + (table["large_y_px"] - table["small_y_px"]) ** 2) ** 0.5
            axis.plot(time, pd.to_numeric(separation, errors="coerce"), lw=0.8, label="pair separation")
            axis.set_ylabel("px")
        elif mode == "Calibration geometry":
            for column, label in (("r_px", "r"), ("theta_rad", "theta"), ("phi_rad", "phi")):
                if column in table:
                    axis.plot(time, table[column], lw=0.75, label=label)
            axis.set_ylabel("px or rad")
        else:
            colors = {
                "static": "#4C78A8",
                "libration": "#F2CF5B",
                "pair_axis_spinning_candidate": "#F28E2B",
                "pair_axis_spinning": "#E45756",
                "reversal": "#B279A2",
                "transition": "#72B7B2",
                "irregular_candidate": "#FF9DA6",
                "invalid": "#777777",
            }
            if "state" in table:
                for state, group in table.groupby("state"):
                    axis.scatter(group["time_s"], group.get("psi_dot", 0), s=3, color=colors.get(str(state), "#aaaaaa"), label=str(state))
            axis.set_ylabel(r"$\dot\psi$ (rad/s)")
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
        self.span = SpanSelector(axis, self._on_span, "horizontal", useblit=True, props={"alpha": 0.22, "facecolor": "#E45756"})
        self.canvas.draw()
        self.redraw_cursor()

    def _on_span(self, minimum: float, maximum: float) -> None:
        if self.derived.empty:
            return
        lo, hi = sorted((minimum, maximum))
        selected = self.derived[self.derived["time_s"].between(lo, hi)]
        if selected.empty:
            return
        self.range_selected.emit(int(selected["frame"].iloc[0]), int(selected["frame"].iloc[-1]))

    def redraw_cursor(self) -> None:
        if self.axes is None:
            return
        for artist in list(self.axes.lines):
            if artist.get_gid() == "playback_cursor":
                artist.remove()
        if self.current_frame is not None and not self.derived.empty:
            nearest = (self.derived["frame"] - self.current_frame).abs().idxmin()
            time_s = float(self.derived.loc[nearest, "time_s"])
            line = self.axes.axvline(time_s, color="black", lw=1.0, alpha=0.75)
            line.set_gid("playback_cursor")
            self.canvas.draw_idle()


class ExperimentDialog(QtWidgets.QDialog):
    def __init__(self, record: ExperimentRecord | None = None, parent=None):
        super().__init__(parent)
        self.record = record
        self.setWindowTitle("Experiment record")
        form = QtWidgets.QFormLayout(self)
        self.experiment_id = QtWidgets.QLineEdit(record.experiment_id if record else "")
        self.experiment_id.setEnabled(record is None)
        self.display_name = QtWidgets.QLineEdit(record.display_name if record else "")
        self.date = QtWidgets.QLineEdit(record.experiment_date if record else "")
        self.date_status = QtWidgets.QComboBox(); self.date_status.addItems(["pending_confirmation", "confirmed"])
        self.raw_dir = QtWidgets.QLineEdit(record.raw_dir if record else "")
        raw_button = QtWidgets.QPushButton("Browse"); raw_button.clicked.connect(self._browse_raw)
        raw_row = QtWidgets.QHBoxLayout(); raw_row.addWidget(self.raw_dir); raw_row.addWidget(raw_button)
        self.trap_x = QtWidgets.QLineEdit("" if record is None or record.trap_x_px is None else str(record.trap_x_px))
        self.trap_y = QtWidgets.QLineEdit("" if record is None or record.trap_y_px is None else str(record.trap_y_px))
        self.trap_se_x = QtWidgets.QLineEdit("" if record is None or record.trap_se_x_px is None else str(record.trap_se_x_px))
        self.trap_se_y = QtWidgets.QLineEdit("" if record is None or record.trap_se_y_px is None else str(record.trap_se_y_px))
        self.pixels_per_mm = QtWidgets.QLineEdit("" if record is None or record.pixels_per_mm is None else str(record.pixels_per_mm))
        self.notes = QtWidgets.QPlainTextEdit(record.notes if record else "")
        if record:
            self.date_status.setCurrentText(record.date_status)
        form.addRow("Experiment ID", self.experiment_id)
        form.addRow("Display name", self.display_name)
        form.addRow("Experiment date", self.date)
        form.addRow("Date status", self.date_status)
        form.addRow("Raw directory", raw_row)
        form.addRow("Trap X (px)", self.trap_x)
        form.addRow("Trap Y (px)", self.trap_y)
        form.addRow("Trap X standard error (px)", self.trap_se_x)
        form.addRow("Trap Y standard error (px)", self.trap_se_y)
        form.addRow("Pixels/mm", self.pixels_per_mm)
        form.addRow("Notes", self.notes)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons)

    def _browse_raw(self) -> None:
        value = QtWidgets.QFileDialog.getExistingDirectory(self, "Raw data directory", self.raw_dir.text() or str(PROJECT_ROOT / "raw_data"))
        if value:
            self.raw_dir.setText(value)

    @staticmethod
    def _optional_float(text: str) -> float | None:
        return float(text) if text.strip() else None

    @staticmethod
    def _optional_error(text: str) -> float | None:
        value = ExperimentDialog._optional_float(text)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Center standard errors must be finite and nonnegative; leave unknown values blank.")
        return value

    def result_record(self) -> ExperimentRecord:
        old = self.record or ExperimentRecord("", "")
        return replace(
            old,
            experiment_id=self.experiment_id.text().strip(),
            display_name=self.display_name.text().strip(),
            experiment_date=self.date.text().strip(),
            date_status=self.date_status.currentText(),
            raw_dir=self.raw_dir.text().strip(),
            trap_x_px=self._optional_float(self.trap_x.text()),
            trap_y_px=self._optional_float(self.trap_y.text()),
            trap_se_x_px=self._optional_error(self.trap_se_x.text()),
            trap_se_y_px=self._optional_error(self.trap_se_y.text()),
            pixels_per_mm=self._optional_float(self.pixels_per_mm.text()),
            pixel_scale_status="confirmed" if self.pixels_per_mm.text().strip() else "pending_confirmation",
            notes=self.notes.toPlainText().strip(),
        )


class VideoDialog(QtWidgets.QDialog):
    def __init__(self, record: VideoRecord | None = None, parent=None):
        super().__init__(parent)
        self.record = record
        self.setWindowTitle("Video record")
        form = QtWidgets.QFormLayout(self)
        self.video_id = QtWidgets.QLineEdit(record.video_id if record else "")
        self.video_id.setEnabled(record is None)
        self.video_path = QtWidgets.QLineEdit(record.video_path if record else "")
        self.tracking = QtWidgets.QLineEdit(record.tracking_csv if record else "")
        self.model = QtWidgets.QLineEdit(record.model_path if record else str(DEFAULT_MODEL_PATH))
        self.fps = QtWidgets.QLineEdit("" if record is None or record.fps is None else str(record.fps))
        self.notes = QtWidgets.QPlainTextEdit(record.notes if record else "")
        for label, edit, mode in (
            ("Video", self.video_path, "video"), ("Tracking CSV", self.tracking, "csv"), ("Model", self.model, "model")
        ):
            button = QtWidgets.QPushButton("Browse")
            button.clicked.connect(lambda _checked=False, target=edit, kind=mode: self._browse(target, kind))
            row = QtWidgets.QHBoxLayout(); row.addWidget(edit); row.addWidget(button); form.addRow(label, row)
        form.insertRow(0, "Video ID", self.video_id)
        form.addRow("FPS", self.fps)
        form.addRow("Notes", self.notes)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons)

    def _browse(self, target: QtWidgets.QLineEdit, kind: str) -> None:
        filters = {"video": "AVI Files (*.avi);;All Files (*)", "csv": "CSV Files (*.csv);;All Files (*)", "model": "Model Files (*.pt);;All Files (*)"}
        value, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", target.text(), filters[kind])
        if value:
            target.setText(value)

    def result_record(self) -> VideoRecord:
        old = self.record or VideoRecord("", "")
        return replace(
            old,
            video_id=self.video_id.text().strip(),
            video_path=self.video_path.text().strip(),
            tracking_csv=self.tracking.text().strip(),
            model_path=self.model.text().strip(),
            fps=float(self.fps.text()) if self.fps.text().strip() else None,
            fps_source="user_confirmed" if self.fps.text().strip() else "pending_confirmation",
            notes=self.notes.toPlainText().strip(),
        )


class ManagedMainWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.catalog: ExperimentCatalog = load_catalog()
        self.current_experiment_id: str | None = None
        self.current_video_id: str | None = None
        self.current_derived = pd.DataFrame()
        self.current_derived_source_path: Path | None = None
        self.current_analysis_output: Path | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.phase_c_worker: PhaseCWorker | None = None
        self.phase_d_worker: PhaseDWorker | None = None
        self._wrap_management_ui()
        self._build_menu()
        self.refresh_tree()

    def _wrap_management_ui(self) -> None:
        tracking_page = self.takeCentralWidget()
        central = QtWidgets.QWidget(); root = QtWidgets.QHBoxLayout(central)
        self.experiment_tree = QtWidgets.QTreeWidget()
        self.experiment_tree.setHeaderLabels(["Experiments"])
        self.experiment_tree.setMinimumWidth(300)
        self.experiment_tree.itemSelectionChanged.connect(self.on_tree_selection)
        self.experiment_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.experiment_tree.customContextMenuRequested.connect(self.show_tree_context_menu)
        root.addWidget(self.experiment_tree, 0)
        self.detail_tabs = QtWidgets.QTabWidget()
        self.detail_tabs.addTab(tracking_page, "Tracking")
        self.analysis_page = self._make_analysis_page()
        self.annotation_page = self._make_annotation_page()
        self.detail_tabs.addTab(self.analysis_page, "Analysis A-D")
        self.detail_tabs.addTab(self.annotation_page, "Annotations")
        root.addWidget(self.detail_tabs, 1)
        self.setCentralWidget(central)
        self.frame_slider.valueChanged.connect(self.analysis_plot.set_current_frame)

    def _make_analysis_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(page)
        self.video_summary = QtWidgets.QLabel("Select a video from the experiment tree")
        self.video_summary.setWordWrap(True); layout.addWidget(self.video_summary)
        warning = QtWidgets.QLabel(
            "Phase C fits a phenomenological center. Phase D tests within-segment prediction without refitting validation data; "
            "neither phase alone qualifies a trapping center or Rlc."
        )
        warning.setWordWrap(True); warning.setStyleSheet("background:#fff4d6;border-left:5px solid #e39d16;padding:8px")
        layout.addWidget(warning)
        action_row = QtWidgets.QHBoxLayout()
        self.run_ab_btn = QtWidgets.QPushButton("Run Phase A-B")
        self.run_ab_btn.clicked.connect(self.run_phase_ab_clicked)
        self.load_preserved_btn = QtWidgets.QPushButton("Load preserved derived (read-only)")
        self.load_preserved_btn.clicked.connect(self.load_preserved_derived)
        action_row.addWidget(self.run_ab_btn); action_row.addWidget(self.load_preserved_btn); action_row.addStretch(1)
        layout.addLayout(action_row)
        self.audit_view = QtWidgets.QPlainTextEdit(); self.audit_view.setReadOnly(True); self.audit_view.setMaximumHeight(150)
        layout.addWidget(self.audit_view)
        self.analysis_plot = AnalysisPlotPanel(); self.analysis_plot.range_selected.connect(self.set_selected_range)
        layout.addWidget(self.analysis_plot, 1)
        phase_row = QtWidgets.QHBoxLayout()
        self.phase_start = QtWidgets.QSpinBox(); self.phase_start.setRange(0, 100000000)
        self.phase_end = QtWidgets.QSpinBox(); self.phase_end.setRange(0, 100000000)
        self.phase_c_btn = QtWidgets.QPushButton("Run Phase C on selected range")
        self.phase_c_btn.clicked.connect(self.run_phase_c_clicked)
        self.phase_d_btn = QtWidgets.QPushButton("Run Phase D core validation")
        self.phase_d_btn.clicked.connect(self.run_phase_d_clicked)
        phase_row.addWidget(QtWidgets.QLabel("Start frame")); phase_row.addWidget(self.phase_start)
        phase_row.addWidget(QtWidgets.QLabel("End frame")); phase_row.addWidget(self.phase_end)
        phase_row.addWidget(self.phase_c_btn); phase_row.addWidget(self.phase_d_btn); phase_row.addStretch(1); layout.addLayout(phase_row)
        result_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.phase_c_summary = QtWidgets.QPlainTextEdit(); self.phase_c_summary.setReadOnly(True)
        self.phase_c_image = QtWidgets.QLabel("No Phase C fit"); self.phase_c_image.setAlignment(QtCore.Qt.AlignCenter)
        result_split.addWidget(self.phase_c_summary); result_split.addWidget(self.phase_c_image); layout.addWidget(result_split)
        phase_d_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.phase_d_summary = QtWidgets.QPlainTextEdit(); self.phase_d_summary.setReadOnly(True)
        self.phase_d_image = QtWidgets.QLabel("No Phase D validation"); self.phase_d_image.setAlignment(QtCore.Qt.AlignCenter)
        phase_d_split.addWidget(self.phase_d_summary); phase_d_split.addWidget(self.phase_d_image); layout.addWidget(phase_d_split)
        return page

    def _make_annotation_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(page)
        self.annotation_table = QtWidgets.QTableWidget(0, 6)
        self.annotation_table.setHorizontalHeaderLabels(["Start", "End", "State", "Reviewer", "Notes", "Video"])
        self.annotation_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.annotation_table.itemSelectionChanged.connect(self.on_annotation_selected)
        layout.addWidget(self.annotation_table, 1)
        form = QtWidgets.QGridLayout()
        self.annotation_start = QtWidgets.QSpinBox(); self.annotation_start.setRange(0, 100000000)
        self.annotation_end = QtWidgets.QSpinBox(); self.annotation_end.setRange(0, 100000000)
        self.annotation_state = QtWidgets.QComboBox(); self.annotation_state.addItems(ANNOTATION_STATES)
        self.annotation_reviewer = QtWidgets.QLineEdit("Boyue")
        self.annotation_notes = QtWidgets.QLineEdit()
        for column, (label, widget) in enumerate((("Start", self.annotation_start), ("End", self.annotation_end), ("State", self.annotation_state), ("Reviewer", self.annotation_reviewer), ("Notes", self.annotation_notes))):
            form.addWidget(QtWidgets.QLabel(label), 0, column); form.addWidget(widget, 1, column)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add annotation"); add.clicked.connect(self.add_annotation)
        update = QtWidgets.QPushButton("Update selected"); update.clicked.connect(self.update_annotation)
        delete = QtWidgets.QPushButton("Delete selected"); delete.clicked.connect(self.delete_annotation)
        buttons.addWidget(add); buttons.addWidget(update); buttons.addWidget(delete); buttons.addStretch(1); layout.addLayout(buttons)
        return page

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Experiment")
        actions = [
            ("New experiment", self.new_experiment),
            ("Edit selected experiment", self.edit_experiment),
            ("Add video", self.add_video),
            ("Edit selected video", self.edit_video),
            ("Link tracking CSV", self.link_tracking),
            ("Link calibration output", self.link_calibration),
            ("Open selected file location", self.open_selected_file_location),
            ("Delete selected experiment", self.delete_selected_experiment),
        ]
        for label, callback in actions:
            action = menu.addAction(label); action.triggered.connect(callback)

    def selected_node_data(self) -> tuple[str, str, str] | None:
        item = self.experiment_tree.currentItem()
        if item is None:
            return None
        data = item.data(0, QtCore.Qt.UserRole)
        return tuple(data) if data else None

    def selected_location(self) -> Path | None:
        data = self.selected_node_data()
        if not data:
            return None
        kind, experiment_id, video_id = data
        experiment = self.catalog.experiment(experiment_id)
        if kind == "experiment":
            return Path(experiment.raw_dir) if experiment.raw_dir else None
        if kind == "calibration":
            if experiment.calibration_output:
                return Path(experiment.calibration_output)
            return Path(experiment.calibration_images[0]) if experiment.calibration_images else None
        video = self.catalog.video(experiment_id, video_id)
        if kind == "video":
            return Path(video.video_path) if video.video_path else None
        if kind == "tracking":
            return Path(video.tracking_csv) if video.tracking_csv else None
        if kind == "analysis":
            if video.analysis_runs:
                return Path(video.analysis_runs[-1].output_dir)
            if video.preserved_derived_tracks:
                return Path(video.preserved_derived_tracks)
            return None
        if kind == "annotations":
            return annotation_path(experiment_id, video_id)
        return None

    def open_selected_file_location(self) -> None:
        location = self.selected_location()
        if location is None:
            QtWidgets.QMessageBox.information(self, "Open file location", "No file or directory is linked to this item.")
            return
        location = location.resolve()
        existing = location if location.exists() else location.parent
        if not existing.exists():
            QtWidgets.QMessageBox.warning(self, "Open file location", f"Location does not exist:\n{location}")
            return
        if sys.platform == "win32" and location.is_file():
            subprocess.Popen(["explorer.exe", "/select,", str(location)])
        else:
            directory = existing if existing.is_dir() else existing.parent
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(directory)))

    def show_tree_context_menu(self, position: QtCore.QPoint) -> None:
        item = self.experiment_tree.itemAt(position)
        if item is None:
            return
        self.experiment_tree.setCurrentItem(item)
        data = self.selected_node_data()
        if not data:
            return
        kind, _, _ = data
        menu = QtWidgets.QMenu(self)
        open_action = menu.addAction("Open file location")
        open_action.triggered.connect(self.open_selected_file_location)
        if kind == "experiment":
            menu.addSeparator()
            edit_action = menu.addAction("Edit experiment")
            edit_action.triggered.connect(self.edit_experiment)
            add_video_action = menu.addAction("Add video")
            add_video_action.triggered.connect(self.add_video)
            menu.addSeparator()
            delete_action = menu.addAction("Delete experiment from catalog")
            delete_action.triggered.connect(self.delete_selected_experiment)
        elif kind in {"video", "tracking", "analysis", "annotations"}:
            menu.addSeparator()
            edit_video_action = menu.addAction("Edit video record")
            edit_video_action.triggered.connect(self.edit_video)
        menu.exec_(self.experiment_tree.viewport().mapToGlobal(position))

    def delete_selected_experiment(self) -> None:
        data = self.selected_node_data()
        experiment_id = data[1] if data else self.current_experiment_id
        if not experiment_id:
            QtWidgets.QMessageBox.information(self, "Delete experiment", "Select an experiment first.")
            return
        experiment = self.catalog.experiment(experiment_id)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete experiment from catalog",
            (
                f"Remove '{experiment.display_name}' from the experiment catalog?\n\n"
                "This only removes the catalog record. Raw data, tracking files, calibrations, "
                "annotations, and analysis outputs will NOT be deleted."
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.catalog.experiments = [record for record in self.catalog.experiments if record.experiment_id != experiment_id]
        save_catalog(self.catalog)
        if self.current_experiment_id == experiment_id:
            self.current_experiment_id = None
            self.current_video_id = None
            self.current_derived = pd.DataFrame()
            self.current_derived_source_path = None
            self.current_analysis_output = None
            self.analysis_plot.set_derived(None)
            self.video_summary.setText("Select an experiment or video from the experiment tree")
            self.audit_view.clear()
            self.annotation_table.setRowCount(0)
        self.refresh_tree()

    def refresh_tree(self) -> None:
        self.experiment_tree.blockSignals(True)
        try:
            self.experiment_tree.clear()
            for experiment in self.catalog.experiments:
                root = QtWidgets.QTreeWidgetItem([experiment.display_name or experiment.experiment_id])
                root.setData(0, QtCore.Qt.UserRole, ("experiment", experiment.experiment_id, ""))
                calibration = QtWidgets.QTreeWidgetItem(["Calibration"])
                calibration.setData(0, QtCore.Qt.UserRole, ("calibration", experiment.experiment_id, ""))
                root.addChild(calibration)
                videos = QtWidgets.QTreeWidgetItem([f"Videos ({len(experiment.videos)})"]); root.addChild(videos)
                for video in experiment.videos:
                    raw_status = "raw ready" if Path(video.video_path).is_file() else "raw missing"
                    tracking_path = Path(video.tracking_csv) if video.tracking_csv else None
                    if not tracking_path or not tracking_path.is_file():
                        tracking_status = "tracking missing"
                    elif video.expected_tracking_sha256 and sha256(tracking_path) != video.expected_tracking_sha256.upper():
                        tracking_status = "tracking blocked"
                    else:
                        tracking_status = "tracking ready"
                    item = QtWidgets.QTreeWidgetItem([f"{video.video_id} [{raw_status}; {tracking_status}]"])
                    item.setData(0, QtCore.Qt.UserRole, ("video", experiment.experiment_id, video.video_id))
                    videos.addChild(item)
                    analysis_status = "run available" if video.analysis_runs else ("preserved read-only" if video.preserved_derived_tracks and Path(video.preserved_derived_tracks).is_file() else "not run")
                    annotations = annotation_path(experiment.experiment_id, video.video_id)
                    annotation_count = len(pd.read_csv(annotations)) if annotations.is_file() else 0
                    for kind, label in (("tracking", f"Tracking [{tracking_status}]"), ("analysis", f"Analysis A-D [{analysis_status}]"), ("annotations", f"Annotations [{annotation_count}]")):
                        child = QtWidgets.QTreeWidgetItem([label]); child.setData(0, QtCore.Qt.UserRole, (kind, experiment.experiment_id, video.video_id)); item.addChild(child)
                self.experiment_tree.addTopLevelItem(root); root.setExpanded(True); videos.setExpanded(True)
        finally:
            self.experiment_tree.blockSignals(False)

    def on_tree_selection(self) -> None:
        item = self.experiment_tree.currentItem()
        if item is None:
            return
        data = item.data(0, QtCore.Qt.UserRole)
        if not data:
            return
        kind, experiment_id, video_id = data
        try:
            experiment = self.catalog.experiment(experiment_id)
        except KeyError:
            return
        self.current_experiment_id = experiment_id
        if kind == "experiment":
            self.video_summary.setText(f"{experiment.display_name}\nDate: {experiment.experiment_date} [{experiment.date_status}]\nRaw: {experiment.raw_dir}\n{experiment.notes}")
            return
        if kind == "calibration":
            self.calibration_image_paths = list(experiment.calibration_images)
            self.calibration_images_edit.setText(f"{len(experiment.calibration_images)} catalog images")
            self.calibration_output_edit.setText(experiment.calibration_output)
            self.trap_x.setValue(self.trap_x.minimum() if experiment.trap_x_px is None else experiment.trap_x_px)
            self.trap_y.setValue(self.trap_y.minimum() if experiment.trap_y_px is None else experiment.trap_y_px)
            self.pixels_per_mm.setValue(self.pixels_per_mm.minimum() if experiment.pixels_per_mm is None else experiment.pixels_per_mm)
            self.detail_tabs.setCurrentIndex(0)
            return
        self.current_video_id = video_id
        self.load_video_record(experiment_id, video_id)
        if kind == "analysis": self.detail_tabs.setCurrentIndex(1)
        elif kind == "annotations": self.detail_tabs.setCurrentIndex(2)
        else: self.detail_tabs.setCurrentIndex(0)

    def current_records(self) -> tuple[ExperimentRecord, VideoRecord]:
        if not self.current_experiment_id or not self.current_video_id:
            raise ValueError("Select a video first")
        return self.catalog.experiment(self.current_experiment_id), self.catalog.video(self.current_experiment_id, self.current_video_id)

    def load_video_record(self, experiment_id: str, video_id: str) -> None:
        experiment, video = self.catalog.experiment(experiment_id), self.catalog.video(experiment_id, video_id)
        self.video_edit.setText(video.video_path); self.model_edit.setText(video.model_path)
        self.output_edit.setText(str(PROJECT_ROOT / "outputs" / "particle_tracking" / f"{video.video_id}_particle_tracking_app"))
        self.trap_x.setValue(self.trap_x.minimum() if experiment.trap_x_px is None else experiment.trap_x_px)
        self.trap_y.setValue(self.trap_y.minimum() if experiment.trap_y_px is None else experiment.trap_y_px)
        self.pixels_per_mm.setValue(self.pixels_per_mm.minimum() if experiment.pixels_per_mm is None else experiment.pixels_per_mm)
        self.update_video_info()
        if video.tracking_csv and Path(video.tracking_csv).is_file():
            rows = read_tracking_csv(video.tracking_csv)
            self.rows = rows; self.rows_by_frame = {int(float(row["frame"])): row for row in rows if "frame" in row}
            self.plot_panel.set_rows(rows)
        self.video_summary.setText(f"{experiment.display_name} / {video.video_id}\nRaw: {video.video_path}\nTracking: {video.tracking_csv}\n{video.notes}")
        self.audit_view.setPlainText("Run Phase A-B to perform current integrity and provenance checks.")
        self.run_ab_btn.setEnabled(True)
        self.load_preserved_btn.setEnabled(bool(video.preserved_derived_tracks and Path(video.preserved_derived_tracks).is_file()))
        self.current_derived = pd.DataFrame(); self.current_derived_source_path = None; self.current_analysis_output = None
        self.analysis_plot.set_derived(None); self.phase_c_summary.clear(); self.phase_c_image.clear(); self.phase_d_summary.clear(); self.phase_d_image.clear()
        self.refresh_annotations()

    def run_phase_ab_clicked(self) -> None:
        try:
            experiment, video = self.current_records()
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Phase A-B", str(error)); return
        self.run_ab_btn.setEnabled(False); self.audit_view.setPlainText("Running Phase A integrity audit and Phase B candidate analysis...")
        self.analysis_worker = AnalysisWorker(experiment, video, self)
        self.analysis_worker.finished_ok.connect(self.on_phase_ab_complete)
        self.analysis_worker.failed.connect(self.on_analysis_failed)
        self.analysis_worker.start()

    def on_phase_ab_complete(self, result: dict) -> None:
        experiment, video = self.current_records()
        video.analysis_runs.append(AnalysisRunRecord(result["run_id"], result["output_dir"], pd.Timestamp.now().isoformat(), input_tracking_sha256=result["tracking_sha256"]))
        save_catalog(self.catalog)
        self.current_analysis_output = Path(result["output_dir"])
        self.current_derived = result["derived"]
        self.current_derived_source_path = self.current_analysis_output / "derived_tracks.csv"
        self.analysis_plot.set_derived(self.current_derived)
        self.audit_view.setPlainText(json.dumps(result["audit"], indent=2, ensure_ascii=False))
        self.run_ab_btn.setEnabled(True); self.refresh_tree(); self.refresh_annotations()

    def on_analysis_failed(self, text: str) -> None:
        self.run_ab_btn.setEnabled(True); self.phase_c_btn.setEnabled(True); self.phase_d_btn.setEnabled(True)
        self.audit_view.setPlainText(text)
        QtWidgets.QMessageBox.critical(self, "Analysis error", text)

    def load_preserved_derived(self) -> None:
        try:
            experiment, video = self.current_records(); path = Path(video.preserved_derived_tracks)
            preserved = pd.read_csv(path)
            self.current_derived = reclassify_preserved_derived(preserved, experiment, video)
            self.current_derived_source_path = path
            self.current_analysis_output = None
            self.analysis_plot.set_derived(self.current_derived)
            self.audit_view.setPlainText(
                f"READ-ONLY preserved derived table:\n{path}\n"
                "States were recomputed in memory with the current rules; the preserved table and upstream tracking CSV were not changed."
            )
        except Exception:
            self.on_analysis_failed(traceback.format_exc())

    def set_selected_range(self, start: int, end: int) -> None:
        self.phase_start.setValue(start); self.phase_end.setValue(end)
        self.annotation_start.setValue(start); self.annotation_end.setValue(end)

    def run_phase_c_clicked(self) -> None:
        if self.current_derived.empty:
            QtWidgets.QMessageBox.warning(self, "Phase C", "Load or run Phase A-B data first."); return
        try:
            experiment, video = self.current_records()
            if self.current_analysis_output is None or not self.current_analysis_output.exists():
                run_id, self.current_analysis_output = create_run_dir(experiment.experiment_id, video.video_id)
                video.analysis_runs.append(AnalysisRunRecord(run_id, str(self.current_analysis_output), pd.Timestamp.now().isoformat(), status="phase_c_from_preserved_derived"))
                save_catalog(self.catalog)
            self.phase_c_btn.setEnabled(False)
            self.phase_c_worker = PhaseCWorker(self.current_derived, experiment, video, self.phase_start.value(), self.phase_end.value(), self.current_analysis_output, self)
            self.phase_c_worker.finished_ok.connect(self.on_phase_c_complete); self.phase_c_worker.failed.connect(self.on_analysis_failed); self.phase_c_worker.start()
        except Exception:
            self.on_analysis_failed(traceback.format_exc())

    def on_phase_c_complete(self, result: dict) -> None:
        self.phase_c_btn.setEnabled(True)
        self.phase_c_summary.setPlainText(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        image = Path(result["output_dir"]) / "phase_c_geometry.png"
        pixmap = QtGui.QPixmap(str(image))
        self.phase_c_image.setPixmap(pixmap.scaled(650, 450, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def run_phase_d_clicked(self) -> None:
        if self.current_derived.empty:
            QtWidgets.QMessageBox.warning(self, "Phase D", "Load or run Phase A-B data first."); return
        try:
            experiment, video = self.current_records()
            if self.current_analysis_output is None or not self.current_analysis_output.exists():
                run_id, self.current_analysis_output = create_run_dir(experiment.experiment_id, video.video_id)
                source_status = "phase_d_from_preserved_derived" if self.current_derived_source_path == Path(video.preserved_derived_tracks) else "phase_d_from_current_derived"
                video.analysis_runs.append(
                    AnalysisRunRecord(run_id, str(self.current_analysis_output), pd.Timestamp.now().isoformat(), status=source_status)
                )
                save_catalog(self.catalog)
            self.phase_d_btn.setEnabled(False)
            self.phase_d_worker = PhaseDWorker(
                self.current_derived,
                experiment,
                video,
                self.phase_start.value(),
                self.phase_end.value(),
                self.current_analysis_output,
                self.current_derived_source_path,
                self,
            )
            self.phase_d_worker.finished_ok.connect(self.on_phase_d_complete)
            self.phase_d_worker.failed.connect(self.on_analysis_failed)
            self.phase_d_worker.start()
        except Exception:
            self.on_analysis_failed(traceback.format_exc())

    def on_phase_d_complete(self, result: dict) -> None:
        self.phase_d_btn.setEnabled(True)
        self.phase_d_summary.setPlainText(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        image = Path(result["output_dir"]) / "phase_d_validation.png"
        pixmap = QtGui.QPixmap(str(image))
        self.phase_d_image.setPixmap(pixmap.scaled(650, 450, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def refresh_annotations(self) -> None:
        self.annotation_table.setRowCount(0)
        if not self.current_experiment_id or not self.current_video_id:
            return
        path = annotation_path(self.current_experiment_id, self.current_video_id); ensure_annotation_file(path)
        table = pd.read_csv(path)
        for _, row in table.iterrows():
            index = self.annotation_table.rowCount(); self.annotation_table.insertRow(index)
            values = [row["start_frame"], row["end_frame"], "pair_axis_spinning" if row["override_state"] == "spinning" else row["override_state"], row.get("reviewer", ""), row.get("notes", ""), row.get("video_id", self.current_video_id)]
            for column, value in enumerate(values): self.annotation_table.setItem(index, column, QtWidgets.QTableWidgetItem(str(value)))
        self.annotation_table.resizeColumnsToContents()

    def annotation_record(self) -> dict:
        if self.annotation_end.value() < self.annotation_start.value():
            raise ValueError("End frame must be greater than or equal to start frame")
        return {"video_id": self.current_video_id, "start_frame": self.annotation_start.value(), "end_frame": self.annotation_end.value(), "override_state": self.annotation_state.currentText(), "reviewer": self.annotation_reviewer.text().strip(), "notes": self.annotation_notes.text().strip()}

    def _write_annotations(self, table: pd.DataFrame) -> None:
        path = annotation_path(self.current_experiment_id or "", self.current_video_id or "")
        path.parent.mkdir(parents=True, exist_ok=True); table.to_csv(path, index=False)
        self.refresh_annotations()
        if not self.current_derived.empty:
            base = self.current_derived.copy()
            if "auto_state" in base:
                base["state"] = base["auto_state"]
            self.current_derived = apply_manual_annotations(base, path); self.analysis_plot.set_derived(self.current_derived)

    def add_annotation(self) -> None:
        try:
            path = annotation_path(self.current_experiment_id or "", self.current_video_id or ""); ensure_annotation_file(path)
            table = pd.read_csv(path); table = pd.concat([table, pd.DataFrame([self.annotation_record()])], ignore_index=True); self._write_annotations(table)
        except Exception as error: QtWidgets.QMessageBox.warning(self, "Annotation", str(error))

    def update_annotation(self) -> None:
        row = self.annotation_table.currentRow()
        if row < 0: return
        path = annotation_path(self.current_experiment_id or "", self.current_video_id or ""); table = pd.read_csv(path); record = self.annotation_record()
        for key, value in record.items(): table.loc[row, key] = value
        self._write_annotations(table)

    def delete_annotation(self) -> None:
        row = self.annotation_table.currentRow()
        if row < 0: return
        path = annotation_path(self.current_experiment_id or "", self.current_video_id or ""); table = pd.read_csv(path).drop(index=row).reset_index(drop=True); self._write_annotations(table)

    def on_annotation_selected(self) -> None:
        row = self.annotation_table.currentRow()
        if row < 0: return
        self.annotation_start.setValue(int(float(self.annotation_table.item(row, 0).text())))
        self.annotation_end.setValue(int(float(self.annotation_table.item(row, 1).text())))
        self.annotation_state.setCurrentText(self.annotation_table.item(row, 2).text())
        self.annotation_reviewer.setText(self.annotation_table.item(row, 3).text())
        self.annotation_notes.setText(self.annotation_table.item(row, 4).text())

    def new_experiment(self) -> None:
        dialog = ExperimentDialog(parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            try: self.catalog.experiments.append(dialog.result_record()); save_catalog(self.catalog); self.refresh_tree()
            except Exception as error: QtWidgets.QMessageBox.warning(self, "Experiment", str(error))

    def edit_experiment(self) -> None:
        if not self.current_experiment_id: return
        record = self.catalog.experiment(self.current_experiment_id); dialog = ExperimentDialog(record, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            try:
                updated = dialog.result_record()
                self.catalog.experiments[self.catalog.experiments.index(record)] = updated
                save_catalog(self.catalog)
                self.refresh_tree()
            except Exception as error:
                QtWidgets.QMessageBox.warning(self, "Experiment", str(error))

    def add_video(self) -> None:
        if not self.current_experiment_id: return
        dialog = VideoDialog(parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            try: self.catalog.experiment(self.current_experiment_id).videos.append(dialog.result_record()); save_catalog(self.catalog); self.refresh_tree()
            except Exception as error: QtWidgets.QMessageBox.warning(self, "Video", str(error))

    def edit_video(self) -> None:
        try: experiment, record = self.current_records()
        except Exception: return
        dialog = VideoDialog(record, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            experiment.videos[experiment.videos.index(record)] = dialog.result_record(); save_catalog(self.catalog); self.refresh_tree()

    def link_tracking(self) -> None:
        try: _, video = self.current_records()
        except Exception: return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Tracking CSV", video.tracking_csv, "CSV Files (*.csv)")
        if path: video.tracking_csv = path; video.expected_tracking_sha256 = ""; save_catalog(self.catalog); self.load_video_record(self.current_experiment_id or "", self.current_video_id or "")

    def link_calibration(self) -> None:
        if not self.current_experiment_id: return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Calibration output", str(PROJECT_ROOT / "outputs" / "trap_center_calibration"))
        if not directory: return
        summary_path = Path(directory) / "trap_center_summary.csv"
        if not summary_path.is_file(): QtWidgets.QMessageBox.warning(self, "Calibration", "trap_center_summary.csv not found"); return
        summary = pd.read_csv(summary_path).iloc[0]; experiment = self.catalog.experiment(self.current_experiment_id)
        experiment.calibration_output = directory; experiment.trap_x_px = float(summary["trap_x_px"]); experiment.trap_y_px = float(summary["trap_y_px"])
        experiment.trap_se_x_px = float(summary.get("standard_error_x_px", summary.get("se_x_px", summary.get("trap_se_x_px", float("nan")))))
        experiment.trap_se_y_px = float(summary.get("standard_error_y_px", summary.get("se_y_px", summary.get("trap_se_y_px", float("nan")))))
        experiment.trap_center_source = "linked_calibration_output_user_confirmation_pending"; save_catalog(self.catalog); self.refresh_tree()


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    window = ManagedMainWindow(); window.show()
    return app.exec_()
