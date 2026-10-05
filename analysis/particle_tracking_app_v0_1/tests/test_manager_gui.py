from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from particle_tracking_app.manager import ManagedMainWindow, PhaseCWorker, PhaseDWorker  # noqa: E402
from particle_tracking_app.catalog import ExperimentCatalog, ExperimentRecord, VideoRecord  # noqa: E402


class ManagerGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_window(self) -> ManagedMainWindow:
        # The production launcher imports torch before Qt. GUI structure tests
        # do not need to reload GPU DLLs and mock the dependency report to keep
        # isolated offscreen test processes deterministic on Windows.
        with patch("particle_tracking_app.app.dependency_report", return_value="test dependencies"):
            return ManagedMainWindow()

    def test_manager_starts_empty_and_has_three_detail_pages(self) -> None:
        window = self.make_window()
        try:
            self.assertEqual(window.detail_tabs.count(), 3)
            self.assertEqual(window.detail_tabs.tabText(1), "Analysis A-D")
            self.assertEqual(window.phase_d_btn.text(), "Run Phase D core validation")
            self.assertEqual(window.experiment_tree.topLevelItemCount(), 0)
        finally:
            window.close()

    def test_delete_experiment_removes_catalog_record_only_after_confirmation(self) -> None:
        window = self.make_window()
        try:
            window.catalog = ExperimentCatalog(experiments=[ExperimentRecord("temporary", "Temporary")])
            window.refresh_tree()
            window.experiment_tree.setCurrentItem(window.experiment_tree.topLevelItem(0))
            with (
                patch("particle_tracking_app.manager.save_catalog") as save,
                patch("particle_tracking_app.manager.QtWidgets.QMessageBox.question", return_value=QtWidgets.QMessageBox.Yes),
            ):
                window.delete_selected_experiment()
            self.assertEqual(window.catalog.experiments, [])
            save.assert_called_once()
        finally:
            window.close()

    def test_phase_c_worker_does_not_shadow_qthread_start_method(self) -> None:
        worker = PhaseCWorker(
            pd.DataFrame(),
            ExperimentRecord("test", "Test"),
            VideoRecord("video", "video.avi"),
            10,
            20,
            Path("output"),
        )
        self.assertTrue(callable(worker.start))
        self.assertEqual(worker.start_frame, 10)
        self.assertEqual(worker.end_frame, 20)

    def test_phase_d_worker_does_not_shadow_qthread_start_method(self) -> None:
        worker = PhaseDWorker(
            pd.DataFrame(),
            ExperimentRecord("test", "Test"),
            VideoRecord("video", "video.avi"),
            10,
            20,
            Path("output"),
            Path("derived.csv"),
        )
        self.assertTrue(callable(worker.start))
        self.assertEqual(worker.start_frame, 10)
        self.assertEqual(worker.end_frame, 20)
        self.assertEqual(worker.derived_source_path, Path("derived.csv"))


if __name__ == "__main__":
    unittest.main()
