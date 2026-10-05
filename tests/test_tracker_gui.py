import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis/particle_tracking_app_v0_1'))
from PyQt5 import QtWidgets
from tinylev_tracker.app import MainWindow, PLOT_MODES
from particle_tracking_app.core import TrackingConfig, ParticleDetection, row_from_particles, read_config, write_config


class TrackerGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        with patch('particle_tracking_app.app.dependency_report', return_value='test'):
            self.window = MainWindow()

    def tearDown(self):
        self.window.close()

    def test_named_tracker_with_unset_calibration_and_all_plot_modes(self):
        self.assertEqual(self.window.windowTitle(), 'Tinylev Tracker v0.5')
        cfg = self.window.make_config()
        self.assertIsNone(cfg.trap_x)
        self.assertIsNone(cfg.trap_y)
        self.assertIsNone(cfg.pixels_per_mm)
        self.assertTrue(Path(cfg.model_path).is_file())
        self.assertTrue(cfg.fast_mode)
        self.assertFalse(cfg.save_annotated_video)
        self.assertFalse(cfg.save_debug_frames)
        self.assertEqual(self.window.plot_panel.series.count(), len(PLOT_MODES))
        self.assertTrue(self.window.annotation_group.isEnabled())

    def test_missing_calibration_stops_worker_and_null_roundtrip(self):
        with patch('particle_tracking_app.app.PreviewWorker') as worker, patch.object(QtWidgets.QMessageBox, 'warning') as warning:
            self.window.run_preview()
            worker.assert_not_called()
            warning.assert_called_once()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            write_config(path, self.window.make_config())
            self.window.apply_config(read_config(path))
            self.assertIsNone(self.window.make_config().trap_x)

    def test_explicit_zero_center_and_invalid_scale(self):
        self.window.trap_x.setValue(0)
        self.window.trap_y.setValue(0)
        self.window.pixels_per_mm.setValue(2)
        bead = ParticleDetection(3, 4, 2, .9, 12, 0, 0)
        cfg = self.window.make_config()
        cfg.particle_count = 1
        self.assertEqual(row_from_particles(0, 30, [bead], [bead], cfg)['cm_distance_from_trap_mm'], 2.5)
        for scale in [0, -1, float('nan'), float('inf')]:
            cfg.pixels_per_mm = scale
            with self.assertRaises(ValueError):
                cfg.validate_calibration()

    def test_selecting_new_video_clears_previous_calibration(self):
        self.window.trap_x.setValue(12)
        self.window.trap_y.setValue(23)
        self.window.pixels_per_mm.setValue(50)
        with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=('different.avi', '')), patch.object(self.window, 'update_video_info'):
            self.window.browse_video()
        cfg = self.window.make_config()
        self.assertIsNone(cfg.trap_x)
        self.assertIsNone(cfg.trap_y)
        self.assertIsNone(cfg.pixels_per_mm)
