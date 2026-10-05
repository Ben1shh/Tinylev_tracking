import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PyQt5 import QtWidgets
from particle_tracking_app.core import TrackingConfig, ParticleDetection, row_from_particles, read_config, write_config, draw_overlay
from particle_tracking_app.manager import ManagedMainWindow, ExperimentDialog, VideoDialog
from particle_tracking_app.catalog import ExperimentCatalog, ExperimentRecord, VideoRecord
from particle_tracking_app.analysis_service import make_analysis_config


class CalibrationRequiredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_unset_calibration_cannot_generate_physical_measurements(self):
        bead = ParticleDetection(10, 20, 2, .9, 12, 0, 0)
        with self.assertRaisesRegex(ValueError, 'calibration|Calibration'):
            row_from_particles(0, 30, [bead], [bead], TrackingConfig(particle_count=1))

    def test_explicit_zero_center_is_valid_and_conversion_is_correct(self):
        bead = ParticleDetection(3, 4, 2, .9, 12, 0, 0)
        row = row_from_particles(0, 30, [bead], [bead], TrackingConfig(particle_count=1, trap_x=0, trap_y=0, pixels_per_mm=2))
        self.assertEqual(row['cm_distance_from_trap_px'], 5)
        self.assertEqual(row['cm_distance_from_trap_mm'], 2.5)

    def test_missing_calibration_roundtrips_as_json_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            write_config(path, TrackingConfig())
            loaded = read_config(path)
            self.assertIsNone(loaded.trap_x)
            self.assertIsNone(loaded.trap_y)
            self.assertIsNone(loaded.pixels_per_mm)

    def test_uncalibrated_overlay_does_not_draw_a_trap(self):
        frame = np.zeros((48, 64, 3), np.uint8)
        config = TrackingConfig(trap_x=None, trap_y=None, pixels_per_mm=None)
        output = draw_overlay(frame, [], [], config, layers={'text': False, 'trap': True})
        np.testing.assert_array_equal(output, frame)

    def test_fresh_manager_is_unset_and_switching_experiments_clears_values(self):
        with patch('particle_tracking_app.app.dependency_report', return_value='test'):
            window = ManagedMainWindow()
        try:
            cfg = window.make_config()
            self.assertIsNone(cfg.trap_x)
            self.assertIsNone(cfg.trap_y)
            self.assertIsNone(cfg.pixels_per_mm)
            window.trap_x.setValue(10)
            window.trap_y.setValue(20)
            window.pixels_per_mm.setValue(30)
            window.catalog = ExperimentCatalog(experiments=[ExperimentRecord('new', 'New', videos=[VideoRecord('video', '')])])
            window.load_video_record('new', 'video')
            cfg = window.make_config()
            self.assertIsNone(cfg.trap_x)
            self.assertIsNone(cfg.trap_y)
            self.assertIsNone(cfg.pixels_per_mm)
        finally:
            window.close()

    def test_invalid_scale_is_rejected_before_measurement(self):
        bead = ParticleDetection(3, 4, 2, .9, 12, 0, 0)
        for scale in [0, -1, float('nan'), float('inf')]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                row_from_particles(0, 30, [bead], [bead], TrackingConfig(trap_x=0, trap_y=0, pixels_per_mm=scale))

    def test_analysis_does_not_treat_unknown_center_error_as_zero(self):
        record = ExperimentRecord('test', 'Test', trap_x_px=0, trap_y_px=0)
        with self.assertRaises(ValueError):
            make_analysis_config(record)

    def test_explicit_analysis_calibration_accepts_zero_uncertainty(self):
        record = ExperimentRecord('test', 'Test', trap_x_px=0, trap_y_px=0, trap_se_x_px=0, trap_se_y_px=0)
        config = make_analysis_config(record)
        self.assertEqual(config.trap_se_x_px, 0)

    def test_new_video_uses_bundled_model(self):
        dialog = VideoDialog()
        self.assertTrue(Path(dialog.result_record().model_path).is_file())
        dialog.close()

    def test_experiment_dialog_preserves_measured_uncertainty_and_unknowns(self):
        dialog = ExperimentDialog()
        self.assertIsNone(dialog.result_record().trap_se_x_px)
        dialog.trap_se_x.setText('0')
        dialog.trap_se_y.setText('0.25')
        record = dialog.result_record()
        self.assertEqual(record.trap_se_x_px, 0)
        self.assertEqual(record.trap_se_y_px, .25)
        reopened = ExperimentDialog(record)
        self.assertEqual(reopened.result_record().trap_se_y_px, .25)
        for invalid in ['-1', 'nan', 'inf']:
            dialog.trap_se_x.setText(invalid)
            with self.assertRaises(ValueError):
                dialog.result_record()
        reopened.close()
        dialog.close()

    def test_edit_invalid_error_warns_without_replacing_record(self):
        with patch('particle_tracking_app.app.dependency_report', return_value='test'):
            window = ManagedMainWindow()
        record = ExperimentRecord('test', 'Test')
        window.catalog = ExperimentCatalog(experiments=[record])
        window.current_experiment_id = 'test'
        try:
            with patch('particle_tracking_app.manager.ExperimentDialog') as dialog, patch('particle_tracking_app.manager.save_catalog') as save, patch.object(QtWidgets.QMessageBox, 'warning') as warning:
                dialog.return_value.exec_.return_value = QtWidgets.QDialog.Accepted
                dialog.return_value.result_record.side_effect = ValueError('invalid error')
                window.edit_experiment()
                warning.assert_called_once()
                save.assert_not_called()
                self.assertIs(window.catalog.experiments[0], record)
        finally:
            window.close()
