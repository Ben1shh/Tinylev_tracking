"""Render the real Qt UI with in-memory synthetic demonstration data only."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis' / 'particle_tracking_app_v0_1'))

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from matplotlib import font_manager
from particle_tracking_app.manager import ManagedMainWindow
from particle_tracking_app.catalog import ExperimentCatalog, ExperimentRecord, VideoRecord
from particle_tracking_app.core import TrackingConfig, ParticleDetection, row_from_particles, draw_overlay

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
app.setStyle('Fusion')
# Windows' offscreen Qt platform has no system font discovery. Load a portable
# font explicitly so the actual widgets remain legible in the saved screenshot.
font_id = QtGui.QFontDatabase.addApplicationFont(font_manager.findfont('DejaVu Sans'))
app.setFont(QtGui.QFont(QtGui.QFontDatabase.applicationFontFamilies(font_id)[0], 9))
with patch('particle_tracking_app.app.dependency_report', return_value='Synthetic UI preview. No experiment files loaded.'):
    window = ManagedMainWindow()
window.resize(1800, 1050)
window.catalog = ExperimentCatalog(experiments=[ExperimentRecord(
    'demo', 'Synthetic demo (not experimental data)', videos=[VideoRecord('demo_pair', 'demo_pair.avi')]
)])
window.refresh_tree()
window.experiment_tree.expandAll()
window.video_edit.setText('demo_pair.avi (synthetic preview)')
window.model_edit.setText('models/best.pt')
window.output_edit.setText('outputs/demo_pair/')
window.calibration_output_edit.setText('outputs/trap_center_calibration/')
window.log.setPlainText('Synthetic demonstration only.\nEnter measured calibration before running tracking.\nFull manager: Tracking / Analysis A-D / Annotations.')
window.show()
app.processEvents()

frame = np.full((560, 860, 3), 32, dtype=np.uint8)
beads = [ParticleDetection(330, 280, 74, .98, 17203, .1, 0), ParticleDetection(497, 280, 93, .99, 27172, .1, 1)]
for bead in beads:
    cv2.circle(frame, (int(bead.cx), int(bead.cy)), int(bead.radius), (205, 205, 205), -1)
frame = draw_overlay(frame, [], beads, TrackingConfig(), layers={'trap': False, 'text': False})
cv2.putText(frame, 'SYNTHETIC DEMO - NOT EXPERIMENTAL DATA', (120, 58), cv2.FONT_HERSHEY_SIMPLEX, .65, (220,220,220), 1, cv2.LINE_AA)
window.show_frame(frame)
rows = []
# Explicit synthetic units for the demonstration plot, not experiment defaults.
config = TrackingConfig(trap_x=430, trap_y=280, pixels_per_mm=100)
for i in range(240):
    angle = i / 30 * 2 * np.pi
    cx, cy = 430 + 20*np.cos(angle), 280 + 12*np.sin(angle)
    pair = [ParticleDetection(cx-80, cy, 74,.98,17203,.1,0), ParticleDetection(cx+87,cy,93,.99,27172,.1,1)]
    rows.append(row_from_particles(i, 30, pair, pair, config))
window.plot_panel.set_rows(rows)
app.processEvents()
destination = ROOT / 'docs' / 'images' / 'experiment-manager-preview.png'
destination.parent.mkdir(parents=True, exist_ok=True)
assert window.grab().save(str(destination))
print(destination)
window.close()
