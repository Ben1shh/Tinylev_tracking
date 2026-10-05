"""Capture the real Tracker UI from a supplied video and matching tracking CSV.

This reads inputs and writes screenshots/provenance only. It does not run new
tracking, load historical calibration placeholders or modify input data.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis/particle_tracking_app_v0_1'))

import numpy as np
from matplotlib import font_manager
from PyQt5 import QtWidgets, QtGui
from tinylev_tracker.app import MainWindow
from particle_tracking_app.core import read_tracking_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--tracking', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=1000)
    args = parser.parse_args()
    video, tracking = args.video.resolve(), args.tracking.resolve()
    if not video.is_file() or not tracking.is_file():
        parser.error('Both input files must exist.')
    source_hash = hashlib.sha256(tracking.read_bytes()).hexdigest()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle('Fusion')
    font_id = QtGui.QFontDatabase.addApplicationFont(font_manager.findfont('DejaVu Sans'))
    app.setFont(QtGui.QFont(QtGui.QFontDatabase.applicationFontFamilies(font_id)[0], 9))
    window = MainWindow()
    window.resize(1600, 1050)
    window.video_edit.setText(str(video))
    window.update_video_info()
    if not window.video_info or not 0 <= args.frame < window.video_info['frames']:
        raise ValueError('Video/frame could not be opened.')
    rows = read_tracking_csv(tracking)
    window.rows = rows
    window.rows_by_frame = {int(float(row['frame'])): row for row in rows}
    window.loaded_tracking_path = tracking
    window.plot_panel.set_rows(rows)
    window.roi_x_max.setValue(window.video_info['width'])
    window.roi_y_max.setValue(window.video_info['height'])
    window.show()
    app.processEvents()
    window.frame_slider.setValue(args.frame)
    window.refresh_current_frame()
    window.plot_panel.series.setCurrentText('d(t)')
    # Equivalent to zooming the plot to the first 10 video seconds.
    window.plot_panel.axes.set_xlim(0, 10)
    kinematics = window.plot_panel.kinematics
    visible = kinematics['d_px'][(kinematics['time_s'] >= 0) & (kinematics['time_s'] <= 10)]
    visible = visible[np.isfinite(visible)]
    if len(visible):
        padding = max(1, .08 * float(np.ptp(visible)))
        window.plot_panel.axes.set_ylim(float(visible.min()) - padding, float(visible.max()) + padding)
    window.plot_panel.canvas.draw()
    window.plot_panel.set_current_frame(args.frame)
    # Keep the decoded frame and cached plot; hide workstation directories in UI.
    window.video_edit.setText(video.name)
    window.model_edit.setText('models/best.pt')
    window.output_edit.setText('outputs/' + video.stem)
    window.log.setPlainText(f'Loaded recording: {video.name}\nLoaded tracking CSV: {len(rows)} rows.\nExisting pixel detections; no calibration assumed.')
    app.processEvents()
    destination = ROOT / 'docs/images/tracker-preview.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert window.grab().save(str(destination))
    window.plot_panel.series.setCurrentText('psi + Omega spectrum')
    window.findChild(QtWidgets.QScrollArea).ensureWidgetVisible(window.annotation_group)
    app.processEvents()
    assert window.grab().save(str(destination.with_name('tracker-annotations-preview.png')))
    metadata = {
        'kind': 'actual Qt application screenshots with recorded experimental video and existing tracking CSV',
        'video_basename': video.name,
        'tracking_sha256': source_hash,
        'frame': args.frame,
        'video_fps': window.video_info['fps'],
        'tracking_rows': len(rows),
        'time_series_visible_seconds': [0, 10],
        'calibration': 'unset; pixel detections and center-independent plots only',
        'display_changes': 'workstation directories replaced with basenames; time-series plot zoomed to 0-10 s',
        'data_changes': 'none; CSV read without relabeling, smoothing or synthetic replacement; plotting uses existing app methods',
        'raw_inputs_in_repository': False,
    }
    (ROOT / 'docs/preview-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    assert hashlib.sha256(tracking.read_bytes()).hexdigest() == source_hash
    window.close()
    print(destination)


if __name__ == '__main__':
    main()
