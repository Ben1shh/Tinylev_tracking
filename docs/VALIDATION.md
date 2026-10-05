# Validation

Windows, independent Python 3.12.14 environment:

- 16 tests passed: numerical kinematics/spectrum checks, annotation revisions and
  provenance, full-frame fast output, backend calibration and Tracker GUI checks.
- Fresh GUI shows Tinylev Tracker v0.5, nine plot modes, manual annotations and
  unset calibration. Explicit zero center is valid; missing/nonfinite calibration
  cannot start inference, and choosing a new video clears previous calibration.
- Existing outputs receive a numbered sibling; annotation revisions preserve prior files.
- The bundled model loaded and completed CPU inference on a synthetic blank
  640x640 frame (zero detections); the actual GUI entry point exited normally.
- README screenshots use the actual Tracker UI, part02 recording frame 1000 and
  its existing 51,070-row tracking CSV, loaded read-only. No new tracking or
  experimental relabeling was performed. Capture with
  `python docs/create_preview.py --video YOUR_VIDEO --tracking MATCHING_CSV --frame 1000`.
  Only screenshot images and source metadata are published; inputs remain local.

Environment: NumPy 2.5.3, OpenCV 5.0.0.93, PyQt5 5.15.11,
Matplotlib 3.11.2, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu,
Ultralytics 8.4.173. These record the tested environment; requirements use minimum
versions. GPU use and a full new experimental video have not been validated in
this release. Synthetic tests do not establish experimental tracking accuracy or
physical particle identity.
