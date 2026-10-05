# Release validation

Validated on Windows with Python 3.12 in an independent environment, from the
repository directory:

- 27 manager/core tests passed, including eleven calibration and packaging regression tests.
- 12 dynamic-analysis synthetic regression tests passed.
- Fresh GUI initialization and switching to an uncalibrated experiment preserve
  unset calibration. Zero center coordinates remain valid when entered.
- Missing calibration cannot generate physical measurements; zero/negative/
  nonfinite scales are rejected. Unknown center errors are not treated as zero.
- Existing model hash matches the local current model; the model is not replaced.
- The bundled model loaded and completed CPU inference on a synthetic blank
  640x640 frame (zero detections). This is a runtime smoke check, not accuracy validation.
- Validation environment: Python 3.12.14, PyQt5 5.15.11, NumPy 2.5.3,
  OpenCV 5.0.0.93, pandas 3.0.6, Matplotlib 3.11.2,
  PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.173.
- README screenshot is rendered from the actual Qt application with explicitly
  synthetic, in-memory data by `python docs/create_preview.py`.

These tests do not establish experimental tracking accuracy, GPU performance,
full-video identity continuity or a physical dynamical state. Original experimental
data and historical analyses are not included or modified. The full manager's
tracking output directory must be newly chosen for each run to avoid replacement.
