# Tinylev Tracking v0.1

NYU CSMR Grier Lab

Local PyQt desktop app for the trained single-class `particle` YOLO segmentation model.

## Setup

1. Open PowerShell and go to the app folder:

```powershell
cd "C:\Tinylev Tracking v0.1 Source"
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

3. Install packages:

```powershell
python -m pip install --upgrade pip
pip install PyQt5 opencv-python matplotlib pandas numpy ultralytics
```

4. *Optional NVIDIA GPU setup:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

5. Test dependencies:

```powershell
python -c "import torch, ultralytics, PyQt5, cv2, matplotlib, pandas, numpy; print('ok'); print('cuda', torch.cuda.is_available())"
```

## Run

```powershell
python run_particle_tracking_app.py
```

Bundled model:

- `models\best.pt`

## Workflow

1. Choose a video.
2. Confirm the model path points to `models\best.pt` or select another `.pt` model.
3. Set Calibration: Trap X, Trap Y, Pixels/mm.
4. Use Preview current frame.
5. Use Run batch tracking.
6. Results are written to the selected Output folder.

Each batch run writes:

- `config.json`
- `tracking.csv`
- optional `annotated_video_frames_<start>_to_<end>.mp4`
- optional `debug_frames\`

## Plot Options

- CM trap distance
- CM XY
- CM XY trajectory
- Orientation
- Angular velocity
- Relative angle phi
- Spin state timeline
- Particle XY
- Confidence

## Particle Identity Tracking

For two-particle tracking, `particle_1_*` and `particle_2_*` are persistent identities. The first tracked frame initializes identities from left to right, and later frames keep each identity by nearest-position continuity from the previous tracked frame. Size-based fields such as `small_*` and `large_*` are measurements only and no longer decide particle identity.

The CSV also includes `small_particle_id` and `large_particle_id` so size changes can be inspected without identity swapping.

## Spin / Angle CSV Columns

The tracking CSV includes:

- `psi_rad`
- `psi_unwrapped_rad`
- `theta_rad`
- `phi_rad`
- `omega_rad_s`
- `spin_freq_hz`
- `spin_state`

Tracking failed frames are kept in the CSV and receive `NaN` for unavailable numeric values.

## Notes

If the GUI opens but tracking fails with an `ultralytics` or `torch` import error, launch the app with the same Python interpreter that successfully runs YOLO training/inference in PyCharm or Jupyter.
