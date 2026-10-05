"""Launch Tinylev Tracker v0.5 with its matching tracking backend."""
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / 'analysis' / 'particle_tracking_app_v0_1'))

try:
    import torch
    import ultralytics
except Exception:
    pass  # The application reports dependency errors when loading a model.

from tinylev_tracker.app import main

if __name__ == '__main__':
    raise SystemExit(main())
