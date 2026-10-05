"""Launch the packaged Experiment Manager with its matching tracking core."""
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "analysis" / "particle_tracking_app_v0_1"))

try:
    import torch
    import ultralytics
except Exception:
    pass  # The application displays the dependency report.

from particle_tracking_app.manager import main

if __name__ == "__main__":
    raise SystemExit(main())
