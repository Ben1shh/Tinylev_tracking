try:
    import torch
    import ultralytics
except Exception:
    pass

from particle_tracking_app.app import main

if __name__ == "__main__":
    raise SystemExit(main())
