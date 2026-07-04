try:
    import torch  # noqa: F401
    import ultralytics  # noqa: F401
except Exception:
    # The GUI will show the full dependency report if YOLO still cannot load.
    pass

from particle_tracking_app.app import main


if __name__ == "__main__":
    raise SystemExit(main())
