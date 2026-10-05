import os
import sys
from pathlib import Path


def _add(path: Path) -> None:
    if not path.exists():
        return
    text = str(path)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(text)
        except OSError:
            pass
    os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")


if hasattr(sys, "_MEIPASS"):
    base = Path(getattr(sys, "_MEIPASS"))
    _add(base / "torch" / "lib")
    _add(base)
