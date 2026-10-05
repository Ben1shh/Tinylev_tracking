from __future__ import annotations

import csv
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

YOLO_IMPORT_ERROR: BaseException | None = None
YOLO_IMPORT_TRACEBACK: str | None = None
YOLO_CLASS = None
_DLL_PATHS_CONFIGURED = False


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parents[1]
RAW_DATA_ROOT = PROJECT_ROOT / "raw_data"
MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "particle_tracking"
DEFAULT_TRAP_CENTER_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "trap_center_calibration"
DEFAULT_MODEL_PATH = MODEL_ROOT / "best.pt"
DEFAULT_VIDEO_PATH = next(iter(sorted(RAW_DATA_ROOT.rglob("*.avi"))), RAW_DATA_ROOT)


@dataclass
class TrackingConfig:
    video_path: str = str(DEFAULT_VIDEO_PATH)
    model_path: str = str(DEFAULT_MODEL_PATH)
    output_dir: str = ""
    trap_x: float | None = None
    trap_y: float | None = None
    pixels_per_mm: float | None = None
    start_frame: int = 0
    end_frame: int | None = None
    frame_stride: int = 1
    imgsz: int = 1024
    conf: float = 0.25
    particle_count: int = 2
    roi_x_min: int = 300
    roi_x_max: int = 900
    roi_y_min: int = 250
    roi_y_max: int = 590
    min_radius: float = 45.0
    max_radius: float = 145.0
    min_pair_distance: float = 75.0
    max_pair_distance: float = 360.0
    mask_open_px: int = 3
    mask_erode_px: int = 2
    circle_radius_scale: float = 1.0
    fast_mode: bool = False
    preview_stride: int = 10
    save_annotated_video: bool = True
    save_debug_frames: bool = True
    debug_frame_limit: int = 25

    def validate_calibration(self) -> None:
        values = (self.trap_x, self.trap_y, self.pixels_per_mm)
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError("Calibration required: set Trap X, Trap Y and Pixels/mm for this experiment.")
        if self.pixels_per_mm <= 0:
            raise ValueError("Calibration Pixels/mm must be positive.")

    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        video = Path(self.video_path)
        return DEFAULT_OUTPUT_ROOT / f"{video.stem}_particle_tracking_app"


@dataclass
class ParticleDetection:
    cx: float
    cy: float
    radius: float
    confidence: float
    mask_area: float
    fit_error: float
    source_index: int
    mask: np.ndarray | None = None


def read_image(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


def write_config(path: str | Path, config: TrackingConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def read_config(path: str | Path) -> TrackingConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrackingConfig(**data)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_tracking_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def configure_torch_dll_paths() -> None:
    global _DLL_PATHS_CONFIGURED
    if _DLL_PATHS_CONFIGURED:
        return
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        base = Path(getattr(sys, "_MEIPASS"))
        candidates.extend([base / "torch" / "lib", base])
    try:
        import torch as _torch_probe

        candidates.append(Path(_torch_probe.__file__).resolve().parent / "lib")
    except Exception:
        pass

    path_entries = []
    for candidate in candidates:
        if candidate.exists():
            path_entries.append(str(candidate))
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(candidate))
                except OSError:
                    pass
    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries + [os.environ.get("PATH", "")])
    _DLL_PATHS_CONFIGURED = True


def load_yolo_class():
    global YOLO_CLASS, YOLO_IMPORT_ERROR, YOLO_IMPORT_TRACEBACK
    if YOLO_CLASS is not None:
        return YOLO_CLASS
    try:
        configure_torch_dll_paths()
        from ultralytics import YOLO
    except Exception as exc:
        YOLO_IMPORT_ERROR = exc
        YOLO_IMPORT_TRACEBACK = traceback.format_exc()
        raise ImportError(
            "Failed to import ultralytics/YOLO in this Python environment.\n"
            f"Python executable: {sys.executable}\n"
            f"Original error: {type(exc).__name__}: {exc}\n\n"
            f"{YOLO_IMPORT_TRACEBACK}"
        ) from exc
    YOLO_CLASS = YOLO
    return YOLO_CLASS


def dependency_report() -> str:
    lines = [f"Python executable: {sys.executable}", f"Python version: {sys.version.split()[0]}"]
    for module in ["cv2", "PyQt5", "matplotlib", "pandas", "numpy", "torch", "ultralytics"]:
        try:
            mod = __import__(module)
            lines.append(f"{module}: OK {getattr(mod, '__version__', '')}".rstrip())
        except Exception as exc:
            lines.append(f"{module}: ERROR {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def load_model(model_path: str | Path):
    YOLO = load_yolo_class()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def polygon_to_mask(points: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    if pts.ndim == 2 and len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask


def clean_inference_mask(mask: np.ndarray, config: TrackingConfig) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if config.mask_open_px > 1:
        k = int(config.mask_open_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    if config.mask_erode_px > 0:
        k = int(config.mask_erode_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.erode(binary, kernel, iterations=1)
    return binary


def fit_circle_from_mask(mask: np.ndarray, confidence: float, source_index: int, config: TrackingConfig) -> ParticleDetection | None:
    clean = clean_inference_mask(mask, config)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        pts = contour.reshape(-1, 2)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
    radius = math.sqrt(area / math.pi) * float(config.circle_radius_scale)
    pts = contour.reshape(-1, 2).astype(np.float32)
    distances = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    fit_error = float(np.std(distances) / max(radius, 1.0)) if len(distances) else 0.0
    return ParticleDetection(float(cx), float(cy), float(radius), float(confidence), area, fit_error, source_index, clean)


def infer_particle_detections(model, frame_bgr: np.ndarray, config: TrackingConfig) -> list[ParticleDetection]:
    result = model.predict(frame_bgr, imgsz=config.imgsz, conf=config.conf, verbose=False)[0]
    if result.masks is None:
        return []
    confidences = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes is not None else []
    detections: list[ParticleDetection] = []
    for idx, poly in enumerate(result.masks.xy):
        score = float(confidences[idx]) if idx < len(confidences) else 0.0
        mask = polygon_to_mask(np.asarray(poly), frame_bgr.shape[:2])
        det = fit_circle_from_mask(mask, score, idx, config)
        if det is None:
            continue
        if not (config.min_radius <= det.radius <= config.max_radius):
            continue
        if not (config.roi_x_min <= det.cx <= config.roi_x_max and config.roi_y_min <= det.cy <= config.roi_y_max):
            continue
        detections.append(det)
    return detections


def sort_small_large(pair: tuple[ParticleDetection, ParticleDetection]) -> tuple[ParticleDetection, ParticleDetection]:
    small, large = sorted(pair, key=lambda d: d.radius)
    return small, large


def center_of_mass_xy(particles: Iterable[ParticleDetection]) -> tuple[float, float]:
    particles = list(particles)
    if not particles:
        return 0.0, 0.0
    weights = [max(d.radius, 0.0) ** 3 for d in particles]
    total_weight = sum(weights)
    if total_weight <= 0:
        return (
            sum(d.cx for d in particles) / len(particles),
            sum(d.cy for d in particles) / len(particles),
        )
    cm_x = sum(d.cx * w for d, w in zip(particles, weights)) / total_weight
    cm_y = sum(d.cy * w for d, w in zip(particles, weights)) / total_weight
    return cm_x, cm_y


def detection_pair_score(
    a: ParticleDetection,
    b: ParticleDetection,
    config: TrackingConfig,
    prev_pair: tuple[ParticleDetection, ParticleDetection] | None = None,
) -> float:
    sep = math.hypot(a.cx - b.cx, a.cy - b.cy)
    if sep < config.min_pair_distance or sep > config.max_pair_distance:
        return -1e9
    cm_x, cm_y = center_of_mass_xy((a, b))
    trap_dist = math.hypot(cm_x - config.trap_x, cm_y - config.trap_y)
    score = a.confidence + b.confidence - 0.20 * (a.fit_error + b.fit_error) - trap_dist / 1200.0
    if prev_pair is not None:
        prev_small, prev_large = prev_pair
        small, large = sort_small_large((a, b))
        continuity = math.hypot(small.cx - prev_small.cx, small.cy - prev_small.cy) + math.hypot(
            large.cx - prev_large.cx, large.cy - prev_large.cy
        )
        score -= continuity / 700.0
    return score


def select_two_particles(
    detections: list[ParticleDetection],
    config: TrackingConfig,
    prev_pair: tuple[ParticleDetection, ParticleDetection] | None = None,
) -> tuple[ParticleDetection, ParticleDetection] | None:
    if len(detections) < 2:
        return None
    best = None
    best_score = -1e9
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            score = detection_pair_score(detections[i], detections[j], config, prev_pair)
            if score > best_score:
                best_score = score
                best = (detections[i], detections[j])
    if best is None:
        return None
    return sort_small_large(best)


def select_particles(
    detections: list[ParticleDetection],
    config: TrackingConfig,
    prev_pair: tuple[ParticleDetection, ParticleDetection] | None = None,
) -> list[ParticleDetection]:
    if config.particle_count == 2:
        pair = select_two_particles(detections, config, prev_pair=prev_pair)
        return list(pair) if pair is not None else []
    ranked = sorted(
        detections,
        key=lambda d: (
            d.confidence - 0.20 * d.fit_error - math.hypot(d.cx - config.trap_x, d.cy - config.trap_y) / 1600.0
        ),
        reverse=True,
    )
    if config.particle_count <= 0:
        selected = ranked
    else:
        selected = ranked[: max(0, int(config.particle_count))]
    return sorted(selected, key=lambda d: (d.cx, d.cy))


def select_single_particle(detections: list[ParticleDetection]) -> ParticleDetection | None:
    """Select the most credible bead without assuming a known trap position."""
    if not detections:
        return None
    return max(detections, key=lambda d: (d.confidence - 0.20 * d.fit_error, d.mask_area))


def row_from_particles(
    frame_idx: int,
    fps: float,
    selected: list[ParticleDetection],
    detections: list[ParticleDetection],
    config: TrackingConfig,
) -> dict:
    config.validate_calibration()
    selected_count = len(selected)
    requested = int(config.particle_count)
    ok = selected_count > 0 if requested <= 0 else selected_count == requested
    base = {
        "frame": frame_idx,
        "time_s": frame_idx / fps if fps > 0 else 0.0,
        "status": "ok" if ok else "missing_particles",
        "num_candidates": len(detections),
        "requested_count": "auto" if requested <= 0 else requested,
        "selected_count": selected_count,
    }
    for idx, det in enumerate(selected, start=1):
        base.update(
            {
                f"particle_{idx}_cx_px": det.cx,
                f"particle_{idx}_cy_px": det.cy,
                f"particle_{idx}_radius_px": det.radius,
                f"particle_{idx}_conf": det.confidence,
                f"particle_{idx}_fit_error": det.fit_error,
                f"particle_{idx}_mask_area": det.mask_area,
            }
        )
    if not selected:
        return base

    cm_x, cm_y = center_of_mass_xy(selected)
    trap_dx = cm_x - config.trap_x
    trap_dy = cm_y - config.trap_y
    base.update(
        {
            "cm_x_px": cm_x,
            "cm_y_px": cm_y,
            "cm_dx_from_trap_px": trap_dx,
            "cm_dy_from_trap_px": trap_dy,
            "cm_distance_from_trap_px": math.hypot(trap_dx, trap_dy),
            "cm_distance_from_trap_mm": math.hypot(trap_dx, trap_dy) / config.pixels_per_mm,
            "centroid_x_px": cm_x,
            "centroid_y_px": cm_y,
            "centroid_dx_from_trap_px": trap_dx,
            "centroid_dy_from_trap_px": trap_dy,
            "centroid_distance_from_trap_px": math.hypot(trap_dx, trap_dy),
            "centroid_distance_from_trap_mm": math.hypot(trap_dx, trap_dy) / config.pixels_per_mm,
        }
    )

    if selected_count != 2:
        return base

    small, large = sort_small_large((selected[0], selected[1]))
    sep_px = math.hypot(large.cx - small.cx, large.cy - small.cy)
    angle_deg = math.degrees(math.atan2(large.cy - small.cy, large.cx - small.cx))
    base.update(
        {
            "small_cx_px": small.cx,
            "small_cy_px": small.cy,
            "small_radius_px": small.radius,
            "small_conf": small.confidence,
            "large_cx_px": large.cx,
            "large_cy_px": large.cy,
            "large_radius_px": large.radius,
            "large_conf": large.confidence,
            "particle_separation_px": sep_px,
            "particle_separation_mm": sep_px / config.pixels_per_mm,
            "angle_deg": angle_deg,
        }
    )
    return base


DEFAULT_LAYERS = {
    "masks": True,
    "circles": True,
    "centers": True,
    "CM": True,
    "trap": False,
    "line": False,
    "direction": False,
    "text": False,
}


def _draw_mask_overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.22) -> None:
    if mask is None:
        return
    colored = np.zeros_like(frame)
    colored[:] = color
    active = mask > 0
    frame[active] = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)[active]


def draw_overlay(
    frame: np.ndarray,
    detections: Iterable[ParticleDetection] | None,
    selected_particles: Iterable[ParticleDetection] | None,
    config: TrackingConfig,
    frame_idx: int | None = None,
    row: dict | None = None,
    layers: dict[str, bool] | None = None,
) -> np.ndarray:
    layers = {**DEFAULT_LAYERS, **(layers or {})}
    out = frame.copy()
    detections = list(detections or [])
    selected_particles = list(selected_particles or [])

    if layers.get("masks"):
        for det in detections:
            _draw_mask_overlay(out, det.mask, (60, 160, 255), alpha=0.18)

    colors = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 180, 0),
        (255, 0, 255),
        (0, 255, 255),
        (180, 80, 255),
        (255, 255, 0),
        (120, 220, 120),
    ]

    if selected_particles:
        if layers.get("masks"):
            for idx, det in enumerate(selected_particles):
                _draw_mask_overlay(out, det.mask, colors[idx % len(colors)], alpha=0.20)
        for idx, det in enumerate(selected_particles, start=1):
            color = colors[(idx - 1) % len(colors)]
            center = (round(det.cx), round(det.cy))
            if layers.get("circles"):
                cv2.circle(out, center, round(det.radius), color, 2)
            if layers.get("centers"):
                cv2.drawMarker(out, center, color, cv2.MARKER_CROSS, 18, 2)
            if layers.get("text"):
                cv2.putText(out, str(idx), (center[0] + 8, center[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        if len(selected_particles) == 2 and (layers.get("line") or layers.get("direction")):
            small, large = sort_small_large((selected_particles[0], selected_particles[1]))
            p1 = (round(small.cx), round(small.cy))
            p2 = (round(large.cx), round(large.cy))
            if layers.get("line"):
                cv2.line(out, p1, p2, (255, 255, 0), 2)
            if layers.get("direction"):
                cv2.arrowedLine(out, p1, p2, (255, 180, 0), 2, tipLength=0.18)
        if layers.get("CM"):
            cm_x, cm_y = center_of_mass_xy(selected_particles)
            cx = round(cm_x)
            cy = round(cm_y)
            cv2.drawMarker(out, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

    if row is not None and int(float(row.get("selected_count", 0) or 0)) > 0:
        if layers.get("circles"):
            for idx in range(1, int(float(row.get("selected_count", 0))) + 1):
                color = colors[(idx - 1) % len(colors)]
                cx = row.get(f"particle_{idx}_cx_px")
                cy = row.get(f"particle_{idx}_cy_px")
                radius = row.get(f"particle_{idx}_radius_px")
                if cx != "" and cy != "" and radius != "" and cx is not None and cy is not None and radius is not None:
                    cv2.circle(out, (round(float(cx)), round(float(cy))), round(float(radius)), color, 2)
        if layers.get("centers"):
            for idx in range(1, int(float(row.get("selected_count", 0))) + 1):
                color = colors[(idx - 1) % len(colors)]
                cx = row.get(f"particle_{idx}_cx_px")
                cy = row.get(f"particle_{idx}_cy_px")
                if cx != "" and cy != "" and cx is not None and cy is not None:
                    cv2.drawMarker(out, (round(float(cx)), round(float(cy))), color, cv2.MARKER_CROSS, 18, 2)
        if layers.get("line") or layers.get("direction"):
            sx, sy = row.get("small_cx_px"), row.get("small_cy_px")
            lx, ly = row.get("large_cx_px"), row.get("large_cy_px")
            if all(v not in ("", None) for v in [sx, sy, lx, ly]):
                p1 = (round(float(sx)), round(float(sy)))
                p2 = (round(float(lx)), round(float(ly)))
                if layers.get("line"):
                    cv2.line(out, p1, p2, (255, 255, 0), 2)
                if layers.get("direction"):
                    cv2.arrowedLine(out, p1, p2, (255, 180, 0), 2, tipLength=0.18)
        if layers.get("CM"):
            cx = row.get("cm_x_px", row.get("centroid_x_px"))
            cy = row.get("cm_y_px", row.get("centroid_y_px"))
            if cx not in ("", None) and cy not in ("", None):
                cv2.drawMarker(out, (round(float(cx)), round(float(cy))), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

    if layers.get("trap") and all(v is not None and math.isfinite(v) for v in (config.trap_x, config.trap_y)):
        cv2.drawMarker(out, (round(config.trap_x), round(config.trap_y)), (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(out, (round(config.trap_x), round(config.trap_y)), 8, (255, 0, 255), 2)

    if layers.get("text"):
        label = f"frame={frame_idx}" if frame_idx is not None else ""
        if row is not None:
            label += f" status={row.get('status', '')}"
            if row.get("status") == "ok":
                label += f" count={row.get('selected_count', '')}"
                if row.get("particle_separation_px") not in ("", None):
                    label += f" sep={float(row.get('particle_separation_px', 0)):.1f}px"
                if row.get("angle_deg") not in ("", None):
                    label += f" angle={float(row.get('angle_deg', 0)):.1f}"
        cv2.putText(out, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return out


def process_frame(model, frame: np.ndarray, frame_idx: int, fps: float, config: TrackingConfig, prev_pair=None):
    config.validate_calibration()
    detections = infer_particle_detections(model, frame, config)
    selected = select_particles(detections, config, prev_pair=prev_pair)
    row = row_from_particles(frame_idx, fps, selected, detections, config)
    return detections, selected, row


def track_video(
    config: TrackingConfig,
    progress_cb: Callable[[int, int], None] | None = None,
    frame_cb: Callable[[np.ndarray, dict], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> list[dict]:
    config.validate_calibration()
    video_path = Path(config.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    model = load_model(config.model_path)
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    save_debug_frames = bool(config.save_debug_frames and not config.fast_mode)
    save_annotated_video = bool(config.save_annotated_video and not config.fast_mode)
    if save_debug_frames:
        debug_dir.mkdir(parents=True, exist_ok=True)
    write_config(output_dir / "config.json", config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = max(0, int(config.start_frame))
    stop = total if config.end_frame is None else min(int(config.end_frame), total)
    if stop <= start:
        raise ValueError(f"Invalid frame range: start={start}, end={config.end_frame}, total={total}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    writer = None
    if save_annotated_video:
        out_video = output_dir / f"annotated_video_frames_{start:07d}_to_{stop - 1:07d}.mp4"
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(config.frame_stride, 1), (width, height))

    rows: list[dict] = []
    prev_pair = None
    frame_idx = start
    written_debug = 0
    while frame_idx < stop:
        if stop_check is not None and stop_check():
            break
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start) % max(config.frame_stride, 1) != 0:
            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx - start, stop - start)
            continue
        detections, selected, row = process_frame(model, frame, frame_idx, fps, config, prev_pair)
        if config.particle_count == 2 and len(selected) == 2:
            prev_pair = (selected[0], selected[1])
        rows.append(row)
        preview_stride = max(1, int(config.preview_stride))
        is_last_processed_frame = frame_idx + max(config.frame_stride, 1) >= stop
        should_preview = (
            not config.fast_mode
            or ((frame_idx - start) // max(config.frame_stride, 1)) % preview_stride == 0
            or is_last_processed_frame
        )
        needs_overlay = writer is not None or save_debug_frames or (frame_cb is not None and should_preview)
        overlay = None
        if needs_overlay:
            layers = None
            if config.fast_mode:
                layers = {
                    "masks": False,
                    "circles": True,
                    "centers": True,
                    "CM": True,
                    "trap": True,
                    "line": False,
                    "direction": False,
                    "text": True,
                }
            overlay = draw_overlay(frame, detections, selected, config, frame_idx=frame_idx, row=row, layers=layers)
        if writer is not None and overlay is not None:
            writer.write(overlay)
        if save_debug_frames and overlay is not None and (
            written_debug < config.debug_frame_limit or written_debug % 100 == 0
        ):
            write_image(debug_dir / f"frame_{frame_idx:07d}.png", overlay)
        written_debug += 1
        if frame_cb and should_preview and overlay is not None:
            frame_cb(overlay, row)
        if progress_cb and (not config.fast_mode or should_preview):
            progress_cb(frame_idx - start + 1, stop - start)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    write_csv(output_dir / "tracking.csv", rows)
    return rows


def get_video_info(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    return info


def read_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _unused_output_dir(requested: str | Path) -> Path:
    """Return a new derived-output path without overwriting an earlier run."""
    requested = Path(requested)
    if not requested.exists():
        return requested
    for index in range(2, 10000):
        candidate = requested.with_name(f"{requested.name}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a new output directory beside: {requested}")


def _draw_trap_center_preview(
    background: np.ndarray,
    valid_rows: list[dict],
    trap_x_px: float,
    trap_y_px: float,
) -> np.ndarray:
    preview = background.copy()
    overlay = preview.copy()
    for row in valid_rows:
        center = (round(float(row["center_x_px"])), round(float(row["center_y_px"])))
        radius = max(1, round(float(row["radius_px"])))
        cv2.circle(overlay, center, radius, (0, 210, 255), 2)
        cv2.drawMarker(overlay, center, (0, 210, 255), cv2.MARKER_CROSS, 14, 1)
    preview = cv2.addWeighted(preview, 0.58, overlay, 0.42, 0)
    mean_center = (round(trap_x_px), round(trap_y_px))
    cv2.drawMarker(preview, mean_center, (255, 0, 255), cv2.MARKER_CROSS, 34, 3)
    cv2.circle(preview, mean_center, 12, (255, 0, 255), 3)
    cv2.putText(
        preview,
        f"mean trap center = ({trap_x_px:.3f}, {trap_y_px:.3f}) px; n={len(valid_rows)}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    return preview


def calibrate_trap_center(
    image_paths: Iterable[str | Path],
    config: TrackingConfig,
    output_dir: str | Path,
    progress_cb: Callable[[int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
    model=None,
) -> dict:
    """Detect one bead per image and average its image-coordinate center."""
    paths = [Path(path).resolve() for path in image_paths]
    if not paths:
        raise ValueError("Select at least one calibration image.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Calibration image not found: {missing[0]}")

    output_dir = _unused_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    annotated_dir = output_dir / "annotated_images"
    annotated_dir.mkdir()
    calibration_config = {
        "purpose": "trap_center_calibration_from_single_particle_images",
        "method": "arithmetic_mean_of_detected_particle_centers_px",
        "image_paths": [str(path) for path in paths],
        "model_path": str(Path(config.model_path).resolve()),
        "shared_model_settings": {
            key: value
            for key, value in asdict(config).items()
            if key
            in {
                "imgsz",
                "conf",
                "mask_open_px",
                "mask_erode_px",
                "circle_radius_scale",
            }
        },
        "calibration_candidate_filter": {
            "roi": "full_image",
            "radius_filter": "disabled",
            "selection": "highest confidence minus 0.20 times normalized circle-fit error; mask area breaks ties",
        },
    }
    (output_dir / "calibration_config.json").write_text(
        json.dumps(calibration_config, indent=2), encoding="utf-8"
    )

    if model is None:
        model = load_model(config.model_path)
    rows: list[dict] = []
    reference_shape: tuple[int, int] | None = None
    preview_background: np.ndarray | None = None
    stopped = False
    total = len(paths)
    for image_index, path in enumerate(paths, start=1):
        if stop_check is not None and stop_check():
            stopped = True
            break
        row = {
            "image_index": image_index,
            "image_path": str(path),
            "image_name": path.name,
            "status": "unreadable_image",
            "used_for_average": False,
            "num_candidates": 0,
        }
        frame = read_image(path)
        if frame is None:
            rows.append(row)
            if progress_cb:
                progress_cb(image_index, total)
            continue
        height, width = frame.shape[:2]
        row.update({"image_width_px": width, "image_height_px": height})
        shape = (height, width)
        if reference_shape is None:
            reference_shape = shape
            preview_background = frame.copy()
        if shape != reference_shape:
            row["status"] = "image_shape_mismatch"
            rows.append(row)
            if progress_cb:
                progress_cb(image_index, total)
            continue

        calibration_inference_config = replace(
            config,
            roi_x_min=0,
            roi_x_max=width,
            roi_y_min=0,
            roi_y_max=height,
            min_radius=0.0,
            max_radius=float(max(width, height)),
        )
        detections = infer_particle_detections(model, frame, calibration_inference_config)
        selected = select_single_particle(detections)
        row["num_candidates"] = len(detections)
        if selected is None:
            row["status"] = "no_particle_detected"
            rows.append(row)
            if progress_cb:
                progress_cb(image_index, total)
            continue

        row.update(
            {
                "status": "ok" if len(detections) == 1 else "ok_multiple_candidates",
                "used_for_average": True,
                "center_x_px": selected.cx,
                "center_y_px": selected.cy,
                "radius_px": selected.radius,
                "confidence": selected.confidence,
                "fit_error": selected.fit_error,
                "mask_area_px2": selected.mask_area,
            }
        )
        rows.append(row)
        annotated = draw_overlay(
            frame,
            detections,
            [selected],
            config,
            row={
                "status": row["status"],
                "selected_count": 1,
                "particle_1_cx_px": selected.cx,
                "particle_1_cy_px": selected.cy,
                "particle_1_radius_px": selected.radius,
            },
            layers={"trap": False, "CM": False, "text": True},
        )
        write_image(annotated_dir / f"{image_index:03d}_{path.stem}.png", annotated)
        if progress_cb:
            progress_cb(image_index, total)

    write_csv(output_dir / "detections.csv", rows)
    valid_rows = [row for row in rows if row.get("used_for_average")]
    summary = {
        "status": "stopped" if stopped else ("ok" if valid_rows else "no_valid_detections"),
        "method": "arithmetic_mean_of_detected_particle_centers_px",
        "input_image_count": total,
        "processed_image_count": len(rows),
        "valid_detection_count": len(valid_rows),
        "excluded_or_failed_count": len(rows) - len(valid_rows),
        "reference_image_width_px": reference_shape[1] if reference_shape else None,
        "reference_image_height_px": reference_shape[0] if reference_shape else None,
        "trap_x_px": None,
        "trap_y_px": None,
        "sample_std_x_px": None,
        "sample_std_y_px": None,
        "standard_error_x_px": None,
        "standard_error_y_px": None,
    }
    preview_path: Path | None = None
    if valid_rows:
        xs = np.asarray([float(row["center_x_px"]) for row in valid_rows], dtype=float)
        ys = np.asarray([float(row["center_y_px"]) for row in valid_rows], dtype=float)
        n = len(valid_rows)
        std_x = float(np.std(xs, ddof=1)) if n > 1 else 0.0
        std_y = float(np.std(ys, ddof=1)) if n > 1 else 0.0
        summary.update(
            {
                "trap_x_px": float(np.mean(xs)),
                "trap_y_px": float(np.mean(ys)),
                "sample_std_x_px": std_x,
                "sample_std_y_px": std_y,
                "standard_error_x_px": std_x / math.sqrt(n),
                "standard_error_y_px": std_y / math.sqrt(n),
            }
        )
        if preview_background is not None:
            preview = _draw_trap_center_preview(
                preview_background, valid_rows, summary["trap_x_px"], summary["trap_y_px"]
            )
            preview_path = output_dir / "overlay_preview.png"
            write_image(preview_path, preview)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "trap_center_summary.csv", [summary])
    return {
        "output_dir": output_dir,
        "rows": rows,
        "summary": summary,
        "preview_path": preview_path,
    }
