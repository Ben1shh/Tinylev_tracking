from __future__ import annotations

import csv
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

YOLO_IMPORT_ERROR: BaseException | None = None
YOLO_IMPORT_TRACEBACK: str | None = None
YOLO_CLASS = None
_DLL_PATHS_CONFIGURED = False


DEFAULT_PROJECT_ROOT = Path(r"C:\SpiningPaperExperiment")
APP_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = APP_ROOT.parent
LOCAL_MODEL_PATH = PACKAGE_ROOT / "models" / "best.pt"
LOCAL_FALLBACK_MODEL_PATH = PACKAGE_ROOT / "models" / "yolo26n.pt"
APP_MODEL_PATH = APP_ROOT / "models" / "best.pt"
APP_FALLBACK_MODEL_PATH = APP_ROOT / "models" / "yolo26n.pt"
TRAINED_MODEL_PATH = DEFAULT_PROJECT_ROOT / "ml_models" / "particle_seg_v2_fast_shutter_finetune" / "weights" / "best.pt"
FALLBACK_MODEL_PATH = DEFAULT_PROJECT_ROOT / "yolo26n.pt"
DEFAULT_MODEL_PATH = next(
    (
        path
        for path in [
            LOCAL_MODEL_PATH,
            LOCAL_FALLBACK_MODEL_PATH,
            APP_MODEL_PATH,
            APP_FALLBACK_MODEL_PATH,
            TRAINED_MODEL_PATH,
            FALLBACK_MODEL_PATH,
        ]
        if path.exists()
    ),
    FALLBACK_MODEL_PATH,
)
DEFAULT_VIDEO_PATH = DEFAULT_PROJECT_ROOT / "LowExporesure_Training" / "12_19_15MJPG-0003.avi"


@dataclass
class TrackingConfig:
    video_path: str = str(DEFAULT_VIDEO_PATH)
    model_path: str = str(DEFAULT_MODEL_PATH)
    output_dir: str = ""
    trap_x: float = 608.385
    trap_y: float = 444.807
    pixels_per_mm: float = 138.0
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
    save_annotated_video: bool = True
    save_debug_frames: bool = True
    debug_frame_limit: int = 25

    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        video = Path(self.video_path)
        return video.parent / f"{video.stem}_particle_tracking_app"


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

def _row_float(row: dict, key: str) -> float:
    value = row.get(key)
    if value in ("", None):
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _valid_number(value: float) -> bool:
    return bool(np.isfinite(value))


def _contiguous_true_segments(mask: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    start: int | None = None
    for idx, ok in enumerate(mask.tolist()):
        if ok and start is None:
            start = idx
        elif not ok and start is not None:
            segments.append(np.arange(start, idx))
            start = None
    if start is not None:
        segments.append(np.arange(start, len(mask)))
    return segments


def compute_pair_kinematics(
    rows: list[dict],
    trap_x: float,
    trap_y: float,
    pixels_per_mm: float,
    omega_threshold: float = 1.0,
) -> list[dict]:
    if not rows:
        return rows

    nan = float("nan")
    numeric_columns = [
        "psi_rad",
        "psi_unwrapped_rad",
        "theta_rad",
        "phi_rad",
        "omega_rad_s",
        "spin_freq_hz",
    ]
    for row in rows:
        for key in numeric_columns:
            row[key] = nan
        row["spin_state"] = 0

    indexed_rows = sorted(enumerate(rows), key=lambda item: _row_float(item[1], "time_s"))
    if not indexed_rows:
        return rows

    original_indices = [idx for idx, _row in indexed_rows]
    sorted_rows = [row for _idx, row in indexed_rows]
    n = len(sorted_rows)
    time = np.array([_row_float(row, "time_s") for row in sorted_rows], dtype=float)

    r1 = np.array([_row_float(row, "particle_1_radius_px") for row in sorted_rows], dtype=float)
    r2 = np.array([_row_float(row, "particle_2_radius_px") for row in sorted_rows], dtype=float)
    if np.isfinite(r1).any() and np.isfinite(r2).any():
        track1_is_small_default = float(np.nanmedian(r1)) <= float(np.nanmedian(r2))
    else:
        track1_is_small_default = True

    psi = np.full(n, nan, dtype=float)
    theta = np.full(n, nan, dtype=float)
    phi = np.full(n, nan, dtype=float)
    for idx, row in enumerate(sorted_rows):
        selected_count = _row_float(row, "selected_count")
        if not _valid_number(selected_count) or int(selected_count) < 2:
            continue

        p1x = _row_float(row, "particle_1_cx_px")
        p1y = _row_float(row, "particle_1_cy_px")
        p1r = _row_float(row, "particle_1_radius_px")
        p2x = _row_float(row, "particle_2_cx_px")
        p2y = _row_float(row, "particle_2_cy_px")
        p2r = _row_float(row, "particle_2_radius_px")
        cm_x = _row_float(row, "cm_x_px")
        cm_y = _row_float(row, "cm_y_px")
        if not all(_valid_number(v) for v in [cm_x, cm_y]):
            continue

        if all(_valid_number(v) for v in [p1x, p1y, p2x, p2y]):
            if _valid_number(p1r) and _valid_number(p2r) and not (np.isfinite(r1).any() and np.isfinite(r2).any()):
                track1_is_small = p1r <= p2r
            else:
                track1_is_small = track1_is_small_default
            if track1_is_small:
                sx, sy, lx, ly = p1x, p1y, p2x, p2y
            else:
                sx, sy, lx, ly = p2x, p2y, p1x, p1y
        else:
            sx = _row_float(row, "small_cx_px")
            sy = _row_float(row, "small_cy_px")
            lx = _row_float(row, "large_cx_px")
            ly = _row_float(row, "large_cy_px")
            if not all(_valid_number(v) for v in [sx, sy, lx, ly]):
                continue

        psi[idx] = math.atan2(ly - sy, lx - sx)
        theta[idx] = math.atan2(cm_y - trap_y, cm_x - trap_x)
        phi[idx] = (psi[idx] - theta[idx] + math.pi) % (2 * math.pi) - math.pi

    psi_unwrapped = np.full(n, nan, dtype=float)
    valid_psi = np.isfinite(psi)
    for segment in _contiguous_true_segments(valid_psi):
        psi_unwrapped[segment] = np.unwrap(psi[segment])

    omega = np.full(n, nan, dtype=float)
    valid_omega_source = np.isfinite(psi_unwrapped) & np.isfinite(time)
    for segment in _contiguous_true_segments(valid_omega_source):
        if len(segment) < 2:
            continue
        tseg = time[segment]
        yseg = psi_unwrapped[segment]
        if np.all(np.diff(tseg) > 0):
            omega[segment] = np.gradient(yseg, tseg)

    spin_freq_hz = nan
    if np.isfinite(omega).any():
        spin_freq_hz = float(np.nanmean(np.abs(omega)) / (2 * math.pi))

    state = np.zeros(n, dtype=int)
    state[omega > omega_threshold] = 1
    state[omega < -omega_threshold] = -1

    for sorted_idx, original_idx in enumerate(original_indices):
        row = rows[original_idx]
        row["psi_rad"] = float(psi[sorted_idx]) if np.isfinite(psi[sorted_idx]) else nan
        row["psi_unwrapped_rad"] = float(psi_unwrapped[sorted_idx]) if np.isfinite(psi_unwrapped[sorted_idx]) else nan
        row["theta_rad"] = float(theta[sorted_idx]) if np.isfinite(theta[sorted_idx]) else nan
        row["phi_rad"] = float(phi[sorted_idx]) if np.isfinite(phi[sorted_idx]) else nan
        row["omega_rad_s"] = float(omega[sorted_idx]) if np.isfinite(omega[sorted_idx]) else nan
        row["spin_freq_hz"] = spin_freq_hz
        row["spin_state"] = int(state[sorted_idx])
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


def particle_distance(a: ParticleDetection, b: ParticleDetection) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def pair_continuity_cost(
    pair: tuple[ParticleDetection, ParticleDetection],
    prev_pair: tuple[ParticleDetection, ParticleDetection],
) -> float:
    a, b = pair
    prev_1, prev_2 = prev_pair
    same_cost = particle_distance(a, prev_1) + particle_distance(b, prev_2)
    swapped_cost = particle_distance(a, prev_2) + particle_distance(b, prev_1)
    return min(same_cost, swapped_cost)


def order_pair_by_identity(
    pair: tuple[ParticleDetection, ParticleDetection],
    prev_pair: tuple[ParticleDetection, ParticleDetection] | None = None,
) -> tuple[ParticleDetection, ParticleDetection]:
    if prev_pair is None:
        return tuple(sorted(pair, key=lambda d: (d.cx, d.cy)))
    a, b = pair
    prev_1, prev_2 = prev_pair
    same_cost = particle_distance(a, prev_1) + particle_distance(b, prev_2)
    swapped_cost = particle_distance(a, prev_2) + particle_distance(b, prev_1)
    return (a, b) if same_cost <= swapped_cost else (b, a)


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
        continuity = pair_continuity_cost((a, b), prev_pair)
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
    return order_pair_by_identity(best, prev_pair)


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


def row_from_particles(
    frame_idx: int,
    fps: float,
    selected: list[ParticleDetection],
    detections: list[ParticleDetection],
    config: TrackingConfig,
) -> dict:
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

    particle_1, particle_2 = selected[0], selected[1]
    small, large = sort_small_large((particle_1, particle_2))
    small_id = 1 if small is particle_1 else 2
    large_id = 1 if large is particle_1 else 2
    sep_px = particle_distance(particle_1, particle_2)
    identity_angle_deg = math.degrees(math.atan2(particle_2.cy - particle_1.cy, particle_2.cx - particle_1.cx))
    size_order_angle_deg = math.degrees(math.atan2(large.cy - small.cy, large.cx - small.cx))
    base.update(
        {
            "particle_1_track_id": 1,
            "particle_2_track_id": 2,
            "particle_pair_distance_px": sep_px,
            "particle_pair_distance_mm": sep_px / config.pixels_per_mm,
            "particle_1_to_2_angle_deg": identity_angle_deg,
            "particle_separation_px": sep_px,
            "particle_separation_mm": sep_px / config.pixels_per_mm,
            "angle_deg": identity_angle_deg,
            "small_particle_id": small_id,
            "small_cx_px": small.cx,
            "small_cy_px": small.cy,
            "small_radius_px": small.radius,
            "small_conf": small.confidence,
            "large_particle_id": large_id,
            "large_cx_px": large.cx,
            "large_cy_px": large.cy,
            "large_radius_px": large.radius,
            "large_conf": large.confidence,
            "size_order_angle_deg": size_order_angle_deg,
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

    if layers.get("trap"):
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
    video_path = Path(config.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    model = load_model(config.model_path)
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    if config.save_debug_frames:
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
    if config.save_annotated_video:
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
        overlay = draw_overlay(frame, detections, selected, config, frame_idx=frame_idx, row=row)
        if writer is not None:
            writer.write(overlay)
        if config.save_debug_frames and (written_debug < config.debug_frame_limit or written_debug % 100 == 0):
            write_image(debug_dir / f"frame_{frame_idx:07d}.png", overlay)
        written_debug += 1
        if frame_cb:
            frame_cb(overlay, row)
        if progress_cb:
            progress_cb(frame_idx - start + 1, stop - start)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    compute_pair_kinematics(rows, config.trap_x, config.trap_y, config.pixels_per_mm)
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

