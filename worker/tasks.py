"""
TrafficLens AI — Complete Celery Worker
========================================
This is the engine of the project. It:
  1. Receives a video path from the FastAPI queue
  2. Extracts frames with OpenCV
  3. Runs YOLOv8 vehicle detection on every Nth frame
  4. Runs helmet violation detection on rider crops
  5. Classifies congestion level with temporal smoothing
  6. Saves violation frame snapshots to disk
  7. Batch-writes all results to PostgreSQL
  8. Updates the job status throughout

Run with:
    celery -A worker.tasks worker --loglevel=info --concurrency=2
"""

import os
import json
import logging
import traceback
from collections import deque
from datetime import datetime
from statistics import mode
from pathlib import Path

import cv2
import numpy as np
from celery import Celery
from sqlalchemy.orm import Session
from ultralytics import YOLO

# ── Import your DB layer ──────────────────────────────────────────────────────
# Adjust these imports to match your actual project structure
from database.database import SessionLocal
from database import models

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SNAPSHOT_DIR    = os.getenv("SNAPSHOT_DIR", "/tmp/trafficlens_snapshots")
VEHICLE_MODEL   = os.getenv("VEHICLE_MODEL_PATH", "yolov8n.pt")
HELMET_MODEL    = os.getenv("HELMET_MODEL_PATH",  "models/helmet_yolov8.pt")

# Process every Nth frame — balance between accuracy and speed
# 5 = analyse 6 frames/sec of a 30fps video (good for demo)
FRAME_SKIP      = int(os.getenv("FRAME_SKIP", 5))

# Batch size for DB writes — commit every N detections for speed
DB_BATCH_SIZE   = int(os.getenv("DB_BATCH_SIZE", 50))

# YOLOv8 COCO class IDs for vehicles
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Congestion thresholds — tune these per camera / junction type
# Tip: Silk Board will need higher thresholds than a residential road
CONGESTION_LOW_MAX    = int(os.getenv("CONGESTION_LOW_MAX",    6))
CONGESTION_MEDIUM_MAX = int(os.getenv("CONGESTION_MEDIUM_MAX", 16))

# Smoothing window — number of frames to average over before assigning level
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", 5))

# Minimum confidence to accept a detection
VEHICLE_CONF_THRESHOLD  = float(os.getenv("VEHICLE_CONF", 0.40))
HELMET_CONF_THRESHOLD   = float(os.getenv("HELMET_CONF",  0.50))

# ROI polygon — pixel coordinates of the road zone to analyse
# These are for a typical wide-angle junction camera.
# Override via env or tune per camera in the cameras table.
# ROI as PERCENTAGES of frame width/height — works for any video resolution
DEFAULT_ROI_RELATIVE = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trafficlens.worker")

Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CELERY APP
# ─────────────────────────────────────────────────────────────────────────────

celery_app = Celery(
    "trafficlens",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    task_track_started=True,
    worker_prefetch_multiplier=1,  # one video at a time per worker
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# Models are loaded once when the worker process starts — NOT per task.
# Loading YOLOv8 per task would add ~2s overhead every video.
# ─────────────────────────────────────────────────────────────────────────────

logger.info("Loading vehicle detection model...")
vehicle_model = YOLO(VEHICLE_MODEL)

helmet_model = None
if Path(HELMET_MODEL).exists():
    logger.info("Loading helmet detection model...")
    helmet_model = YOLO(HELMET_MODEL)
else:
    logger.warning(
        f"Helmet model not found at {HELMET_MODEL}. "
        "Violation detection will be skipped. "
        "Train or download the model and set HELMET_MODEL_PATH."
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONGESTION CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class CongestionClassifier:
    """
    Wraps raw vehicle counts into Low/Medium/High labels.
    Uses a rolling window (mode vote) to smooth out noisy single frames.

    Example:
        classifier = CongestionClassifier(window=5)
        result = classifier.update(vehicle_count=12)
        # → {"count": 12, "level": "Medium", "buffer": [8,10,12,11,12]}
    """

    def __init__(self, window: int = SMOOTHING_WINDOW):
        self.buffer = deque(maxlen=window)

    def _raw_level(self, count: int) -> str:
        if count < CONGESTION_LOW_MAX:
            return "Low"
        elif count < CONGESTION_MEDIUM_MAX:
            return "Medium"
        else:
            return "High"

    def update(self, count: int) -> dict:
        self.buffer.append(count)
        frame_levels  = [self._raw_level(c) for c in self.buffer]
        smoothed      = mode(frame_levels)  # majority vote across window

        return {
            "count":  count,
            "level":  smoothed,
            "buffer": list(self.buffer),
        }

    def reset(self):
        self.buffer.clear()


# ─────────────────────────────────────────────────────────────────────────────
# ROI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_roi_mask(frame_shape: tuple, roi_points: list) -> np.ndarray:
    """Build a binary mask for the ROI polygon."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts  = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def point_in_roi(cx: float, cy: float, roi_mask: np.ndarray) -> bool:
    """Check if a bounding box centre point falls inside the ROI."""
    h, w = roi_mask.shape
    x = max(0, min(int(cx), w - 1))
    y = max(0, min(int(cy), h - 1))
    return roi_mask[y, x] > 0


# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_vehicles(frame: np.ndarray, roi_mask: np.ndarray) -> tuple[int, list]:
    """
    Run YOLOv8 on a frame and count vehicles inside the ROI.

    Returns:
        vehicle_count  — integer count of vehicles in ROI
        detections     — list of dicts for storing raw box data
    """
    results     = vehicle_model(frame, verbose=False)[0]
    count       = 0
    detections  = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)

        if cls_id not in VEHICLE_CLASS_IDS:
            continue
        if conf < VEHICLE_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if not point_in_roi(cx, cy, roi_mask):
            continue

        count += 1
        detections.append({
            "class": VEHICLE_CLASS_IDS[cls_id],
            "conf":  round(conf, 3),
            "bbox":  [round(x1), round(y1), round(x2), round(y2)],
        })

    return count, detections


# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_helmet_violations(
    frame:        np.ndarray,
    vehicle_boxes: list,
    job_id:       str,
    frame_number: int,
    timestamp_sec: float,
) -> list:
    """
    Run the helmet model directly on the FULL frame — it was trained to
    detect heads/helmets in complete traffic scenes, not pre-cropped
    motorcycle regions. No cropping needed.
    """
    if helmet_model is None:
        return []

    violations = []

    helmet_results = helmet_model(frame, verbose=False)[0]


    for hbox in helmet_results.boxes:
        cls_name = helmet_model.names[int(hbox.cls)]
        conf     = float(hbox.conf)

        if cls_name == "Without Helmet" and conf >= HELMET_CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, hbox.xyxy[0].tolist())

            snapshot_path = _save_snapshot(
                frame, job_id, frame_number, "no_helmet",
                bbox=[x1, y1, x2, y2]
            )

            violations.append({
                "violation_type": "no_helmet",
                "confidence":     round(conf, 4),
                "frame_number":   frame_number,
                "timestamp_sec":  round(timestamp_sec, 2),
                "snapshot_path":  snapshot_path,
                "bbox":           [x1, y1, x2, y2],
            })

    return violations


def _save_snapshot(
    frame:        np.ndarray,
    job_id:       str,
    frame_number: int,
    vtype:        str,
    bbox:         list = None,
) -> str:
    """
    Save a JPEG crop of the violation frame.
    Draws a red bounding box on the crop before saving.
    Returns the file path as a string.
    """
    annotated = frame.copy()

    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated, vtype.replace("_", " ").upper(),
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    filename = f"{job_id[:8]}_{vtype}_f{frame_number:05d}.jpg"
    path     = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _update_job(db: Session, job_id: str, **kwargs):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
        db.commit()


def _flush_detections(db: Session, batch: list):
    """Bulk-insert a batch of Detection objects and clear the list."""
    if batch:
        db.bulk_save_objects(batch)
        db.commit()
        batch.clear()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CELERY TASK
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=1)
def process_video(self, job_id: str, video_path: str, roi_points: list = None):
    """
    Full ML pipeline for a single traffic video.

    Arguments:
        job_id      — UUID string matching a row in the jobs table
        video_path  — absolute path to the uploaded video file
        roi_points  — optional list of [x, y] pairs defining the road zone.
                      Defaults to DEFAULT_ROI if not provided.

    The task updates job.status at each stage so the frontend can poll
    for progress via GET /results/{job_id}.
    """
    db         = SessionLocal()
    classifier = CongestionClassifier()
    

    logger.info(f"[{job_id[:8]}] Starting pipeline for {video_path}")

    try:
        # ── 1. Mark job as processing ─────────────────────────────────────
        _update_job(db, job_id, status="processing")

        # ── 2. Open video ─────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        logger.info(
            f"[{job_id[:8]}] Video: {total_frames} frames, "
            f"{duration_sec:.1f}s, {fps:.1f}fps"
        )

        # ── 3. Build ROI mask (once, reused every frame) ──────────────────
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame from video.")

         # Convert relative ROI to actual pixel coordinates based on real frame size
        h, w = first_frame.shape[:2]
        roi_relative = roi_points or DEFAULT_ROI_RELATIVE
        roi = [(int(x * w), int(y * h)) for x, y in roi_relative]

        roi_mask = build_roi_mask(first_frame.shape, roi)

        # ── 4. Frame loop ─────────────────────────────────────────────────
        frame_number    = 0
        analysed_frames = 0
        detection_batch = []   # buffer for bulk DB insert
        total_violations = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for speed — process every FRAME_SKIP-th frame
            if frame_number % FRAME_SKIP != 0:
                frame_number += 1
                continue

            timestamp_sec = frame_number / fps

            # ── Vehicle detection ──────────────────────────────────────
            vehicle_count, raw_boxes = detect_vehicles(frame, roi_mask)

            # ── Congestion classification ──────────────────────────────
            congestion = classifier.update(vehicle_count)

            # ── Build Detection DB object ──────────────────────────────
            detection_batch.append(models.Detection(
                job_id=job_id,
                frame_number=frame_number,
                timestamp_sec=round(timestamp_sec, 2),
                vehicle_count=vehicle_count,
                congestion_level=congestion["level"],
                raw_detections=json.dumps(raw_boxes),
            ))

            # ── Batch commit to DB every N frames ─────────────────────
            if len(detection_batch) >= DB_BATCH_SIZE:
                _flush_detections(db, detection_batch)
                logger.debug(
                    f"[{job_id[:8]}] Flushed batch at frame {frame_number}"
                )

            # ── Violation detection (only when helmet model is loaded) ──
            violations = detect_helmet_violations(
                frame, raw_boxes, job_id, frame_number, timestamp_sec
            )

            for v in violations:
                db.add(models.Violation(
                    job_id=job_id,
                    violation_type=v["violation_type"],
                    confidence=v["confidence"],
                    frame_number=v["frame_number"],
                    timestamp_sec=v["timestamp_sec"],
                    snapshot_path=v["snapshot_path"],
                    bbox_x1=v["bbox"][0],
                    bbox_y1=v["bbox"][1],
                    bbox_x2=v["bbox"][2],
                    bbox_y2=v["bbox"][3],
                ))
                db.commit()
                total_violations += 1

            frame_number    += 1
            analysed_frames += 1

            # Log progress every 100 analysed frames
            # TEMPORARY DEBUG — log every analysed frame
            pct = (frame_number / total_frames * 100) if total_frames else 0
            logger.info(
                f"[{job_id[:8]}] {pct:.0f}% — "
                f"frame {frame_number}/{total_frames}, "
                f"raw_count: {vehicle_count}, "
                f"classes: {[d['class'] for d in raw_boxes]}, "
                f"level: {congestion['level']}, "
                f"violations so far: {total_violations}"
            )

        # ── 5. Flush any remaining detections ────────────────────────────
        _flush_detections(db, detection_batch)
        cap.release()

        # ── 6. Mark job complete ──────────────────────────────────────────
        _update_job(
            db, job_id,
            status="complete",
            frame_count=analysed_frames,
            duration_sec=round(duration_sec, 2),
            completed_at=datetime.utcnow(),
        )

        logger.info(
            f"[{job_id[:8]}] ✅ Complete — "
            f"{analysed_frames} frames analysed, "
            f"{total_violations} violations found"
        )

        return {
            "job_id":          job_id,
            "status":          "complete",
            "analysed_frames": analysed_frames,
            "duration_sec":    round(duration_sec, 2),
            "total_violations": total_violations,
        }

    except Exception as exc:
        # ── On any error: mark job failed, log full traceback ────────────
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error(f"[{job_id[:8]}] ❌ Failed — {error_msg}")

        try:
            _update_job(db, job_id, status="failed", error_msg=str(exc))
        except Exception:
            pass  # Don't let a DB error hide the original exception

        raise self.retry(exc=exc, countdown=5) if self.request.retries < 1 else exc

    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: ANNOTATED VIDEO EXPORT
# Call this as a separate task after process_video completes
# to produce a labelled output video for the dashboard's video player.
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task
def export_annotated_video(job_id: str, video_path: str, roi_points: list = None):
    """
    Re-processes the video to produce an annotated output MP4 with:
      - Bounding boxes drawn around each detected vehicle
      - Congestion level banner at the top of the frame
      - Violation labels in red

    Output is saved to SNAPSHOT_DIR/{job_id}_annotated.mp4
    This is a nice-to-have — run it after process_video for demo purposes.
    """
    roi_relative = roi_points or DEFAULT_ROI_RELATIVE
    cap  = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open {video_path} for annotation")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(SNAPSHOT_DIR, f"{job_id}_annotated.mp4")
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    classifier = CongestionClassifier()
    roi_mask   = None
    frame_num  = 0

    LEVEL_COLORS = {
        "Low":    (34, 197, 94),   # green (BGR)
        "Medium": (245, 158, 11),  # amber
        "High":   (239, 68, 68),   # red
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi_mask is None:
            roi_mask = build_roi_mask(frame.shape, roi)

        if frame_num % FRAME_SKIP == 0:
            count, boxes = detect_vehicles(frame, roi_mask)
            result = classifier.update(count)
            level  = result["level"]
            color  = LEVEL_COLORS[level]

            # Draw vehicle bounding boxes
            for det in boxes:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, det["class"],
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

            # Draw congestion banner across the top
            cv2.rectangle(frame, (0, 0), (width, 36), (0, 0, 0), -1)
            banner = (
                f"Congestion: {level}   "
                f"Vehicles: {count}   "
                f"Time: {frame_num/fps:.1f}s"
            )
            cv2.putText(
                frame, banner,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

            # Draw ROI polygon outline
            pts = np.array(roi, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True,
                          color=(255, 255, 0), thickness=1)

        writer.write(frame)
        frame_num += 1

    cap.release()
    writer.release()
    logger.info(f"[{job_id[:8]}] Annotated video saved to {out_path}")
    return out_path