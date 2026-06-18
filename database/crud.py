from sqlalchemy.orm import Session
from sqlalchemy import func
from database import models
from datetime import datetime
import json


def create_job(db: Session, job_id: str, filename: str,
               video_path: str, camera_id: str = None):
    job = models.Job(
        id=job_id,
        filename=filename,
        status="queued",
        video_path=video_path,
        camera_id=camera_id
    )
    db.add(job)
    db.commit()
    return job


def update_job_status(db: Session, job_id: str, status: str,
                      frame_count: int = None, duration_sec: float = None,
                      error_msg: str = None):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if job:
        job.status = status
        if frame_count:   job.frame_count  = frame_count
        if duration_sec:  job.duration_sec = duration_sec
        if error_msg:     job.error_msg    = error_msg
        if status == "complete":
            job.completed_at = datetime.utcnow()
        db.commit()


def save_detection(db: Session, job_id: str, frame_number: int,
                   timestamp_sec: float, vehicle_count: int,
                   congestion_level: str, raw_boxes: list = None):
    detection = models.Detection(
        job_id=job_id,
        frame_number=frame_number,
        timestamp_sec=round(timestamp_sec, 2),
        vehicle_count=vehicle_count,
        congestion_level=congestion_level,
        raw_detections=json.dumps(raw_boxes) if raw_boxes else None
    )
    db.add(detection)


def save_violation(db: Session, job_id: str, violation_type: str,
                   confidence: float, frame_number: int,
                   timestamp_sec: float, bbox: list = None,
                   snapshot_path: str = None):
    violation = models.Violation(
        job_id=job_id,
        violation_type=violation_type,
        confidence=round(confidence, 4),
        frame_number=frame_number,
        timestamp_sec=round(timestamp_sec, 2),
        snapshot_path=snapshot_path,
        bbox_x1=bbox[0] if bbox else None,
        bbox_y1=bbox[1] if bbox else None,
        bbox_x2=bbox[2] if bbox else None,
        bbox_y2=bbox[3] if bbox else None,
    )
    db.add(violation)
    db.commit()


def get_congestion_distribution(db: Session, job_id: str):
    rows = (
        db.query(models.Detection.congestion_level, func.count())
        .filter(models.Detection.job_id == job_id)
        .group_by(models.Detection.congestion_level)
        .all()
    )
    return {level: count for level, count in rows}