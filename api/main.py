from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid, shutil, os
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from api.schemas import JobResponse, DetectionRecord, ViolationRecord, SummaryStats
from database.database import get_db, create_tables
from database import models

app = FastAPI(title="TrafficLens AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def startup():
    create_tables()


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "TrafficLens AI"}


@app.post("/analyze/video", response_model=JobResponse)
async def analyze_video(file: UploadFile = File(...)):
    from worker.tasks import process_video

    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Only MP4, AVI, MOV files accepted")

    job_id    = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}.mp4")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db: Session = next(get_db())
    job = models.Job(
        id=job_id,
        filename=file.filename,
        status="queued",
        video_path=save_path,
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()

    process_video.delay(job_id, save_path)

    return JobResponse(
        job_id=job_id,
        status="queued",
        filename=file.filename,
        created_at=job.created_at
    )


@app.get("/results/{job_id}", response_model=JobResponse)
def get_results(job_id: str):
    db: Session = next(get_db())
    job = db.query(models.Job).filter(models.Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobResponse(
        job_id=job.id,
        status=job.status,
        filename=job.filename,
        created_at=job.created_at
    )


@app.get("/detections/{job_id}", response_model=List[DetectionRecord])
def get_detections(job_id: str, limit: int = 500):
    db: Session = next(get_db())

    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    detections = (
        db.query(models.Detection)
        .filter(models.Detection.job_id == job_id)
        .order_by(models.Detection.frame_number)
        .limit(limit)
        .all()
    )

    return [
        DetectionRecord(
            frame_number=d.frame_number,
            timestamp_sec=d.timestamp_sec,
            vehicle_count=d.vehicle_count,
            congestion_level=d.congestion_level
        )
        for d in detections
    ]


@app.get("/violations/{job_id}", response_model=List[ViolationRecord])
def get_violations(job_id: str, violation_type: str = None):
    db: Session = next(get_db())

    query = db.query(models.Violation).filter(models.Violation.job_id == job_id)
    if violation_type:
        query = query.filter(models.Violation.violation_type == violation_type)

    violations = query.order_by(models.Violation.frame_number).all()

    return [
        ViolationRecord(
            violation_type=v.violation_type,
            confidence=round(v.confidence, 3),
            frame_number=v.frame_number,
            timestamp_sec=v.timestamp_sec,
            snapshot_path=v.snapshot_path
        )
        for v in violations
    ]


@app.get("/stats/summary", response_model=SummaryStats)
def get_summary():
    db: Session = next(get_db())

    total_videos     = db.query(models.Job).filter(models.Job.status == "complete").count()
    total_violations = db.query(models.Violation).count()

    avg_result = db.query(func.avg(models.Detection.vehicle_count)).scalar()
    avg_count  = round(float(avg_result or 0), 1)

    congestion_counts = (
        db.query(models.Detection.congestion_level, func.count())
        .group_by(models.Detection.congestion_level)
        .all()
    )
    congestion_dist = {level: count for level, count in congestion_counts}
    peak = max(congestion_dist, key=congestion_dist.get) if congestion_dist else "Low"

    violation_counts = (
        db.query(models.Violation.violation_type, func.count())
        .group_by(models.Violation.violation_type)
        .all()
    )
    violations_by_type = {vtype: count for vtype, count in violation_counts}

    return SummaryStats(
        total_videos_processed=total_videos,
        total_violations=total_violations,
        peak_congestion_level=peak,
        avg_vehicle_count=avg_count,
        violations_by_type=violations_by_type,
        congestion_distribution=congestion_dist
    )