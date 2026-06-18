from sqlalchemy import (
    Column, String, Integer, Float, 
    DateTime, ForeignKey, Text, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Job(Base):
    """
    One row per uploaded video. Tracks processing state.
    All other tables foreign-key back to this.
    """
    __tablename__ = "jobs"

    id           = Column(String(36), primary_key=True)   # UUID
    filename     = Column(String(255), nullable=False)
    status       = Column(String(20),  nullable=False, default="queued")
    # status values: "queued" | "processing" | "complete" | "failed"

    camera_id    = Column(String(36), ForeignKey("cameras.id"), nullable=True)
    video_path   = Column(Text, nullable=True)   # path on disk
    error_msg    = Column(Text, nullable=True)   # populated if status = "failed"

    frame_count  = Column(Integer, nullable=True)   # filled after processing
    duration_sec = Column(Float,   nullable=True)

    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships — lets you do job.detections, job.violations in Python
    detections = relationship("Detection", back_populates="job", cascade="all, delete-orphan")
    violations = relationship("Violation", back_populates="job", cascade="all, delete-orphan")
    camera     = relationship("Camera",    back_populates="jobs")


class Detection(Base):
    """
    One row per analysed frame in a video.
    Stores the vehicle count and congestion level at that moment.
    """
    __tablename__ = "detections"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    job_id            = Column(String(36), ForeignKey("jobs.id"), nullable=False)

    frame_number      = Column(Integer, nullable=False)
    timestamp_sec     = Column(Float,   nullable=False)   # seconds into the video
    vehicle_count     = Column(Integer, nullable=False)
    congestion_level  = Column(String(10), nullable=False)  # "Low"|"Medium"|"High"
    raw_detections    = Column(Text, nullable=True)
    # raw_detections stores the YOLO JSON: 
    # '[{"class":"car","conf":0.91,"bbox":[x1,y1,x2,y2]}, ...]'

    job = relationship("Job", back_populates="detections")

    # Index on job_id — you'll query "all detections for job X" constantly
    __table_args__ = (
        Index("ix_detections_job_id", "job_id"),
        Index("ix_detections_job_frame", "job_id", "frame_number"),
    )


class Violation(Base):
    """
    One row per detected violation event (no helmet, signal jump, etc.)
    Linked to the specific frame where it occurred.
    """
    __tablename__ = "violations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    job_id          = Column(String(36), ForeignKey("jobs.id"), nullable=False)

    violation_type  = Column(String(50), nullable=False)
    # types: "no_helmet" | "signal_jump" | "wrong_way"

    confidence      = Column(Float,   nullable=False)
    frame_number    = Column(Integer, nullable=False)
    timestamp_sec   = Column(Float,   nullable=False)
    snapshot_path   = Column(Text, nullable=True)
    # snapshot_path: path to the saved JPEG crop of the violation frame

    bbox_x1 = Column(Float, nullable=True)   # bounding box of the violating vehicle
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)

    detected_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="violations")

    __table_args__ = (
        Index("ix_violations_job_id", "job_id"),
        Index("ix_violations_type",   "violation_type"),
    )


class Camera(Base):
    """
    Optional but impressive — maps a job to a real Bengaluru location.
    Lets you show a city map with congestion overlaid per junction.
    """
    __tablename__ = "cameras"

    id          = Column(String(36), primary_key=True)  # UUID
    name        = Column(String(255), nullable=False)
    # e.g. "Silk Board Junction – Camera 3"

    location    = Column(String(255), nullable=True)
    # e.g. "Silk Board Junction, Bengaluru"

    latitude    = Column(Float, nullable=True)   # 12.9177
    longitude   = Column(Float, nullable=True)   # 77.6225

    zone        = Column(String(100), nullable=True)
    # e.g. "South Bengaluru", "Electronic City", "Whitefield"

    is_active   = Column(String(5), default="true")
    created_at  = Column(DateTime, default=datetime.utcnow)

    jobs = relationship("Job", back_populates="camera")