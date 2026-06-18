from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class JobResponse(BaseModel):
    job_id: str
    status: str
    filename: Optional[str] = None
    created_at: Optional[datetime] = None


class DetectionRecord(BaseModel):
    frame_number: int
    timestamp_sec: float
    vehicle_count: int
    congestion_level: str


class ViolationRecord(BaseModel):
    violation_type: str
    confidence: float
    frame_number: int
    timestamp_sec: float
    snapshot_path: Optional[str] = None


class SummaryStats(BaseModel):
    total_videos_processed: int
    total_violations: int
    peak_congestion_level: str
    avg_vehicle_count: float
    violations_by_type: dict
    congestion_distribution: dict