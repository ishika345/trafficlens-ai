import requests

BASE_URL = "http://localhost:8000"

def upload_video(file_bytes, filename):
    r = requests.post(
        f"{BASE_URL}/analyze/video",
        files={"file": (filename, file_bytes, "video/mp4")}
    )
    r.raise_for_status()
    return r.json()

def get_job_status(job_id):
    r = requests.get(f"{BASE_URL}/results/{job_id}")
    r.raise_for_status()
    return r.json()

def get_detections(job_id):
    r = requests.get(f"{BASE_URL}/detections/{job_id}")
    r.raise_for_status()
    return r.json()

def get_violations(job_id, violation_type=None):
    params = {}
    if violation_type:
        params["violation_type"] = violation_type
    r = requests.get(f"{BASE_URL}/violations/{job_id}", params=params)
    r.raise_for_status()
    return r.json()

def get_summary():
    r = requests.get(f"{BASE_URL}/stats/summary")
    r.raise_for_status()
    return r.json()