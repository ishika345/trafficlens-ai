from celery import Celery

celery = Celery('worker', broker='redis://localhost:6379/0')

@celery.task
def process_video(video_path: str):
    return {"status": "processed", "path": video_path}