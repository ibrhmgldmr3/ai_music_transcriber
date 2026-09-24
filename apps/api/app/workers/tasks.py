from app.services.transcription import run_transcription
from app.workers.celery_app import celery_app


@celery_app.task(name="transcribe_project")
def transcribe_project(project_id: str) -> None:
    run_transcription(project_id)
