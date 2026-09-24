"""Celery application.

celery -A app.workers.celery_app:celery_app worker --loglevel=info --concurrency=1
"""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "music_transcriber",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # transcriptions are long; take one at a time
)
