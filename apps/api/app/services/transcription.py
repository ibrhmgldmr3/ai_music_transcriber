import logging
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.models import Project, ProjectStatus, SessionLocal

logger = logging.getLogger(__name__)

# One model per process, used by one transcription at a time: in-process jobs run in
# FastAPI's thread pool and would otherwise load/run the model concurrently.
_model_lock = threading.Lock()

INTERNAL_ERROR = "Transcription failed because of a server error. Check the server logs."


class TranscriptionError(Exception):
    """A failure whose message is safe to show to users."""


@lru_cache(maxsize=1)
def get_predictor():
    # Imported lazily: torch is heavy and only needed where transcriptions run.
    from ml.inference.predict import Predictor

    return Predictor.from_checkpoint(
        settings.model_checkpoint,
        device=settings.model_device,
        tab_checkpoint=settings.model_tab_checkpoint,
    )


@lru_cache(maxsize=1)
def get_separator():
    from ml.inference.separation import GuitarSeparator

    return GuitarSeparator(device=settings.model_device)


def _load_separator():
    try:
        return get_separator()
    except ImportError as exc:
        logger.error("Song mode unavailable: %s", exc)
        raise TranscriptionError(
            "Song mode needs the 'demucs' package on the server (pip install demucs)."
        ) from exc


def _load_predictor():
    try:
        return get_predictor()
    except FileNotFoundError as exc:
        logger.error("Model checkpoint missing: %s", exc)
        raise TranscriptionError(
            "No trained model is installed on the server yet (see scripts/train.py)."
        ) from exc


def _check_audio(path: str) -> None:
    import librosa

    try:
        seconds = librosa.get_duration(path=path)
    except Exception as exc:
        raise TranscriptionError("The audio file could not be decoded.") from exc
    if seconds <= 0:
        raise TranscriptionError("The recording is empty.")
    if seconds > settings.max_audio_minutes * 60:
        raise TranscriptionError(
            f"The recording is {seconds / 60:.1f} minutes long; "
            f"the limit is {settings.max_audio_minutes:g} minutes."
        )


def _update(project_id: str, **fields: Any) -> bool:
    """Apply fields in a short-lived session. False if the project no longer exists."""
    with SessionLocal() as db:
        project = db.get(Project, project_id)
        if project is None:
            return False
        for key, value in fields.items():
            setattr(project, key, value)
        db.commit()
        return True


def run_transcription(project_id: str) -> None:
    """Transcribe a project's audio and store the note events. Safe to run in a worker.

    No database session is held during inference, and a project deleted meanwhile is
    simply skipped.
    """
    with SessionLocal() as db:
        project = db.get(Project, project_id)
        audio_path = project.audio_path if project is not None else None
        separate = bool(project.separate_guitar) if project is not None else False
    if audio_path is None or not _update(project_id, status=ProjectStatus.processing, error=None):
        logger.warning("Project %s disappeared before transcription", project_id)
        return

    try:
        _check_audio(audio_path)
        with _model_lock:
            separator = _load_separator() if separate else None
            result = _load_predictor().transcribe(audio_path, separator=separator)
    except TranscriptionError as exc:
        logger.warning("Transcription of %s rejected: %s", project_id, exc)
        _update(project_id, status=ProjectStatus.failed, error=str(exc))
        return
    except Exception:  # report a generic failure; details stay in the logs
        logger.exception("Transcription failed for %s", project_id)
        _update(project_id, status=ProjectStatus.failed, error=INTERNAL_ERROR)
        return

    stored = _update(
        project_id,
        notes=[note.to_dict() for note in result.notes],
        duration=result.duration,
        tempo=result.tempo,
        tuning=result.tuning,
        status=ProjectStatus.completed,
    )
    if stored:
        logger.info("Transcribed %s: %d notes", project_id, len(result.notes))
    else:
        logger.info("Project %s was deleted during transcription", project_id)


def job_is_stale(project: Project, now: datetime | None = None) -> bool:
    """True if a pending/processing job has not been updated within the job timeout."""
    updated = project.updated_at
    if updated.tzinfo is None:  # SQLite returns naive datetimes
        updated = updated.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return now - updated > timedelta(minutes=settings.job_timeout_minutes)


def fail_interrupted_jobs() -> int:
    """Mark jobs left pending/processing by a previous (in-process) run as failed."""
    active = (ProjectStatus.pending, ProjectStatus.processing)
    with SessionLocal() as db:
        projects = list(db.scalars(select(Project).where(Project.status.in_(active))))
        for project in projects:
            project.status = ProjectStatus.failed
            project.error = "Interrupted by a server restart. Start the transcription again."
        db.commit()
        return len(projects)
