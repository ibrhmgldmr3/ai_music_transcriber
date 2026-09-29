import hashlib
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.models import Project, ProjectStatus, SessionLocal
from music_core.tab import open_strings

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
def model_version() -> str:
    """The installed models, e.g. ``guitar_v8@1a2b3c4d+guitar_v7@5e6f7a8b``.

    Checkpoint folder plus a fingerprint of the file (size and modification time), so a
    retrained or replaced model gets a new version. Cached like the predictor: both
    describe what this process loaded when it first needed them.
    """
    paths = [settings.model_checkpoint]
    if settings.model_tab_checkpoint:
        paths.append(settings.model_tab_checkpoint)
    return "+".join(_fingerprint(path) for path in paths)


def voice_version() -> str:
    """The voice method's version (``ml.inference.voice``); it has no model file."""
    from ml.inference.voice import VOICE_VERSION

    return VOICE_VERSION


def _fingerprint(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return f"{path.parent.name}@missing"
    digest = hashlib.sha1(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:8]
    return f"{path.parent.name}@{digest}"


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
        voice = project is not None and project.source == "voice"
        tuning = open_strings(project.tuning_name, project.capo) if project is not None else None
    started = _update(
        project_id, status=ProjectStatus.processing, error=None, progress=0.0, stage="loading"
    )
    if audio_path is None or not started:
        logger.warning("Project %s disappeared before transcription", project_id)
        return

    try:
        _check_audio(audio_path)
        with _model_lock:
            separator = _load_separator() if separate else None
            if voice:
                from ml.inference.voice import transcribe_voice

                result = transcribe_voice(
                    audio_path,
                    tuning=tuning,
                    separator=partial(separator, stem="vocals") if separator else None,
                    progress=_progress_reporter(project_id),
                )
                version = voice_version()
            else:
                result = _load_predictor().transcribe(
                    audio_path,
                    separator=separator,
                    tuning=tuning,
                    progress=_progress_reporter(project_id),
                )
                version = model_version()
    except TranscriptionError as exc:
        logger.warning("Transcription of %s rejected: %s", project_id, exc)
        _update(project_id, status=ProjectStatus.failed, error=str(exc), progress=None, stage=None)
        return
    except Exception:  # report a generic failure; details stay in the logs
        logger.exception("Transcription failed for %s", project_id)
        _update(
            project_id, status=ProjectStatus.failed, error=INTERNAL_ERROR, progress=None, stage=None
        )
        return

    stored = _update(
        project_id,
        notes=[note.to_dict() for note in result.notes],
        duration=result.duration,
        tempo=result.tempo,
        tuning=result.tuning,
        transpose=result.transpose,
        model_version=version,
        edited=False,
        progress=None,
        stage=None,
        status=ProjectStatus.completed,
    )
    if stored:
        logger.info("Transcribed %s: %d notes", project_id, len(result.notes))
    else:
        logger.info("Project %s was deleted during transcription", project_id)


def _progress_reporter(project_id: str, interval: float = 0.5):
    """Write progress to the project at most every ``interval`` seconds (and on every new
    stage). Each write also refreshes ``updated_at``, so a long job never looks stale."""
    last = {"time": 0.0, "stage": None}

    def report(fraction: float, stage: str) -> None:
        now = time.monotonic()
        if stage == last["stage"] and now - last["time"] < interval:
            return
        last.update(time=now, stage=stage)
        _update(project_id, progress=round(fraction, 3), stage=stage)

    return report


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
