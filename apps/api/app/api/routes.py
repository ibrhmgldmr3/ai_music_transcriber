from pathlib import PurePath
from urllib.parse import quote

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Project, ProjectStatus, get_db
from app.schemas import NotesUpdate, ProjectOut, TranscriptionOut
from app.services.storage import (
    InvalidUpload,
    clean_text,
    delete_file,
    media_type,
    sanitize_filename,
    save_upload,
    stored_file,
)
from app.services.transcription import job_is_stale, run_transcription
from music_core.analysis import Key, estimate_key
from music_core.midi import notes_to_midi_bytes
from music_core.musicxml import notes_to_musicxml
from music_core.notes import Note
from music_core.tab import DEFAULT_NUM_FRETS, STANDARD_TUNING, assign_tab, tab_to_ascii

router = APIRouter(prefix="/projects", tags=["projects"])

ACTIVE_STATUSES = (ProjectStatus.pending, ProjectStatus.processing)


def _get_project(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


def _require_transcription(project: Project) -> list[Note]:
    if project.status != ProjectStatus.completed:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Transcription is not ready (status: {project.status.value})"
        )
    return [Note.from_dict(n) for n in project.notes or []]


def _enqueue(project_id: str, background_tasks: BackgroundTasks) -> None:
    if settings.use_celery:
        from app.workers.tasks import transcribe_project

        transcribe_project.delay(project_id)
    else:
        background_tasks.add_task(run_transcription, project_id)


def _attachment(filename: str) -> dict[str, str]:
    # RFC 5987 so non-ASCII project names (ş, ğ, ...) survive the header.
    return {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}


def _transcription_out(project: Project) -> TranscriptionOut:
    notes = project.notes or []
    confidences = [n["confidence"] for n in notes if n.get("confidence") is not None]
    return TranscriptionOut(
        project_id=project.id,
        tempo=project.tempo,
        tuning=project.tuning or list(STANDARD_TUNING),
        mean_confidence=sum(confidences) / len(confidences) if confidences else None,
        notes=notes,
        beats_per_measure=project.beats_per_measure or 4,
        key=project.key_name,
        downbeat=project.downbeat,
    )


def _chosen_key(project: Project) -> Key | None:
    return Key.parse(project.key_name) if project.key_name else None


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str | None = Form(None),
    separate_guitar: bool = Form(False),
    db: Session = Depends(get_db),
) -> Project:
    """Upload a recording and start transcribing it.

    ``separate_guitar`` (song mode) isolates the guitar from a band mix first.
    """
    try:
        audio_path = save_upload(file)
    except InvalidUpload as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    filename = sanitize_filename(file.filename) or audio_path.name
    project = Project(
        name=clean_text(name) or clean_text(PurePath(filename).stem) or "Untitled",
        filename=filename,
        audio_path=str(audio_path),
        separate_guitar=separate_guitar,
    )
    try:
        db.add(project)
        db.commit()
        db.refresh(project)
    except Exception:
        db.rollback()
        delete_file(audio_path)  # don't leave an orphaned upload behind
        raise
    _enqueue(project.id, background_tasks)
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[Project]:
    query = select(Project).order_by(Project.created_at.desc()).limit(limit).offset(offset)
    return list(db.scalars(query))


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db)) -> Project:
    return _get_project(db, project_id)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> Response:
    # A running transcription notices the deletion and discards its result.
    project = _get_project(db, project_id)
    delete_file(project.audio_path)
    db.delete(project)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{project_id}/retranscribe", response_model=ProjectOut)
def retranscribe(
    project_id: str,
    background_tasks: BackgroundTasks,
    separate_guitar: bool | None = Query(None, description="switch song mode on/off"),
    db: Session = Depends(get_db),
) -> Project:
    project = _get_project(db, project_id)
    # A job that stopped reporting (crashed worker, lost queue) may be restarted.
    if project.status in ACTIVE_STATUSES and not job_is_stale(project):
        raise HTTPException(status.HTTP_409_CONFLICT, "Transcription already in progress")
    if separate_guitar is not None:
        project.separate_guitar = separate_guitar
    project.status = ProjectStatus.pending
    project.error = None
    db.commit()
    _enqueue(project.id, background_tasks)
    return project


@router.get("/{project_id}/audio")
def get_audio(project_id: str, db: Session = Depends(get_db)) -> FileResponse:
    project = _get_project(db, project_id)
    path = stored_file(project.audio_path)
    if path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audio file missing")
    return FileResponse(path, media_type=media_type(path), filename=project.filename)


@router.get("/{project_id}/transcription", response_model=TranscriptionOut)
def get_transcription(project_id: str, db: Session = Depends(get_db)) -> TranscriptionOut:
    project = _get_project(db, project_id)
    _require_transcription(project)
    return _transcription_out(project)


@router.put("/{project_id}/notes", response_model=TranscriptionOut)
def update_notes(
    project_id: str, payload: NotesUpdate, db: Session = Depends(get_db)
) -> TranscriptionOut:
    """Replace the notes with the editor's version.

    Notes without a (consistent) string/fret get one from the fingering optimizer;
    positions the user chose are kept.
    """
    project = _get_project(db, project_id)
    _require_transcription(project)
    notes = [Note.from_dict(note.model_dump()) for note in payload.notes]
    highest_fret = max((n.fret for n in notes if n.fret is not None), default=0)
    notes = assign_tab(
        notes,
        project.tuning or STANDARD_TUNING,
        num_frets=max(DEFAULT_NUM_FRETS, highest_fret),  # never discard a chosen high fret
    )
    project.notes = [note.to_dict() for note in notes]
    project.edited = True
    if payload.tempo is not None:
        project.tempo = round(payload.tempo, 2)
    if payload.beats_per_measure is not None:
        project.beats_per_measure = payload.beats_per_measure
    # An explicit null returns key / bar grid to the estimate; omitted keeps them.
    if "key" in payload.model_fields_set:
        project.key_name = payload.key
    if "downbeat" in payload.model_fields_set:
        project.downbeat = payload.downbeat
    db.commit()
    return _transcription_out(project)


@router.get("/{project_id}/midi")
def export_midi(project_id: str, db: Session = Depends(get_db)) -> Response:
    project = _get_project(db, project_id)
    notes = _require_transcription(project)
    data = notes_to_midi_bytes(
        notes,
        tempo=project.tempo or 120.0,
        key=_chosen_key(project) or estimate_key(notes),
        beats_per_measure=project.beats_per_measure or 4,
    )
    return Response(data, media_type="audio/midi", headers=_attachment(f"{project.name}.mid"))


@router.get("/{project_id}/musicxml")
def export_musicxml(project_id: str, db: Session = Depends(get_db)) -> Response:
    """Notation + tablature for MuseScore, Guitar Pro and other score editors."""
    project = _get_project(db, project_id)
    notes = _require_transcription(project)
    data = notes_to_musicxml(
        notes,
        tempo=project.tempo or 120.0,
        tuning=project.tuning or STANDARD_TUNING,
        title=project.name,
        beats_per_measure=project.beats_per_measure or 4,
        key=_chosen_key(project),
        downbeat=project.downbeat,
    )
    return Response(
        data,
        media_type="application/vnd.recordare.musicxml+xml",
        headers=_attachment(f"{project.name}.musicxml"),
    )


@router.get("/{project_id}/tab", response_class=PlainTextResponse)
def export_tab(project_id: str, db: Session = Depends(get_db)) -> PlainTextResponse:
    project = _get_project(db, project_id)
    notes = _require_transcription(project)
    text = tab_to_ascii(notes, project.tuning or STANDARD_TUNING)
    return PlainTextResponse(text, headers=_attachment(f"{project.name}.txt"))
