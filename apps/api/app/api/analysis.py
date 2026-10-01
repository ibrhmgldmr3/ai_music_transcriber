from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.models import Project, get_db
from app.schemas import AnalysisOut, AnalysisRequest, ChordOut, KeyOut, RenderRequest
from app.schemas.project import BarOut, KeySpanOut, RhythmOut, VoicingOut, VoicingRequest
from app.services.storage import clean_text
from music_core.analysis import Analysis, Chord, Key, KeySpan, analyze
from music_core.guitar_chords import ChordSymbol, voicing
from music_core.musicxml import first_bar_line, notes_to_musicxml
from music_core.notes import Note
from music_core.tab import open_strings

router = APIRouter(prefix="/analysis", tags=["analysis"])
render_router = APIRouter(prefix="/render", tags=["analysis"])
chords_router = APIRouter(prefix="/chords", tags=["analysis"])


def _key_out(key: Key | None) -> KeyOut | None:
    if key is None:
        return None
    return KeyOut(name=key.name, tonic=key.tonic, mode=key.mode, fifths=key.fifths)


def project_context(project: Project, beat_grid: bool | None = None) -> dict[str, Any]:
    """``analyze`` arguments from what a project's transcription found in the audio: the
    tracked beats (when its grid follows them, or ``beat_grid`` says so), the recognized
    chords to fuse with the notes, a song's keys and strums, and the recording's length."""
    stored = project.beats or {}
    follow = project.beat_grid if beat_grid is None else beat_grid
    context: dict[str, Any] = {"end": project.duration}
    if follow and len(stored.get("beats", [])) >= 2:
        context["beats"] = stored["beats"]
        context["downbeats"] = stored.get("downbeats", [])
        context["end"] = max(project.duration or 0.0, stored["beats"][-1])
    if stored.get("chord_scores"):
        context["chord_scores"] = stored["chord_scores"]
    if stored.get("strums") is not None:
        context["strums"] = [(float(t), float(s)) for t, s in stored["strums"]]
    if project.keys:
        context["keys"] = [KeySpan(k["start"], k["end"], Key.parse(k["key"])) for k in project.keys]
    return context


def song_chords(project: Project) -> list[Chord] | None:
    """A song project's chords as the exports write them; None for other projects."""
    if project.chords is None:
        return None
    chords = []
    for c in project.chords:
        symbol = ChordSymbol.parse(c["label"])
        if symbol is not None:
            chords.append(Chord(c["start"], c["end"], symbol.root, symbol.quality, symbol.bass))
    return chords


def analysis_out(analysis: Analysis, notes: list[Note], end: float) -> AnalysisOut:
    grid = analysis.grid
    origin = first_bar_line(notes, grid)
    n = grid.beats_per_measure
    inside = [(i, t) for i, t in enumerate(grid.beats) if -1e-6 <= t <= end + 1e-6]
    rhythm = analysis.rhythm
    return AnalysisOut(
        tempo=analysis.tempo,
        beats_per_measure=analysis.beats_per_measure,
        downbeat=analysis.downbeat,
        key=_key_out(analysis.key),
        estimated_key=_key_out(analysis.estimated_key),
        chords=[
            ChordOut(start=c.start, end=c.end, label=c.label(analysis.key)) for c in analysis.chords
        ],
        tracked=analysis.tracked,
        beats=[round(t, 4) for _, t in inside],
        bars=[
            BarOut(time=round(t, 4), number=(i - origin) // n + 1)
            for i, t in inside
            if grid.is_bar_line(i)
        ],
        keys=[KeySpanOut(start=s.start, end=s.end, key=s.key.name) for s in analysis.keys],
        rhythm=RhythmOut(
            per_beat=rhythm.per_beat,
            pattern=list(rhythm.pattern),
            text=rhythm.text(),
            bars=[list(b.hits) for b in rhythm.bars],
            bar_times=[round(b.start, 4) for b in rhythm.bars],
        )
        if rhythm is not None and any(rhythm.pattern)
        else None,
    )


def _context(payload: AnalysisRequest, db: Session) -> dict[str, Any]:
    if payload.project_id is None:
        return {}
    project = db.get(Project, payload.project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project_context(project, payload.beat_grid)


@router.post("", response_model=AnalysisOut)
def analyze_notes(payload: AnalysisRequest, db: Session = Depends(get_db)) -> AnalysisOut:
    """Key, bar grid, chord symbols and strumming pattern of the given notes.

    Works on unsaved edits, so the editor can show them as the notes change; with a
    ``project_id`` the beats, chords, keys and strums found in its recording join in.
    Exports run the same analysis on the stored notes.
    """
    notes = [Note.from_dict(note.model_dump()) for note in payload.notes]
    context = _context(payload, db)
    end = max(payload.duration or 0.0, context.pop("end", None) or 0.0)
    analysis = analyze(
        notes,
        payload.tempo,
        payload.beats_per_measure,
        key=Key.parse(payload.key) if payload.key else None,
        downbeat=payload.downbeat,
        end=end,
        **context,
    )
    return analysis_out(analysis, notes, max(end, max((n.end for n in notes), default=0.0)))


@render_router.post("/musicxml")
def render_musicxml(payload: RenderRequest, db: Session = Depends(get_db)) -> Response:
    """MusicXML of the given notes, like the project export but for unsaved edits."""
    notes = [Note.from_dict(note.model_dump()) for note in payload.notes]
    context = _context(payload, db)
    project = db.get(Project, payload.project_id) if payload.project_id else None
    data = notes_to_musicxml(
        notes,
        tempo=payload.tempo or 120.0,
        tuning=payload.tuning,
        title=clean_text(payload.title) or "Transcription",
        beats_per_measure=payload.beats_per_measure,
        key=Key.parse(payload.key) if payload.key else None,
        downbeat=payload.downbeat,
        capo=payload.capo,
        chords=song_chords(project) if project is not None else None,
        beats=context.get("beats"),
        downbeats=context.get("downbeats"),
        keys=context.get("keys"),
        chord_scores=context.get("chord_scores"),
    )
    return Response(data, media_type="application/vnd.recordare.musicxml+xml")


@chords_router.post("/voicings", response_model=list[VoicingOut])
def chord_voicings(payload: VoicingRequest) -> list[VoicingOut]:
    """Names and fingerings of chord labels, as shapes from a capo in a tuning."""
    key = Key.parse(payload.key) if payload.key else None
    tuning = open_strings(payload.tuning_name)
    out = []
    for label in payload.labels:
        try:
            chord = ChordSymbol.parse(label)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown chord label"
            ) from exc
        if chord is None:
            out.append(
                VoicingOut(label=label, name="N.C.", shape="N.C.", frets=None, difficulty=None)
            )
            continue
        shape = chord.transposed(-payload.capo)
        v = voicing(shape, tuning)
        out.append(
            VoicingOut(
                label=label,
                name=chord.name(key),
                shape=shape.name(None if payload.capo else key),
                frets=list(v.frets) if v else None,
                difficulty=round(v.difficulty, 2) if v else None,
            )
        )
    return out
