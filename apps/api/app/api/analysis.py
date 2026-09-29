from fastapi import APIRouter, Response

from app.schemas import AnalysisOut, AnalysisRequest, ChordOut, KeyOut, RenderRequest
from app.services.storage import clean_text
from music_core.analysis import Analysis, Key, analyze
from music_core.musicxml import notes_to_musicxml
from music_core.notes import Note

router = APIRouter(prefix="/analysis", tags=["analysis"])
render_router = APIRouter(prefix="/render", tags=["analysis"])


def _key_out(key: Key | None) -> KeyOut | None:
    if key is None:
        return None
    return KeyOut(name=key.name, tonic=key.tonic, mode=key.mode, fifths=key.fifths)


def analysis_out(analysis: Analysis) -> AnalysisOut:
    return AnalysisOut(
        tempo=analysis.tempo,
        beats_per_measure=analysis.beats_per_measure,
        downbeat=analysis.downbeat,
        key=_key_out(analysis.key),
        estimated_key=_key_out(analysis.estimated_key),
        chords=[
            ChordOut(start=c.start, end=c.end, label=c.label(analysis.key)) for c in analysis.chords
        ],
    )


@router.post("", response_model=AnalysisOut)
def analyze_notes(payload: AnalysisRequest) -> AnalysisOut:
    """Key, bar grid and chord symbols of the given notes.

    Stateless, so the editor can show them for unsaved edits; exports run the same
    analysis on the stored notes.
    """
    notes = [Note.from_dict(note.model_dump()) for note in payload.notes]
    return analysis_out(
        analyze(
            notes,
            payload.tempo,
            payload.beats_per_measure,
            key=Key.parse(payload.key) if payload.key else None,
            downbeat=payload.downbeat,
        )
    )


@render_router.post("/musicxml")
def render_musicxml(payload: RenderRequest) -> Response:
    """MusicXML of the given notes, like the project export but for unsaved edits."""
    notes = [Note.from_dict(note.model_dump()) for note in payload.notes]
    data = notes_to_musicxml(
        notes,
        tempo=payload.tempo or 120.0,
        tuning=payload.tuning,
        title=clean_text(payload.title) or "Transcription",
        beats_per_measure=payload.beats_per_measure,
        key=Key.parse(payload.key) if payload.key else None,
        downbeat=payload.downbeat,
        capo=payload.capo,
    )
    return Response(data, media_type="application/vnd.recordare.musicxml+xml")
