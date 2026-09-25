from fastapi import APIRouter

from app.schemas import AnalysisOut, AnalysisRequest, ChordOut, KeyOut
from music_core.analysis import Analysis, Key, analyze
from music_core.notes import Note

router = APIRouter(prefix="/analysis", tags=["analysis"])


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
