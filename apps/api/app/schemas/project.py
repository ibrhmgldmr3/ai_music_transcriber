# Keep in sync with packages/shared-types/index.ts.
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import settings
from app.models.project import ProjectStatus

MAX_NOTE_SECONDS = 6 * 3600.0  # far beyond max_audio_minutes; bounds MIDI/tab export sizes
MAX_STRINGS = 12
MAX_FRET = 36


class NoteSchema(BaseModel):
    # NaN/Infinity would be stored and later break JSON responses and exports.
    model_config = ConfigDict(allow_inf_nan=False)

    pitch: int = Field(ge=0, le=127)
    start: float = Field(ge=0, le=MAX_NOTE_SECONDS)
    end: float = Field(ge=0, le=MAX_NOTE_SECONDS)
    velocity: int = Field(default=80, ge=1, le=127)
    string: int | None = Field(default=None, ge=0, lt=MAX_STRINGS)
    fret: int | None = Field(default=None, ge=0, le=MAX_FRET)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "NoteSchema":
        if self.end < self.start:
            raise ValueError("end must not be before start")
        if (self.string is None) != (self.fret is None):
            raise ValueError("string and fret must be given together")
        return self


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    filename: str
    status: ProjectStatus
    separate_guitar: bool
    error: str | None
    duration: float | None
    created_at: datetime
    updated_at: datetime


class TranscriptionOut(BaseModel):
    project_id: str
    tempo: float | None
    tuning: list[int]
    mean_confidence: float | None
    notes: list[NoteSchema]


class NotesUpdate(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    notes: list[NoteSchema] = Field(max_length=settings.max_notes)
    # Corrected tempo (BPM); the estimate can be off, and exports quantize to it.
    tempo: float | None = Field(default=None, ge=20, le=400)
