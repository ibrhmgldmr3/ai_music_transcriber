import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, Integer, String, Text, false, true
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Mapped, deferred, mapped_column
from sqlalchemy.types import TypeDecorator

from app.models.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend.

    SQLite drops the offset and hands back naive values, which the API would serialize
    without a zone and browsers would then read as local time.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


class ProjectStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Project(Base):
    """An uploaded recording and its (editable) transcription."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(255))
    audio_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus), default=ProjectStatus.pending
    )
    # What was recorded: "guitar" (the transcription model), "voice" (a sung, hummed or
    # whistled melody, ml.inference.voice) or "song" (a whole song: its chords and sung
    # melody, ml.inference.song), the last two then set for guitar.
    source: Mapped[str] = mapped_column(String(16), default="guitar", server_default="guitar")
    # Song mode: isolate the guitar (voice: the vocals) from a band mix before transcribing.
    separate_guitar: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    # The guitar's tuning (music_core.tab.TUNINGS) and capo; `tuning` below holds the
    # resulting open strings, capo included, which the frets count from.
    tuning_name: Mapped[str] = mapped_column(
        String(32), default="standard", server_default="standard"
    )
    capo: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # Song projects: the transcription picks the capo that makes the chords easiest.
    capo_auto: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    # Models that produced the transcription (services.transcription.model_version) and
    # whether the user has saved edits since, which a new transcription would discard.
    model_version: Mapped[str | None] = mapped_column(String(96), nullable=True)
    edited: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # While a transcription runs: the done fraction and its stage (Predictor.transcribe).
    progress: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    tempo: Mapped[float | None] = mapped_column(Float, nullable=True)
    tuning: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Semitones the notes were moved from the recording (voice: octaves, to fit the guitar).
    transpose: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    notes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Voice mode: the sung pitch the editor draws behind the notes (TranscriptionResult.
    # pitch_curve). Loaded only when asked for: it can hold tens of thousands of values.
    pitch_curve: Mapped[dict | None] = deferred(mapped_column(JSON, nullable=True))
    # Song projects: chords [{start, end, label}] (Harte labels, editable) and their keys
    # over time [{start, end, key}] when the song modulates.
    chords: Mapped[list | None] = mapped_column(JSON, nullable=True)
    keys: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Tracked in the recording (guitar and song projects): {"beats": [...], "downbeats":
    # [...]}, guitar projects' "chord_scores" (the chord recognizer's best labels per beat,
    # fused with the notes) and song projects' "strums" ([time, strength] of the
    # accompaniment, for the strumming pattern). Loaded only when asked for.
    beats: Mapped[dict | None] = deferred(mapped_column(JSON, nullable=True))
    # Whether the bar grid follows the tracked beats (else one tempo, `tempo`).
    beat_grid: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())
    # Notation, set by the user; None means estimated from the notes (music_core.analysis).
    beats_per_measure: Mapped[int] = mapped_column(Integer, default=4, server_default="4")
    key_name: Mapped[str | None] = mapped_column(String(16), nullable=True)  # "A minor"
    downbeat: Mapped[float | None] = mapped_column(Float, nullable=True)  # a bar line (s)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_utcnow, onupdate=_utcnow)
