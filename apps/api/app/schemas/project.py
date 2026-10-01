# Keep in sync with packages/shared-types/index.ts.
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from app.config import settings
from app.models.project import ProjectStatus
from music_core.analysis import KEY_NAMES
from music_core.guitar_chords import ChordSymbol
from music_core.tab import MAX_CAPO, TUNINGS

MAX_NOTE_SECONDS = 6 * 3600.0  # far beyond max_audio_minutes; bounds MIDI/tab export sizes
MAX_STRINGS = 12
MAX_FRET = 36
MIN_TEMPO, MAX_TEMPO = 20, 400
MIN_BEATS, MAX_BEATS = 2, 7  # beats per measure (quarter-note beats)


def _known_key(value: str | None) -> str | None:
    if value is not None and value not in KEY_NAMES:
        raise ValueError(f"unknown key; expected one of: {', '.join(KEY_NAMES)}")
    return value


KeyName = Annotated[str | None, AfterValidator(_known_key)]
TuningName = Literal[tuple(TUNINGS)]  # type: ignore[valid-type]
# What was recorded: a guitar, a voice (sung, hummed or whistled melody) or a whole song.
SourceName = Literal["guitar", "voice", "song"]
Tempo = Annotated[float | None, Field(ge=MIN_TEMPO, le=MAX_TEMPO)]
BeatsPerMeasure = Annotated[int, Field(ge=MIN_BEATS, le=MAX_BEATS)]
Downbeat = Annotated[float | None, Field(ge=0, le=MAX_NOTE_SECONDS)]


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


class KeySpanOut(BaseModel):
    """A song's key over a stretch of time, e.g. {"start": 0, "end": 92.4, "key": "A minor"}."""

    start: float
    end: float
    key: str


class SongChord(BaseModel):
    """A chord of a song project, e.g. {"start": 1.5, "end": 3.0, "label": "A:min7"}, or
    a slash chord in Harte syntax, "D:maj/3" (D over F#)."""

    model_config = ConfigDict(allow_inf_nan=False)

    start: float = Field(ge=0, le=MAX_NOTE_SECONDS)
    end: float = Field(ge=0, le=MAX_NOTE_SECONDS)
    label: str = Field(max_length=16)  # Harte syntax: root, ':' and a quality

    @model_validator(mode="after")
    def _check(self) -> "SongChord":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        try:
            if ChordSymbol.parse(self.label) is None:
                raise ValueError
        except ValueError:
            raise ValueError("not a chord label") from None
        return self


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    filename: str
    status: ProjectStatus
    source: SourceName
    separate_guitar: bool
    tuning_name: str
    capo: int
    capo_auto: bool
    # Models that transcribed it (null before this was recorded) and whether the user
    # saved edits since.
    model_version: str | None
    edited: bool
    # While transcribing: done fraction (0-1) and stage (loading, separating,
    # transcribing, finishing).
    progress: float | None
    stage: str | None
    error: str | None
    duration: float | None
    created_at: datetime
    updated_at: datetime


class ModelInfo(BaseModel):
    version: str  # compare with ProjectOut.model_version of guitar projects
    voice_version: str  # ... of voice projects
    song_version: str  # ... and of song projects
    notes_model: str  # checkpoint folder, e.g. "guitar_v8"
    tab_model: str | None


class PitchCurveOut(BaseModel):
    """The sung pitch of a voice project, on the notes' scale."""

    frame_rate: float  # values per second, the first at 0 s
    values: list[float | None]  # MIDI pitch (fractional), None where nothing is sung


class TranscriptionOut(BaseModel):
    project_id: str
    tempo: float | None
    # Open-string pitches the frets count from (capo included), lowest string first.
    tuning: list[int]
    tuning_name: str
    capo: int
    # Semitones the notes were moved from the recording (voice mode fits the melody
    # into the guitar's range by octaves).
    transpose: int
    capo_auto: bool
    # Song projects: the chords and their keys over time (when the song modulates).
    chords: list[SongChord] | None
    keys: list[KeySpanOut] | None
    # The beats and bar lines tracked in the recording (guitar and song projects), and
    # whether the bar grid follows them (else the one tempo above).
    beats: list[float]
    downbeats: list[float]
    beat_grid: bool
    mean_confidence: float | None
    notes: list[NoteSchema]
    # Notation chosen by the user; null key/downbeat mean "estimate from the notes".
    beats_per_measure: int
    key: str | None
    downbeat: float | None


class NotesUpdate(BaseModel):
    """The editor's notes plus notation corrections.

    Omitted fields keep their stored value; ``key``/``downbeat`` set to null go back to
    the estimate.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    notes: list[NoteSchema] = Field(max_length=settings.max_notes)
    # Corrected tempo (BPM); the estimate can be off, and exports quantize to it.
    tempo: Tempo = None
    beats_per_measure: BeatsPerMeasure | None = None
    key: KeyName = None
    downbeat: Downbeat = None
    # Song projects: the edited chords (omitted: unchanged).
    chords: list[SongChord] | None = Field(default=None, max_length=5000)
    # Whether the bar grid follows the tracked beats (omitted: unchanged).
    beat_grid: bool | None = None


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    notes: list[NoteSchema] = Field(max_length=settings.max_notes)
    tempo: Tempo = None
    beats_per_measure: BeatsPerMeasure = 4
    key: KeyName = None  # null: estimate
    downbeat: Downbeat = None  # null: estimate
    # The project the notes belong to: its tracked beats (with ``beat_grid``), recognized
    # chords, keys and strums join the analysis. Without one, the notes alone are used.
    project_id: str | None = Field(default=None, max_length=36)
    beat_grid: bool = True
    duration: float | None = Field(default=None, ge=0, le=MAX_NOTE_SECONDS)  # grid's extent


class RenderRequest(AnalysisRequest):
    """Notes and notation to engrave, e.g. unsaved edits for the editor's score view."""

    # Open strings the frets count from (capo included), lowest string first.
    tuning: list[Annotated[int, Field(ge=0, le=127)]] = Field(min_length=4, max_length=MAX_STRINGS)
    capo: int = Field(default=0, ge=0, le=12)
    title: str = Field(default="Transcription", max_length=255)


class KeyOut(BaseModel):
    name: str  # "Bb major"
    tonic: int  # pitch class, 0 = C
    mode: str  # "major" | "minor"
    fifths: int  # key signature: sharps > 0, flats < 0


class VoicingRequest(BaseModel):
    """Chord labels to name and finger, e.g. a song's chords with a capo."""

    labels: list[str] = Field(max_length=500)
    capo: int = Field(default=0, ge=0, le=MAX_CAPO)
    tuning_name: TuningName = "standard"
    key: KeyName = None  # spells the names (Bb or A#)


class VoicingOut(BaseModel):
    label: str
    name: str  # the chord as it sounds, e.g. "Cm"
    shape: str  # what to play from the capo, e.g. "Am" with a capo at 3
    frets: list[int] | None  # per string, lowest first; -1 muted, 0 open; null: no shape
    difficulty: float | None


class ChordOut(BaseModel):
    start: float
    end: float
    label: str  # "F#m7", "D/F#"


class BarOut(BaseModel):
    time: float
    number: int  # as the MusicXML export numbers its measures (1: the first note's bar)


class RhythmOut(BaseModel):
    """The strumming pattern: struck eighths or sixteenths of a bar."""

    per_beat: int  # 2: eighths, 4: sixteenths
    pattern: list[bool]  # the song's bar: struck slots
    text: str  # e.g. "D-DU-UDU": D down, U up (the pendulum rule), - not struck
    bars: list[list[bool]]  # every bar's struck slots, from ``bar_times``
    bar_times: list[float]


class AnalysisOut(BaseModel):
    tempo: float  # with tracked beats: of the median beat
    beats_per_measure: int
    downbeat: float  # the first bar line at or after 0 s
    key: KeyOut | None  # the chosen key, else the estimate
    estimated_key: KeyOut | None
    chords: list[ChordOut]
    # The grid: every beat (bar lines included) and the bar lines, numbered.
    tracked: bool  # following the tracked beats rather than one tempo
    beats: list[float]
    bars: list[BarOut]
    keys: list[KeySpanOut]  # a song's keys over time
    rhythm: RhythmOut | None
