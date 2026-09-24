"""Note representation and pitch conversions."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from typing import Any

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_PITCH_CLASSES = {name: i for i, name in enumerate(NOTE_NAMES)} | {
    "Db": 1,
    "Eb": 3,
    "Gb": 6,
    "Ab": 8,
    "Bb": 10,
}
_NAME_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")

A4_MIDI = 69
A4_HZ = 440.0


@dataclass
class Note:
    """A single note event (pitch, onset ``start``, offset ``end``).

    ``string`` / ``fret`` are optional guitar positions: string 0 is the lowest string
    (low E in standard tuning) and fret 0 is the open string. ``confidence`` is the
    model's certainty in [0, 1] (None for ground truth or hand-entered notes).
    """

    pitch: int
    start: float
    end: float
    velocity: int = 80
    string: int | None = None
    fret: int | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Note end ({self.end}) is before its start ({self.start})")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def name(self) -> str:
        return midi_to_name(self.pitch)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Note:
        return cls(**{k: v for k, v in data.items() if k in _NOTE_FIELDS})


_NOTE_FIELDS = {f.name for f in fields(Note)}


def midi_to_hz(midi: float) -> float:
    return A4_HZ * 2.0 ** ((midi - A4_MIDI) / 12.0)


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        raise ValueError("Frequency must be positive")
    return A4_MIDI + 12.0 * math.log2(hz / A4_HZ)


def midi_to_name(midi: int) -> str:
    """60 -> 'C4'."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def name_to_midi(name: str) -> int:
    """'C4' -> 60, 'Eb3' -> 51."""
    match = _NAME_RE.match(name.strip())
    if not match:
        raise ValueError(f"Invalid note name: {name!r}")
    letter, accidental, octave = match.groups()
    pitch_class = _PITCH_CLASSES[letter.upper() + accidental]
    return (int(octave) + 1) * 12 + pitch_class


def sort_notes(notes: Iterable[Note]) -> list[Note]:
    return sorted(notes, key=lambda n: (n.start, n.pitch))
