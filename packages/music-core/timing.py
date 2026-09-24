"""Frame / time / beat conversions and quantization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from music_core.notes import Note


def frame_rate(sample_rate: int, hop_length: int) -> float:
    return sample_rate / hop_length


def frames_to_time(frames, sample_rate: int, hop_length: int):
    return np.asarray(frames) * hop_length / sample_rate


def time_to_frames(times, sample_rate: int, hop_length: int):
    return np.floor(np.asarray(times) * sample_rate / hop_length).astype(int)


def seconds_per_beat(tempo: float) -> float:
    if tempo <= 0:
        raise ValueError("Tempo must be positive")
    return 60.0 / tempo


def seconds_to_beats(seconds: float, tempo: float) -> float:
    return seconds / seconds_per_beat(tempo)


def beats_to_seconds(beats: float, tempo: float) -> float:
    return beats * seconds_per_beat(tempo)


def quantize_time(seconds: float, tempo: float, subdivision: int = 4) -> float:
    """Snap a time to the nearest 1/``subdivision`` of a beat."""
    grid = seconds_per_beat(tempo) / subdivision
    return round(seconds / grid) * grid


def quantize_notes(notes: Iterable[Note], tempo: float, subdivision: int = 4) -> list[Note]:
    """Snap note boundaries to the beat grid, keeping every note at least one grid step long."""
    grid = seconds_per_beat(tempo) / subdivision
    quantized = []
    for note in notes:
        start = quantize_time(note.start, tempo, subdivision)
        end = max(start + grid, quantize_time(note.end, tempo, subdivision))
        quantized.append(replace(note, start=start, end=end))
    return quantized
