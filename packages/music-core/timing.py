"""Frame / time / beat conversions, quantization and the beat grid."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from music_core.notes import Note


@dataclass(frozen=True)
class BeatGrid:
    """Beat times and bar lines: beat ``i`` starts a bar when ``(i - phase)`` is a multiple
    of ``beats_per_measure``.

    The beats are either evenly spaced (one tempo: ``fixed``) or tracked in the
    recording (``tracked``), so the grid can follow a player without a click. They cover
    the music: tracked beats are extended past both ends at the edge's tempo.
    """

    beats: tuple[float, ...]  # increasing, at least two
    beats_per_measure: int
    phase: int  # in [0, beats_per_measure)

    @classmethod
    def fixed(
        cls, tempo: float, beats_per_measure: int, downbeat: float, start: float, end: float
    ) -> BeatGrid:
        """Beats every ``60 / tempo`` s through a bar line at ``downbeat``, covering
        [``start``, ``end``] and a bar more on either side."""
        period = 60.0 / tempo
        first = math.floor((min(start, 0.0) - downbeat) / period) - beats_per_measure
        last = math.ceil((end - downbeat) / period) + beats_per_measure
        beats = tuple(downbeat + i * period for i in range(first, last + 1))
        return cls(beats, beats_per_measure, (-first) % beats_per_measure)

    @classmethod
    def tracked(
        cls, beats: Sequence[float], beats_per_measure: int, phase: int, start: float, end: float
    ) -> BeatGrid:
        """Tracked beats, extended at the edges' tempo to cover [``start``, ``end``] and a
        bar more on either side; ``phase`` counts from ``beats[0]``."""
        beats = [float(b) for b in sorted(beats)]
        if len(beats) < 2:
            raise ValueError("A tracked grid needs at least two beats")
        head = float(np.median(np.diff(beats[:5])))
        tail = float(np.median(np.diff(beats[-5:])))
        before = max(0, math.ceil((beats[0] - min(start, 0.0)) / head)) + beats_per_measure
        after = max(0, math.ceil((end - beats[-1]) / tail)) + beats_per_measure
        extended = (
            [beats[0] - k * head for k in range(before, 0, -1)]
            + beats
            + [beats[-1] + k * tail for k in range(1, after + 1)]
        )
        return cls(tuple(extended), beats_per_measure, (phase + before) % beats_per_measure)

    def position(self, time: float | np.ndarray) -> float | np.ndarray:
        """Fractional beat index of ``time`` (linear past the ends)."""
        beats = np.asarray(self.beats)
        index = np.arange(len(beats), dtype=float)
        t = np.asarray(time, dtype=float)
        inside = np.interp(t, beats, index)
        first = (t - beats[0]) / (beats[1] - beats[0])
        last = len(beats) - 1 + (t - beats[-1]) / (beats[-1] - beats[-2])
        out = np.where(t < beats[0], first, np.where(t > beats[-1], last, inside))
        return float(out) if np.ndim(out) == 0 else out

    def time(self, position: float | np.ndarray) -> float | np.ndarray:
        """Inverse of ``position``."""
        beats = np.asarray(self.beats)
        p = np.asarray(position, dtype=float)
        index = np.arange(len(beats), dtype=float)
        inside = np.interp(p, index, beats)
        first = beats[0] + p * (beats[1] - beats[0])
        last = beats[-1] + (p - len(beats) + 1) * (beats[-1] - beats[-2])
        out = np.where(p < 0, first, np.where(p > len(beats) - 1, last, inside))
        return float(out) if np.ndim(out) == 0 else out

    def is_bar_line(self, index: int) -> bool:
        return (index - self.phase) % self.beats_per_measure == 0

    @property
    def bar_lines(self) -> list[float]:
        return [t for i, t in enumerate(self.beats) if self.is_bar_line(i)]

    @property
    def tempo(self) -> float:
        """BPM of the median beat."""
        return 60.0 / float(np.median(np.diff(self.beats)))

    def tempo_at(self, index: int) -> float:
        """BPM of the beat starting at ``beats[index]`` (the last one for the last beat)."""
        index = min(max(index, 0), len(self.beats) - 2)
        return 60.0 / (self.beats[index + 1] - self.beats[index])

    def nearest_beat(self, time: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self.beats) - time)))

    def with_bar_line_at(self, time: float) -> BeatGrid:
        """The same beats with a bar line on the beat nearest to ``time``."""
        return replace(self, phase=self.nearest_beat(time) % self.beats_per_measure)


def median_tempo(beats: Sequence[float]) -> float:
    """BPM of the median interval between tracked beats."""
    return 60.0 / float(np.median(np.diff(np.asarray(beats, dtype=float))))


def downbeat_phase(
    beats: Sequence[float], downbeats: Sequence[float], beats_per_measure: int
) -> int:
    """Which beat of ``beats`` (mod ``beats_per_measure``) most of the tracked bar lines
    ``downbeats`` fall on: regular bars through a tracker's occasional missed bar line."""
    beats = np.asarray(beats, dtype=float)
    if not len(beats) or not len(downbeats):
        return 0
    votes = Counter(int(np.argmin(np.abs(beats - d))) % beats_per_measure for d in downbeats)
    return votes.most_common(1)[0][0]


def tracked_meter(beats: Sequence[float], downbeats: Sequence[float]) -> int:
    """Beats per bar (2-7): the most common count of beats between tracked bar lines."""
    beats = np.asarray(beats, dtype=float)
    counts = Counter(
        int(np.sum((beats >= a - 1e-3) & (beats < b - 1e-3)))
        for a, b in zip(downbeats, downbeats[1:])
    )
    return next((n for n, _ in counts.most_common() if 2 <= n <= 7), 4)


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
