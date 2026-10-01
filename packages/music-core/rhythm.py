"""Strumming rhythm: which eighths or sixteenths of each bar are struck.

Strums (times and strengths: the accompaniment's onsets, or chords in a transcription)
are placed on the beat grid's sixteenths. A song whose strums hardly ever fall between
the eighths is written in eighths. Every bar gets its pattern of struck slots, and the
most common bar pattern is the song's rhythm.

Directions follow the pendulum rule guitar teachers use: the hand moves down and up
every slot (eighth or sixteenth) whether it strikes or not, so a strum on an even slot
is a downstroke and on an odd slot an upstroke. On GuitarSet's strummed accompaniments
(directions read from the per-string onsets) the rule is right for 78% of the strums:
93% in funk, 80% rock, 74% singer-songwriter, 69% jazz, 67% bossa nova, which is
plucked rather than strummed (scripts/benchmark_strums.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from music_core.notes import Note
from music_core.timing import BeatGrid

STRUM_GROUP = 0.06  # s: notes starting this close together are one strum
# A transcription restarts only some strings of a chord strummed again, so two notes make
# a strum: on GuitarSet player 05 that found 90% of the strums, against 84% with three.
MIN_STRUM_NOTES = 2
# Sixteenth-note songs: at least this share of the strum strength off the eighths.
SIXTEENTHS_SHARE = 0.1
# A strum counts in a bar's pattern when at least this strong (of the song's typical one).
MIN_STRENGTH = 0.3


@dataclass(frozen=True)
class BarRhythm:
    start: float  # the bar line (seconds)
    hits: tuple[bool, ...]  # per slot: beats_per_measure * per_beat


@dataclass(frozen=True)
class Rhythm:
    per_beat: int  # 2: eighths, 4: sixteenths
    bars: tuple[BarRhythm, ...]
    pattern: tuple[bool, ...]  # the most common bar

    @staticmethod
    def down(slot: int) -> bool:
        """The pendulum rule: even slots are downstrokes."""
        return slot % 2 == 0

    def text(self, hits: Sequence[bool] | None = None) -> str:
        """'D-DU-UDU' style: D down, U up, - not struck (the pattern by default)."""
        hits = self.pattern if hits is None else hits
        return "".join(("D" if self.down(i) else "U") if h else "-" for i, h in enumerate(hits))


def strums_from_notes(notes: Sequence[Note]) -> list[tuple[float, float]]:
    """(time, strength) of the chords in a transcription: ``MIN_STRUM_NOTES`` or more notes
    starting within ``STRUM_GROUP``; the strength is how many."""
    onsets = sorted(n.start for n in notes if n.end > n.start)
    groups: list[list[float]] = []
    for t in onsets:
        if groups and t - groups[-1][0] <= STRUM_GROUP:
            groups[-1].append(t)
        else:
            groups.append([t])
    return [(g[0], float(len(g))) for g in groups if len(g) >= MIN_STRUM_NOTES]


def strum_rhythm(
    strums: Sequence[tuple[float, float]],
    grid: BeatGrid,
    start: float = 0.0,
    end: float | None = None,
) -> Rhythm | None:
    """The bars' strumming patterns from (time, strength) strums between ``start`` and
    ``end``; None without strums."""
    strums = [(t, s) for t, s in strums if t >= start - 1e-6 and (end is None or t < end)]
    if not strums:
        return None
    times = np.array([t for t, _ in strums])
    strength = np.array([s for _, s in strums], dtype=float)
    sixteenth = np.round(np.asarray(grid.position(times)) * 4).astype(int)
    off_eighths = strength[sixteenth % 2 == 1].sum() / strength.sum()
    per_beat = 4 if off_eighths >= SIXTEENTHS_SHARE else 2
    slot = (
        sixteenth if per_beat == 4 else np.round(np.asarray(grid.position(times)) * 2).astype(int)
    )
    typical = float(np.median(strength))
    strong = strength >= MIN_STRENGTH * typical

    n = grid.beats_per_measure
    slots = n * per_beat
    bar_lines = [i for i in range(len(grid.beats)) if grid.is_bar_line(i)]
    first = int(np.floor(slot.min() / per_beat))
    last = int(np.floor(slot.max() / per_beat))
    bars: list[BarRhythm] = []
    for index in bar_lines:
        if index + n <= first or index > last:
            continue
        hits = [False] * slots
        for k in slot[strong]:
            offset = int(k) - index * per_beat
            if 0 <= offset < slots:
                hits[offset] = True
        bars.append(BarRhythm(float(grid.beats[index]), tuple(hits)))
    return Rhythm(per_beat, tuple(bars), _consensus(bars, slots))


def _consensus(bars: Sequence[BarRhythm], slots: int) -> tuple[bool, ...]:
    """The slots struck in at least half of the bars that are strummed (two strums or
    more): one stray or missed strum in a bar doesn't change the song's pattern."""
    strummed = [b.hits for b in bars if sum(b.hits) >= 2]
    if not strummed:
        return tuple([False] * slots)
    share = np.mean(np.array(strummed, dtype=float), axis=0)
    return tuple(bool(s >= 0.5) for s in share)
