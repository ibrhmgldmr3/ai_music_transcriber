"""Strums in GuitarSet: reference strum times and directions from the per-string notes.

GuitarSet annotates every string separately (a hexaphonic pickup), so a strum shows as
notes on several strings starting within a few tens of milliseconds, low strings first
for a downstroke and high strings first for an upstroke. ``strums`` groups onsets into
such events; ``beat_slots`` places times on a sixteenth-note grid of the beats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.preprocessing.annotations import load_guitarset_jams

GROUP = 0.06  # s: onsets this close to the previous one belong to the same strum
MIN_STRINGS = 3


@dataclass(frozen=True)
class Strum:
    time: float  # the first string's onset
    strings: int
    down: bool | None  # None when the order is unclear (strings starting together)


def strums(jams: Path) -> list[Strum]:
    """The strums of a GuitarSet take: three or more strings starting within ``GROUP``."""
    notes = sorted(load_guitarset_jams(jams), key=lambda n: n.start)
    events: list[list[tuple[float, int]]] = []
    for note in notes:
        if note.string is None:
            continue
        if events and note.start - events[-1][-1][0] <= GROUP:
            events[-1].append((note.start, note.string))
        else:
            events.append([(note.start, note.string)])
    out = []
    for event in events:
        strings = {s for _, s in event}
        if len(strings) < MIN_STRINGS:
            continue
        times = np.array([t for t, _ in event])
        index = np.array([s for _, s in event], dtype=float)
        spread = times.max() - times.min()
        slope = np.polyfit(times - times.min(), index, 1)[0] if spread > 0.004 else 0.0
        # A downstroke reaches the high strings (larger index) later: positive slope.
        out.append(Strum(float(times.min()), len(strings), None if slope == 0 else slope > 0))
    return out


def beats_of(jams: Path) -> np.ndarray:
    data = json.loads(jams.read_text(encoding="utf-8"))
    beats = next(a for a in data["annotations"] if a["namespace"] == "beat_position")["data"]
    return np.array([b["time"] for b in beats])


def beat_slots(times: np.ndarray, beats: np.ndarray, per_beat: int = 4) -> np.ndarray:
    """Fractional beat position of each time (beats extended at both ends by their median
    spacing), rounded to 1/``per_beat`` of a beat: slot ``k`` is beat ``k // per_beat``."""
    beats = np.asarray(beats, dtype=float)
    period = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    grid = np.concatenate([[beats[0] - period * 8], beats, [beats[-1] + period * 8]])
    index = np.concatenate([[-8.0], np.arange(len(beats), dtype=float), [len(beats) + 7.0]])
    position = np.interp(times, grid, index)
    return np.round(position * per_beat).astype(int)
