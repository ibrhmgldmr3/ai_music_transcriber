"""Chords for guitar: names, playable shapes and a capo that makes them easy.

In standard tuning a chord gets the shape a guitarist expects: an open chord when there
is one (x32010 for C, xx0232 for D, x02210 for Am), else a movable barre shape with the
root on the low E or the A string (133211 for F, x24432 for Bm), each with a difficulty.
Other tunings have no such canon, so their shapes are searched: every string muted, open
or fretted within a four-fret window, the root lowest, every characteristic tone sounding
(the fifth may be left out), at most four fingers (a barre counts as one), scored like
the canonical shapes (open strings are free; fingers, barres, stretches, high positions
and muted strings cost).

A capo moves the easy shapes: a song in Ab (Ab, Db, Eb, Fm) is G, C, D, Em shapes with a
capo at the first fret. ``suggest_capo`` picks the fret that makes the whole song easiest.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from music_core.analysis import Key, KeySpan, pitch_class_name
from music_core.tab import STANDARD_TUNING

# Harte qualities (as ml.inference.chords recognizes them) -> intervals above the root and
# the suffix a chord sheet writes.
QUALITIES: dict[str, tuple[tuple[int, ...], str]] = {
    "maj": ((0, 4, 7), ""),
    "min": ((0, 3, 7), "m"),
    "7": ((0, 4, 7, 10), "7"),
    "maj7": ((0, 4, 7, 11), "maj7"),
    "min7": ((0, 3, 7, 10), "m7"),
    "maj6": ((0, 4, 7, 9), "6"),
    "min6": ((0, 3, 7, 9), "m6"),
    "sus2": ((0, 2, 7), "sus2"),
    "sus4": ((0, 5, 7), "sus4"),
    "dim": ((0, 3, 6), "dim"),
    "dim7": ((0, 3, 6, 9), "dim7"),
    "hdim7": ((0, 3, 6, 10), "m7b5"),
    "aug": ((0, 4, 8), "aug"),
    "minmaj7": ((0, 3, 7, 11), "m(maj7)"),
    "5": ((0, 7), "5"),
}
_ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SPAN = 4  # frets a hand covers
_MAX_FINGERS = 4
_NECK = 0.08  # extra difficulty per fret a barre shape sits up the neck
_UNPLAYABLE = 10.0  # difficulty of a chord without a shape

# Open chords in standard tuning: (chord, shape from low E to high e, difficulty).
_OPEN_SHAPES = [
    ("C", "x32010", 1.4), ("A", "x02220", 1.2), ("G", "320003", 1.4), ("E", "022100", 1.0),
    ("D", "xx0232", 1.3),
    ("A:min", "x02210", 1.2), ("E:min", "022000", 0.8), ("D:min", "xx0231", 1.5),
    ("C:7", "x32310", 1.8), ("A:7", "x02020", 1.0), ("G:7", "320001", 1.5),
    ("E:7", "020100", 0.9), ("D:7", "xx0212", 1.5), ("B:7", "x21202", 2.2),
    ("C:maj7", "x32000", 1.2), ("A:maj7", "x02120", 1.4), ("G:maj7", "320002", 1.5),
    ("E:maj7", "021100", 1.4), ("D:maj7", "xx0222", 1.4), ("F:maj7", "xx3210", 1.6),
    ("A:min7", "x02010", 1.0), ("E:min7", "022030", 1.0), ("D:min7", "xx0211", 1.6),
    ("A:sus2", "x02200", 1.0), ("D:sus2", "xx0230", 1.2), ("E:sus4", "022200", 1.1),
    ("A:sus4", "x02230", 1.2), ("D:sus4", "xx0233", 1.3),
    ("C:maj6", "x32210", 1.8), ("A:maj6", "x02222", 1.4), ("G:maj6", "320000", 1.2),
    ("E:maj6", "022120", 1.5), ("D:maj6", "xx0202", 1.4),
    ("A:min6", "x02212", 1.6), ("E:min6", "022020", 1.2), ("D:min6", "xx0201", 1.6),
    ("E:5", "022xxx", 0.8), ("A:5", "x022xx", 0.8), ("D:5", "xx023x", 0.9),
    ("C:aug", "x32110", 1.9),
]  # fmt: skip

# Movable shapes: (root string, fret offsets from the root's fret per string, None =
# muted, difficulty at the first fret).
_MOVABLE: dict[str, list[tuple[int, tuple[int | None, ...], float]]] = {
    "maj": [(0, (0, 2, 2, 1, 0, 0), 3.0), (1, (None, 0, 2, 2, 2, 0), 3.0)],
    "min": [(0, (0, 2, 2, 0, 0, 0), 3.0), (1, (None, 0, 2, 2, 1, 0), 3.0)],
    "7": [(0, (0, 2, 0, 1, 0, 0), 3.0), (1, (None, 0, 2, 0, 2, 0), 3.0)],
    "maj7": [(0, (0, None, 1, 1, 0, None), 2.6), (1, (None, 0, 2, 1, 2, 0), 3.0)],
    "min7": [(0, (0, 2, 0, 0, 0, 0), 3.0), (1, (None, 0, 2, 0, 1, 0), 3.0)],
    "maj6": [(0, (0, None, -1, 1, 0, None), 2.8), (1, (None, 0, 2, 2, 2, 2), 3.2)],
    "min6": [(0, (0, None, -1, 0, 0, None), 2.8), (1, (None, 0, 2, -1, 1, None), 3.0)],
    "sus2": [(1, (None, 0, 2, 2, 0, 0), 3.0)],
    "sus4": [(0, (0, 2, 2, 2, 0, 0), 3.0), (1, (None, 0, 2, 2, 3, 0), 3.2)],
    "5": [(0, (0, 2, 2, None, None, None), 1.5), (1, (None, 0, 2, 2, None, None), 1.5)],
    "dim": [(1, (None, 0, 1, 2, 1, None), 2.6)],
    "dim7": [(1, (None, 0, 1, -1, 1, None), 2.6), (0, (0, None, -1, 0, -1, None), 2.8)],
    "hdim7": [(1, (None, 0, 1, 0, 1, None), 2.6), (0, (0, None, 0, 0, -1, None), 2.8)],
    "aug": [(1, (None, 0, 3, 2, 2, None), 3.0)],
    "minmaj7": [(1, (None, 0, 2, 1, 1, 0), 3.2)],
}


# Harte bass intervals ("C:maj/3" is C over E) and the names written for each.
BASS_INTERVALS = {"b2": 1, "2": 2, "b3": 3, "3": 4, "4": 5, "#4": 6, "b5": 6, "5": 7, "#5": 8,
                  "b6": 8, "6": 9, "bb7": 9, "b7": 10, "7": 11}  # fmt: skip
_INTERVAL_NAMES = {1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "b6", 9: "6",
                   10: "b7", 11: "7"}  # fmt: skip


@dataclass(frozen=True)
class ChordSymbol:
    root: int  # pitch class
    quality: str  # a key of QUALITIES
    bass: int | None = None  # pitch class of a bass note other than the root (slash chord)

    @classmethod
    def parse(cls, label: str) -> ChordSymbol | None:
        """'A#:min7' / 'Bb:maj' / 'C' / 'D:maj/3' (Harte: D over its third, F#) -> a
        symbol; None for 'N' (no chord) or 'X'."""
        if label in ("N", "X", ""):
            return None
        head, _, bass_text = label.partition("/")
        root_text, _, quality = head.partition(":")
        root = _note(root_text)
        if root is None:
            raise ValueError(f"Not a chord: {label!r}")
        quality = quality.split("(")[0] or "maj"
        if quality not in QUALITIES:
            raise ValueError(f"Unknown chord quality: {label!r}")
        bass = None
        if bass_text:
            if bass_text in BASS_INTERVALS:
                bass = (root + BASS_INTERVALS[bass_text]) % 12
            elif bass_text != "1":
                bass = _note(bass_text)  # "C/E" as people write it
                if bass is None:
                    raise ValueError(f"Unknown bass note: {label!r}")
        return cls(root, quality, None if bass == root else bass)

    def transposed(self, semitones: int) -> ChordSymbol:
        bass = None if self.bass is None else (self.bass + semitones) % 12
        return ChordSymbol((self.root + semitones) % 12, self.quality, bass)

    def name(self, key: Key | None = None) -> str:
        """'Bbm7', 'F#', 'Cmaj7', 'D/F#' ... spelled for ``key``."""
        text = pitch_class_name(self.root, key) + QUALITIES[self.quality][1]
        return text if self.bass is None else f"{text}/{pitch_class_name(self.bass, key)}"

    @property
    def harte(self) -> str:
        text = f"{pitch_class_name(self.root, Key(0, 'major'))}:{self.quality}"
        if self.bass is not None:
            text += "/" + _INTERVAL_NAMES[(self.bass - self.root) % 12]
        return text

    @property
    def tones(self) -> set[int]:
        """Pitch classes of the chord, its bass included."""
        tones = {(self.root + i) % 12 for i in QUALITIES[self.quality][0]}
        return tones if self.bass is None else tones | {self.bass}


def _note(text: str) -> int | None:
    """'Bb' / 'F#' -> pitch class."""
    root = _ROOTS.get(text[:1])
    if root is None or text[1:].strip("#b"):
        return None
    return (root + text[1:].count("#") - text[1:].count("b")) % 12


@dataclass(frozen=True)
class Voicing:
    frets: tuple[int, ...]  # per string, lowest first; -1 = muted, 0 = open
    difficulty: float

    @property
    def diagram(self) -> str:
        """'x32010' (frets above 9 in parentheses)."""
        return "".join("x" if f < 0 else str(f) if f < 10 else f"({f})" for f in self.frets)


def voicing(
    chord: ChordSymbol, tuning: Sequence[int] = STANDARD_TUNING, max_fret: int = 15
) -> Voicing | None:
    """The easiest shape of ``chord`` (frets counted from the nut, or from a capo)."""
    return _voicing(chord, tuple(tuning), max_fret)


@lru_cache(maxsize=4096)
def _voicing(chord: ChordSymbol, tuning: tuple[int, ...], max_fret: int) -> Voicing | None:
    if tuning == tuple(STANDARD_TUNING) and chord.bass is None:
        return _standard_voicing(chord, max_fret)
    # Slash chords (and every chord in other tunings) are searched: the bass note lowest,
    # e.g. D/F# 200232, G/B x20003, C/E 032010.
    return _searched_voicing(chord, tuning, max_fret)


def _standard_voicing(chord: ChordSymbol, max_fret: int) -> Voicing | None:
    """The open shape if there is one, else the easiest movable shape."""
    candidates = [
        Voicing(tuple(-1 if c == "x" else int(c) for c in shape), cost)
        for label, shape, cost in _OPEN_SHAPES
        if ChordSymbol.parse(label) == chord
    ]
    for string, offsets, cost in _MOVABLE.get(chord.quality, []):
        root_fret = (chord.root - STANDARD_TUNING[string]) % 12
        for fret in (root_fret, root_fret + 12):
            frets = tuple(-1 if o is None else fret + o for o in offsets)
            if min(f for f in frets if f >= 0) >= 1 and max(frets) <= max_fret:
                candidates.append(Voicing(frets, cost + _NECK * (fret - 1)))
    return min(candidates, key=lambda v: v.difficulty, default=None)


def _searched_voicing(chord: ChordSymbol, tuning: tuple[int, ...], max_fret: int) -> Voicing | None:
    intervals = QUALITIES[chord.quality][0]
    tones = chord.tones
    needed = {(chord.root + i) % 12 for i in intervals if i != 7 or chord.quality == "5"}
    bass = chord.root if chord.bass is None else chord.bass
    needed.add(bass)
    # A slash chord is the bass under a full chord, not a thin triad on the top strings.
    min_strings = 3 if chord.bass is None else 4
    best: Voicing | None = None
    for low in range(0, max_fret - _SPAN + 2):
        options = [_string_options(open_pitch, tones, low, max_fret) for open_pitch in tuning]
        for frets in itertools.product(*options):
            if sum(f >= 0 for f in frets) < min_strings:
                continue
            cost = _difficulty(frets, tuning, bass, needed)
            if cost is not None and (best is None or cost < best.difficulty):
                best = Voicing(frets, round(cost, 3))
    return best


def _string_options(open_pitch: int, tones: set[int], low: int, max_fret: int) -> list[int]:
    """Muted, open (if a chord tone) and the chord tones within the window [low, low+3]."""
    options = [-1]
    if open_pitch % 12 in tones:
        options.append(0)
    for fret in range(max(1, low), min(max_fret, low + _SPAN - 1) + 1):
        if (open_pitch + fret) % 12 in tones:
            options.append(fret)
    return options


def _difficulty(
    frets: tuple[int, ...], tuning: tuple[int, ...], root: int, needed: set[int]
) -> float | None:
    """How hard a shape is to play, or None if it isn't a chord shape with ``root`` (the
    bass note of a slash chord) lowest."""
    played = [i for i, f in enumerate(frets) if f >= 0]
    if len(played) < 3:
        return None
    # Muted strings only below the chord (and possibly the top string): a strummed chord
    # can't skip strings in the middle.
    top = len(frets) - 1
    if any(frets[i] < 0 for i in range(played[0], played[-1] + 1)) or played[-1] < top - 1:
        return None
    pitches = [tuning[i] + frets[i] for i in played]
    if pitches[0] % 12 != root or not needed <= {p % 12 for p in pitches}:
        return None
    if min(pitches) != pitches[0]:  # a higher string sounding below the bass
        return None
    fretted = [f for f in frets if f > 0]
    lowest = min(fretted, default=0)
    fingers, barre = len(fretted), False
    if fingers > _MAX_FINGERS:
        # Too many notes for four fingers: the index finger must lie across the lowest
        # fret (a barre), which works only if no open string sounds above it.
        at_lowest = [i for i, f in enumerate(frets) if f == lowest]
        barre = all(frets[i] != 0 for i in range(at_lowest[0], top + 1))
        fingers = 1 + sum(f > lowest for f in fretted)
        if not barre or fingers > _MAX_FINGERS:
            return None
    return (
        0.3 * fingers
        + (2.0 if barre else 0.0)
        + 0.25 * (max(fretted, default=0) - lowest)  # stretch
        + 0.15 * lowest
        - 0.3 * sum(f == 0 for f in frets)
        + 0.4 * sum(f < 0 for f in frets)
        + (0.5 if frets[top] < 0 else 0.0)
    )


def suggest_capo(
    chords: Sequence[tuple[ChordSymbol, float]],
    tuning: Sequence[int] = STANDARD_TUNING,
    max_capo: int = 7,
) -> tuple[int, dict[int, float]]:
    """The capo fret whose shapes make the chords easiest, weighted by how long each
    chord sounds, and every fret's mean difficulty. A capo is a small bother of its own,
    so ties go to the lower fret."""
    total = sum(duration for _, duration in chords) or 1.0
    costs: dict[int, float] = {}
    for capo in range(max_capo + 1):
        cost = 0.0
        for chord, duration in chords:
            shape = voicing(chord.transposed(-capo), tuning)
            cost += duration * (shape.difficulty if shape else _UNPLAYABLE)
        costs[capo] = round(cost / total + 0.05 * capo, 4)
    return min(costs, key=lambda capo: (costs[capo], capo)), costs


def chord_sheet(
    chords: Sequence[tuple[float, float, ChordSymbol]],
    downbeats: Sequence[float],
    *,
    title: str = "",
    key: Key | None = None,
    capo: int = 0,
    tempo: float | None = None,
    beats_per_measure: int = 4,
    tuning: Sequence[int] = STANDARD_TUNING,
    bars_per_line: int = 4,
    keys: Sequence[KeySpan] | None = None,
    rhythm: str | None = None,
) -> str:
    """Plain-text chord sheet: the shapes to play (from the capo), bar by bar, and each
    shape's fingering. ``tuning`` is the guitar's own, without the capo. Bars run between
    the ``downbeats`` (2 s each without them), with a pickup bar for chords before the
    first bar line, as in the web editor's chart. ``keys`` (a modulating song's keys)
    are listed with the bar each starts in, ``rhythm`` ("D-DU-UDU") under the header."""
    header = [title] if title else []
    facts = [f"Ton: {key.name}"] if key else []
    if capo:
        facts.append(f"Capo {capo}")
    if tempo:
        facts.append(f"{round(tempo)} BPM")
    facts.append(f"{beats_per_measure}/4")
    header.append(" · ".join(facts))
    if rhythm:
        header.append(f"Ritim: {rhythm}  (D aşağı, U yukarı, - boş; her ölçü)")

    def shape(chord: ChordSymbol) -> str:
        return chord.transposed(-capo).name(None if capo else key)

    lines: list[str] = []
    if chords:
        start = min(s for s, _, _ in chords)
        end = max(e for _, e, _ in chords)
        bars = [t for t in downbeats if t < end] or [2.0 * i for i in range(math.ceil(end / 2))]
        if start < bars[0] - 0.05:
            bars = [start, *bars]  # a pickup bar
        bounds = [*bars, max(end, bars[-1] + 1e-3)]
        cells = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            names = [shape(c) for s, e, c in chords if s < b - 1e-3 and e > a + 1e-3]
            cells.append(" ".join(dict.fromkeys(names)) or "-")  # in order, no repeats
        width = max(len(c) for c in cells) + 2
        for i in range(0, len(cells), bars_per_line):
            lines.append(
                "|" + "|".join(f" {c:<{width - 1}}" for c in cells[i : i + bars_per_line]) + "|"
            )
        if keys and len(keys) > 1:
            changes = [
                f"ölçü {1 + sum(b <= span.start + 0.05 for b in bounds[1:-1])}: {span.key.name}"
                for span in keys[1:]
            ]
            header.append("Ton değişimi: " + ", ".join(changes))

    legend = []
    for chord in dict.fromkeys(c for _, _, c in chords):
        v = voicing(chord.transposed(-capo), tuning)
        sounding = f" (= {chord.name(key)})" if capo else ""
        legend.append(f"{shape(chord):<8} {v.diagram if v else '?':<8}{sounding}")
    return "\n".join([*header, "", *lines, "", "Akorlar (kalın telden ince tele):", *legend]) + "\n"
