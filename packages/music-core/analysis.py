"""Musical analysis of note events: key, beat/bar grid and chord symbols.

Everything is derived from the notes alone, so it follows the user's edits and tempo
corrections without running the model again:

- key: duration-weighted pitch-class histogram correlated with major/minor key profiles
- bar grid: for the given tempo and meter, the beat phase that puts the most (long and
  bass) onsets on beats, then the beat where the harmony changes most, i.e. the bar line
- chords: per-beat chroma of the sounding notes matched against chord templates and
  smoothed with Viterbi, so a chord lasts until the harmony actually changes
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from music_core.notes import Note
from music_core.rhythm import Rhythm, strum_rhythm, strums_from_notes
from music_core.timing import BeatGrid, downbeat_phase, median_tempo

# Temperley's key profiles from the Kostka-Payne corpus (Music and Probability, 2007),
# tonic first. On GuitarSet they beat Krumhansl-Kessler, clearly so on solos.
_MAJOR_PROFILE = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460,
                           0.096, 0.715, 0.104, 0.366, 0.057, 0.400])  # fmt: skip
_MINOR_PROFILE = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460,
                           0.105, 0.747, 0.404, 0.067, 0.133, 0.330])  # fmt: skip

# Conventional tonic name and key signature (sharps > 0, flats < 0) per tonic pitch class.
_KEYS = {
    "major": [("C", 0), ("Db", -5), ("D", 2), ("Eb", -3), ("E", 4), ("F", -1),
              ("F#", 6), ("G", 1), ("Ab", -4), ("A", 3), ("Bb", -2), ("B", 5)],
    "minor": [("C", -3), ("C#", 4), ("D", -1), ("Eb", -6), ("E", 1), ("F", -4),
              ("F#", 3), ("G", -2), ("G#", 5), ("A", 0), ("Bb", -5), ("B", 2)],
}  # fmt: skip
# Keep in sync with KEY_NAMES in apps/web/lib/music.ts.
KEY_NAMES = tuple(f"{name} {mode}" for mode, table in _KEYS.items() for name, _ in table)

_LETTERS = "CDEFGAB"
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_CIRCLE = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
# Spelling of notes outside the key: sharp keys use sharps, flat keys flats, C/Am the
# most common choice for each.
_CHROMATIC = {
    1: ("C#", "C#", "Db"),
    3: ("Eb", "D#", "Eb"),
    6: ("F#", "F#", "Gb"),
    8: ("Ab", "G#", "Ab"),
    10: ("Bb", "A#", "Bb"),
}

# Chord qualities: intervals above the root, display suffix, Harte shorthand, and a
# penalty. The penalty prefers the simpler chord when two explain the notes equally
# well, and keeps power/sus/dim chords for when the notes really are just those: a
# faint third must not turn a major chord into a power chord.
_QUALITIES = {
    "maj": ((0, 4, 7), "", "maj", 0.0),
    "min": ((0, 3, 7), "m", "min", 0.0),
    "7": ((0, 4, 7, 10), "7", "7", 0.02),
    "maj7": ((0, 4, 7, 11), "maj7", "maj7", 0.02),
    "min7": ((0, 3, 7, 10), "m7", "min7", 0.02),
    "5": ((0, 7), "5", "5", 0.15),
    "sus2": ((0, 2, 7), "sus2", "sus2", 0.15),
    "sus4": ((0, 5, 7), "sus4", "sus4", 0.15),
    "dim": ((0, 3, 6), "dim", "dim", 0.15),
}

# Beat grid: phase candidates per beat and how sharply an onset must sit on a beat.
_PHASE_STEPS = 64
_PHASE_KAPPA = 4.0
# Chords: "no chord" beats score this, which a single note stays below. Tuned on the
# GuitarSet accompaniments of players 00-03.
_NO_CHORD_SCORE = 0.65
_CHORD_CHANGE_PENALTY = 0.3
_BASS_BONUS = 0.2
# Chords need this many notes sounding at once on average. Over a single melody line
# (GuitarSet solos) the labels were right only ~1/3 of the time, so solos stay unlabeled;
# accompaniments lose 2 points of root accuracy.
_MIN_POLYPHONY = 1.3
_BASS_MAX_PITCH = 52  # E3: notes up to here on the low strings count as bass notes
_DOWNBEAT_BASS_WEIGHT = 0.25


def _pc(value: int) -> int:
    return value % 12


def _tonic_pc(name: str) -> int:
    alter = name[1:].count("#") - name[1:].count("b")
    return (_NATURAL[name[0]] + alter) % 12


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class, 0 = C
    mode: str  # "major" or "minor"

    @classmethod
    def parse(cls, name: str) -> Key:
        """'A minor' -> Key(9, 'minor'); only the names in KEY_NAMES are accepted."""
        if name not in KEY_NAMES:
            raise ValueError(f"Unknown key: {name!r}")
        tonic, mode = name.split(" ")
        return cls(_tonic_pc(tonic), mode)

    @property
    def fifths(self) -> int:
        return _KEYS[self.mode][self.tonic][1]

    @property
    def tonic_name(self) -> str:
        return _KEYS[self.mode][self.tonic][0]

    @property
    def name(self) -> str:
        return f"{self.tonic_name} {self.mode}"


def spell(pc: int, key: Key | None = None) -> tuple[str, int]:
    """(letter, alteration) of a pitch class in ``key``; e.g. 10 in F major -> ('B', -1)."""
    pc = _pc(pc)
    fifths = key.fifths if key else 0
    relative_major = _CIRCLE[fifths + 7]
    letter0 = _LETTERS.index(relative_major[0])
    tonic = _tonic_pc(relative_major)
    steps = [(tonic + step) % 12 for step in _MAJOR_SCALE]
    degree = steps.index(pc) if pc in steps else None
    if degree is None and key is not None and key.mode == "minor" and pc == (key.tonic - 1) % 12:
        degree = 4  # raised 7th of the minor key (G# in A minor): the scale's 5th letter up
    if degree is not None:
        letter = _LETTERS[(letter0 + degree) % 7]
        return letter, (pc - _NATURAL[letter] + 6) % 12 - 6
    if pc not in _CHROMATIC:  # a natural outside the key (D in Ab major)
        return next(letter for letter, value in _NATURAL.items() if value == pc), 0
    name = _CHROMATIC[pc][0 if fifths == 0 else 1 if fifths > 0 else 2]
    return name[0], 1 if name.endswith("#") else -1


def pitch_class_name(pc: int, key: Key | None = None) -> str:
    letter, alter = spell(pc, key)
    return letter + ("#" * alter if alter > 0 else "b" * -alter)


@dataclass(frozen=True)
class Chord:
    start: float
    end: float
    root: int  # pitch class
    quality: str  # a key of _QUALITIES
    bass: int | None = None  # pitch class of a bass note other than the root (slash chord)

    @property
    def suffix(self) -> str:
        """'' for major, 'm', '7', 'maj7', 'm7', '5', 'sus2', 'sus4', 'dim', and with chords
        recognized in the audio also '6', 'm6', 'dim7', 'm7b5', 'aug', 'm(maj7)'."""
        if self.quality in _QUALITIES:
            return _QUALITIES[self.quality][1]
        from music_core.guitar_chords import QUALITIES  # imports this module

        return QUALITIES[self.quality][1]

    def label(self, key: Key | None = None) -> str:
        """'Bbmaj7', 'F#m', 'D/F#' ... spelled for ``key``."""
        text = pitch_class_name(self.root, key) + self.suffix
        if self.bass is not None:
            text += "/" + pitch_class_name(self.bass, key)
        return text

    @property
    def harte(self) -> str:
        """Harte-style label ('A#:maj7') for mir_eval, without the bass."""
        quality = _QUALITIES[self.quality][2] if self.quality in _QUALITIES else self.quality
        return f"{pitch_class_name(self.root, Key(0, 'major'))}:{quality}"


@dataclass(frozen=True)
class Analysis:
    tempo: float  # BPM; with tracked beats, of the median beat
    beats_per_measure: int
    downbeat: float  # the first bar line at or after 0 s
    key: Key | None  # the given key, else the song's main key or the estimate
    estimated_key: Key | None
    chords: list[Chord]
    grid: BeatGrid  # every beat and bar line, covering the notes
    tracked: bool  # the grid follows tracked beats rather than one tempo
    keys: list[KeySpan] = field(default_factory=list)  # a song's keys, if it modulates
    rhythm: Rhythm | None = None  # the strumming pattern


def analyze(
    notes: Sequence[Note],
    tempo: float | None,
    beats_per_measure: int = 4,
    key: Key | None = None,
    downbeat: float | None = None,
    beats: Sequence[float] | None = None,
    downbeats: Sequence[float] | None = None,
    end: float | None = None,
    chord_scores: Sequence[dict] | None = None,
    keys: Sequence[KeySpan] | None = None,
    strums: Sequence[tuple[float, float]] | None = None,
) -> Analysis:
    """Key, bar grid, chords and strumming pattern of ``notes`` in ``beats_per_measure``/4.

    The grid follows ``beats`` (tracked in the recording, seconds) when there are any,
    with bar lines where most of the tracked ``downbeats`` fall; otherwise one tempo
    (``tempo`` BPM) with the beat phase and bar lines found from the notes. ``key`` and
    ``downbeat`` (the time of any bar line) override the estimates. The grid covers the
    notes and, if given, up to ``end`` seconds.

    ``chord_scores`` (a chord recognizer's best labels per beat: ``{"start": s, "top":
    [[harte, log-probability], ...]}``) are fused with the notes' chords. ``keys`` are a
    song's keys over time; the longest is its key. ``strums`` ((time, strength) pairs)
    give the strumming pattern; without them, the notes' chords do.
    """
    notes = [n for n in notes if n.end > n.start]
    beats = [float(b) for b in beats] if beats is not None else []
    keys = list(keys or [])
    estimated = estimate_key(notes)
    if keys:
        estimated = max(keys, key=lambda s: s.end - s.start).key
    start = min((n.start for n in notes), default=0.0)
    stop = max([n.end for n in notes] + [end or 0.0, *beats[-1:]] + [1.0])
    has_downbeat = downbeat is not None and math.isfinite(downbeat)
    tracked = len(beats) >= 2
    if tracked:
        grid = BeatGrid.tracked(beats, beats_per_measure, 0, start, stop)
        if has_downbeat:
            grid = grid.with_bar_line_at(downbeat)
        elif downbeats is not None and len(downbeats):
            grid = replace(grid, phase=downbeat_phase(grid.beats, downbeats, beats_per_measure))
        else:
            features = _BeatGrid.build(notes, grid.beats)
            if features is not None:
                grid = replace(grid, phase=features.downbeat_beat(beats_per_measure))
        tempo = median_tempo(beats)
    else:
        if tempo is None or not (math.isfinite(tempo) and tempo > 0):
            tempo = 120.0
        period = 60.0 / tempo
        phase = (downbeat if has_downbeat else beat_phase(notes, tempo)) % period
        grid = BeatGrid.fixed(tempo, beats_per_measure, phase, start, stop)
        if has_downbeat:
            grid = grid.with_bar_line_at(downbeat)
        else:
            features = _BeatGrid.build(notes, grid.beats)
            if features is not None:
                grid = replace(grid, phase=features.downbeat_beat(beats_per_measure))
    features = _BeatGrid.build(notes, grid.beats)
    if features is None:
        chords = []
    elif chord_scores:
        chords = features.fused_chords(chord_scores)
    else:
        chords = features.chords()
    if strums is None:  # a melody line's double stops are no strumming pattern
        strums = strums_from_notes(notes) if features and features.polyphonic() else []
    return Analysis(
        tempo=tempo,
        beats_per_measure=beats_per_measure,
        downbeat=next((t for t in grid.bar_lines if t >= -1e-9), grid.bar_lines[-1]),
        key=key or estimated,
        estimated_key=estimated,
        chords=chords,
        grid=grid,
        tracked=tracked,
        keys=keys,
        rhythm=strum_rhythm(strums, grid),
    )


def estimate_key(notes: Sequence[Note]) -> Key | None:
    """Best-correlating major/minor key profile; None without notes."""
    histogram = np.zeros(12)
    for note in notes:
        # Capped so one long ringing note doesn't outweigh everything else.
        histogram[_pc(note.pitch)] += min(note.end - note.start, 2.0)
    if histogram.sum() <= 0 or np.ptp(histogram) == 0:
        return None
    best, best_score = None, -np.inf
    for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        for tonic in range(12):
            score = np.corrcoef(histogram, np.roll(profile, tonic))[0, 1]
            if score > best_score:
                best, best_score = Key(tonic, mode), score
    return best


_KEY_TABLE = [Key(tonic, mode) for mode in ("major", "minor") for tonic in range(12)]
_KEY_PROFILES = np.array(
    [np.roll(_MAJOR_PROFILE if k.mode == "major" else _MINOR_PROFILE, k.tonic) for k in _KEY_TABLE]
)
# Modulations cost this much (in summed profile correlation). 1 found AAM's section keys
# best, but invented a key change in 15% of GuitarSet's one-key excerpts; 2 keeps most of
# the gain on AAM and invents almost none (scripts/benchmark_keys.py).
KEY_CHANGE_PENALTY = 2.0
_KEY_CONTEXT = 2  # bars on each side that a bar's profile is pooled with


@dataclass(frozen=True)
class KeySpan:
    start: float
    end: float
    key: Key


def segment_keys(
    notes: Sequence[Note],
    bars: Sequence[float],
    change_penalty: float = KEY_CHANGE_PENALTY,
) -> list[KeySpan]:
    """The key over time, for songs that modulate.

    Each bar's duration-weighted pitch-class profile, pooled with ``_KEY_CONTEXT`` bars on
    either side, is correlated with the 24 key profiles; a Viterbi path over the bars
    chooses the keys, paying ``change_penalty`` for every change. ``bars`` are the bar
    lines (seconds); the first span starts at 0 and the last ends with the last note.
    One span when nothing changes. Empty without notes.
    """
    notes = [n for n in notes if n.end > n.start]
    if not notes:
        return []
    end = max(n.end for n in notes)
    bounds = sorted({0.0, *[b for b in bars if 0.0 < b < end], end})
    profiles = np.zeros((len(bounds) - 1, 12))
    starts = np.array(bounds[:-1])
    for note in notes:
        first = max(0, int(np.searchsorted(starts, note.start, side="right")) - 1)
        for row in range(first, len(starts)):
            a, b = bounds[row], bounds[row + 1]
            if a >= note.end:
                break
            overlap = min(note.end, b) - max(note.start, a)
            if overlap > 0:
                profiles[row, _pc(note.pitch)] += overlap
    cum = np.vstack([np.zeros(12), np.cumsum(profiles, axis=0)])
    rows = len(profiles)
    scores = np.zeros((rows, len(_KEY_TABLE)))
    for row in range(rows):
        lo, hi = max(0, row - _KEY_CONTEXT), min(rows, row + _KEY_CONTEXT + 1)
        pooled = cum[hi] - cum[lo]
        if pooled.sum() > 0 and np.ptp(pooled) > 0:
            centered = pooled - pooled.mean()
            profiles_c = _KEY_PROFILES - _KEY_PROFILES.mean(axis=1, keepdims=True)
            scores[row] = (
                profiles_c
                @ centered
                / (np.linalg.norm(profiles_c, axis=1) * np.linalg.norm(centered))
            )
    path = _viterbi_keys(scores, change_penalty)
    spans: list[KeySpan] = []
    for row, state in enumerate(path):
        key = _KEY_TABLE[state]
        if spans and spans[-1].key == key:
            spans[-1] = KeySpan(spans[-1].start, bounds[row + 1], key)
        else:
            spans.append(KeySpan(bounds[row], bounds[row + 1], key))
    return spans


def _viterbi_keys(scores: np.ndarray, change_penalty: float) -> list[int]:
    total = scores[0].copy()
    back = np.zeros(scores.shape, dtype=int)
    for row in range(1, len(scores)):
        best = int(np.argmax(total))
        switch = total[best] - change_penalty
        stay = total >= switch
        back[row] = np.where(stay, np.arange(scores.shape[1]), best)
        total = np.where(stay, total, switch) + scores[row]
    path = [int(np.argmax(total))]
    for row in range(len(scores) - 1, 0, -1):
        path.append(int(back[row, path[-1]]))
    return path[::-1]


def _weighted_onsets(notes: Sequence[Note]) -> tuple[np.ndarray, np.ndarray]:
    """Onset times and how strongly each marks a beat: long notes and bass notes fall on
    beats more often than passing notes."""
    onsets = np.array([n.start for n in notes])
    weights = np.array(
        [(1 + (n.pitch <= _BASS_MAX_PITCH)) * min(n.end - n.start, 1.0) for n in notes]
    )
    return onsets, weights


def refine_tempo(notes: Sequence[Note], tempo: float, spread: float = 0.04) -> float:
    """Fine-tune a rough tempo estimate to the notes, within ``spread`` (4 %).

    Audio tempo estimators work on a coarse lag grid, but 1 % off already shifts the bar
    grid by a whole beat every 100 beats. The refined tempo maximizes the beat and
    half-beat periodicity of the onsets: on GuitarSet transcriptions whose estimate was
    within 4 %, it got 94 % of them within 0.3 % of the true tempo, versus 14 % before.
    """
    notes = [n for n in notes if n.end > n.start]
    if len(notes) < 2 or not (math.isfinite(tempo) and tempo > 0):
        return tempo
    onsets, weights = _weighted_onsets(notes)
    best, best_score = tempo, -1.0
    for candidate in tempo * (1 + np.linspace(-spread, spread, 161)):
        angle = 2 * np.pi * onsets * candidate / 60.0
        score = abs(weights @ np.exp(1j * angle)) + abs(weights @ np.exp(2j * angle))
        if score > best_score:
            best, best_score = float(candidate), score
    return best


def beat_phase(notes: Sequence[Note], tempo: float) -> float:
    """Offset in [0, beat) of the beat grid that best lines up with the onsets."""
    if not notes:
        return 0.0
    period = 60.0 / tempo
    onsets, weights = _weighted_onsets(notes)
    candidates = np.arange(_PHASE_STEPS) / _PHASE_STEPS * period
    angle = 2 * np.pi * (onsets[:, None] - candidates[None, :]) / period
    kernel = np.exp(_PHASE_KAPPA * (np.cos(angle) - 1))
    score = (weights[:, None] * kernel).sum(axis=0)
    return float(candidates[int(np.argmax(score))])


@dataclass
class _BeatGrid:
    """Per-beat features; row ``i`` is the beat from ``beats[i]`` to ``beats[i + 1]``."""

    beats: np.ndarray  # beat times (seconds), increasing
    chroma: np.ndarray  # (rows, 12): seconds each pitch class sounds within the beat
    bass: np.ndarray  # (rows,): pitch class of the lowest sounding note, -1 if silent
    bass_onset: np.ndarray  # (rows,): low notes starting on (near) the beat

    @classmethod
    def build(cls, notes: Sequence[Note], beats: Sequence[float]) -> _BeatGrid | None:
        beats = np.asarray(beats, dtype=float)
        if not notes or len(beats) < 2:
            return None
        size = len(beats) - 1
        index = np.arange(len(beats), dtype=float)
        chroma = np.zeros((size, 12))
        lowest = np.full(size, 128)
        bass_onset = np.zeros(size)
        for note in notes:
            pos = float(np.interp(note.start, beats, index))
            nearest = round(pos)
            if note.pitch <= _BASS_MAX_PITCH and abs(pos - nearest) < 0.2 and nearest < size:
                bass_onset[nearest] += 1
            b0 = int(np.searchsorted(beats, note.start, side="right")) - 1
            b1 = int(np.searchsorted(beats, note.end, side="right")) - 1
            for b in range(max(b0, 0), min(b1, size - 1) + 1):
                overlap = min(note.end, beats[b + 1]) - max(note.start, beats[b])
                if overlap > 0:
                    chroma[b, _pc(note.pitch)] += overlap
                    lowest[b] = min(lowest[b], note.pitch)
        bass = np.where(lowest < 128, lowest % 12, -1)
        return cls(beats, chroma, bass, bass_onset)

    def time(self, row: int) -> float:
        return float(self.beats[row])

    def polyphonic(self) -> bool:
        """Whether the notes are chords rather than one melody line: on average at least
        ``_MIN_POLYPHONY`` sounding wherever something sounds."""
        sounding = self.chroma.sum(axis=1) > 0
        if not sounding.any():
            return False
        return self.chroma.sum() / np.diff(self.beats)[sounding].sum() >= _MIN_POLYPHONY

    def downbeat_beat(self, beats_per_measure: int) -> int:
        """Which beats are bar lines: the ``r`` in [0, beats_per_measure) for which rows
        ``r``, ``r + beats_per_measure``, ... start bars."""
        sounding = np.flatnonzero(self.chroma.sum(axis=1) > 0)
        if beats_per_measure <= 1 or not len(sounding):
            return 0
        lo, hi = int(sounding[0]), int(sounding[-1]) + 1  # the beats with notes
        size = hi - lo
        if size < beats_per_measure:
            return lo % beats_per_measure
        # Chords mostly change on bar lines: harmonic novelty, i.e. how different the
        # next bar sounds from the previous one, is the main cue; bass notes help a bit.
        chroma = self.chroma[lo:hi]
        cum = np.vstack([np.zeros(12), np.cumsum(chroma, axis=0)])
        novelty = np.zeros(size)
        for row in range(1, size):
            before = cum[row] - cum[max(0, row - beats_per_measure)]
            after = cum[min(size, row + beats_per_measure)] - cum[row]
            norm = np.linalg.norm(before) * np.linalg.norm(after)
            if norm > 0:
                novelty[row] = 1 - before @ after / norm
        evidence = _zscore(novelty) + _DOWNBEAT_BASS_WEIGHT * _zscore(self.bass_onset[lo:hi])
        beat_of_bar = np.arange(lo, hi) % beats_per_measure
        scores = [evidence[beat_of_bar == r].mean() for r in range(beats_per_measure)]
        return int(np.argmax(scores))

    def chords(self) -> list[Chord]:
        labels = _decode_chords(self.chroma, self.bass)
        chords: list[Chord] = []
        row = 0
        while row < len(labels):
            end = row
            while end + 1 < len(labels) and labels[end + 1] == labels[row]:
                end += 1
            length = self.beats[end + 1] - self.beats[row]
            polyphony = self.chroma[row : end + 1].sum() / length if length > 0 else 0.0
            if labels[row] is not None and polyphony >= _MIN_POLYPHONY:
                root, quality = labels[row]
                chords.append(
                    Chord(
                        start=max(0.0, self.time(row)),
                        end=self.time(end + 1),
                        root=root,
                        quality=quality,
                        bass=self._slash_bass(row, end, root, quality),
                    )
                )
            row = end + 1
        return chords

    def fused_chords(self, scores: Sequence[dict]) -> list[Chord]:
        """Chords from a recognizer's per-beat labels (``analyze``'s ``chord_scores``) and
        the notes together: each beat's log-probabilities plus ``FUSION_WEIGHT`` times
        how well the notes match each chord, Viterbi over the beats. On GuitarSet player
        05 this got 0.66 of the time's major/minor chords right, against 0.64 for the
        recognizer and 0.63 for the notes alone (weight chosen on players 00-03).

        A single melody line (a solo) still gets no chords; the rest of the recording
        gets them where notes sound, even where only a few do (an arpeggio, a bass run).
        """
        from music_core.guitar_chords import ChordSymbol

        if not self.polyphonic():
            return []
        sounding = self.chroma.sum(axis=1) > 0
        starts = np.array([float(s["start"]) for s in scores])
        vocab = sorted({label for s in scores for label, _ in s["top"]} - {"N", "X"})
        symbols = [ChordSymbol.parse(label) for label in vocab]
        templates = np.zeros((len(vocab) + 1, 12))  # the last state: no chord
        for i, symbol in enumerate(symbols):
            templates[i, list(symbol.tones)] = 1.0
        norms = np.linalg.norm(templates, axis=1, keepdims=True)
        templates = np.divide(templates, norms, out=np.zeros_like(templates), where=norms > 0)
        index = {label: i for i, label in enumerate(vocab)} | {"N": len(vocab)}

        rows = len(self.beats) - 1
        emission = np.empty((rows, len(vocab) + 1))
        for row in range(rows):
            middle = (self.beats[row] + self.beats[row + 1]) / 2
            k = min(max(int(np.searchsorted(starts, middle, side="right")) - 1, 0), len(scores) - 1)
            top = [(label, float(p)) for label, p in scores[k]["top"] if label in index]
            emission[row] = min((p for _, p in top), default=0.0) - 1.0
            for label, p in top:
                emission[row, index[label]] = p
        chroma_norms = np.linalg.norm(self.chroma, axis=1, keepdims=True)
        unit = np.divide(
            self.chroma, chroma_norms, out=np.zeros_like(self.chroma), where=chroma_norms > 0
        )
        emission += FUSION_WEIGHT * unit @ templates.T
        path = _viterbi(emission, FUSION_CHANGE_PENALTY)

        chords: list[Chord] = []
        row = 0
        while row < rows:
            end = row
            while end + 1 < rows and path[end + 1] == path[row]:
                end += 1
            state = path[row]
            if state < len(vocab) and sounding[row : end + 1].any():
                symbol = symbols[state]
                chords.append(
                    Chord(
                        start=max(0.0, self.time(row)),
                        end=self.time(end + 1),
                        root=symbol.root,
                        quality=symbol.quality,
                        bass=self._slash_bass(row, end, symbol.root, symbol.quality),
                    )
                )
            row = end + 1
        return chords

    def _slash_bass(self, row: int, end: int, root: int, quality: str) -> int | None:
        weights = np.zeros(12)
        for b in range(row, end + 1):
            if self.bass[b] >= 0:
                weights[self.bass[b]] += self.chroma[b, self.bass[b]]
        if weights.sum() == 0:
            return None
        bass = int(np.argmax(weights))
        tones = {(root + i) % 12 for i in _intervals(quality)}
        return bass if bass != root and bass in tones else None


def _intervals(quality: str) -> tuple[int, ...]:
    if quality in _QUALITIES:
        return _QUALITIES[quality][0]
    from music_core.guitar_chords import QUALITIES  # imports this module

    return QUALITIES[quality][0]


# Fusing a chord recognizer with the notes (``_BeatGrid.fused_chords``): the weight of the
# notes' template match against the recognizer's log-probabilities, chosen on GuitarSet
# players 00-03, and the change penalty that suited the recognizer alone.
FUSION_WEIGHT = 4.0
FUSION_CHANGE_PENALTY = 1.0


def _viterbi(scores: np.ndarray, change_penalty: float) -> list[int]:
    """Best state sequence for per-step ``scores`` when changing state costs ``change_penalty``."""
    if not len(scores):
        return []
    total = scores[0].copy()
    back = np.zeros(scores.shape, dtype=np.int32)
    stay = np.arange(scores.shape[1])
    for t in range(1, len(scores)):
        best = int(total.argmax())
        switch = total[best] - change_penalty > total
        back[t] = np.where(switch, best, stay)
        total = np.where(switch, total[best] - change_penalty, total) + scores[t]
    path = [int(total.argmax())]
    for t in range(len(scores) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def _zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def _chord_templates() -> tuple[list[tuple[int, str]], np.ndarray, np.ndarray]:
    labels, templates, penalties = [], [], []
    for quality, (intervals, _, _, penalty) in _QUALITIES.items():
        for root in range(12):
            template = np.zeros(12)
            template[[(root + i) % 12 for i in intervals]] = 1
            labels.append((root, quality))
            templates.append(template / np.linalg.norm(template))
            penalties.append(penalty)
    return labels, np.array(templates), np.array(penalties)


_TEMPLATE_LABELS, _TEMPLATES, _PENALTIES = _chord_templates()
_TEMPLATE_ROOTS = np.array([root for root, _ in _TEMPLATE_LABELS])


def _decode_chords(chroma: np.ndarray, bass: np.ndarray) -> list[tuple[int, str] | None]:
    """Viterbi over (chord templates + "no chord") with a constant chord-change penalty."""
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    unit = np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 0)
    emission = unit @ _TEMPLATES.T - _PENALTIES
    emission += _BASS_BONUS * (bass[:, None] == _TEMPLATE_ROOTS[None, :])
    emission = np.hstack([emission, np.full((len(chroma), 1), _NO_CHORD_SCORE)])

    size, states = emission.shape
    score = emission[0].copy()
    back = np.zeros((size, states), dtype=int)
    for row in range(1, size):
        best = int(np.argmax(score))
        switch = score[best] - _CHORD_CHANGE_PENALTY
        stay = score >= switch
        back[row] = np.where(stay, np.arange(states), best)
        score = np.where(stay, score, switch) + emission[row]
    path = [int(np.argmax(score))]
    for row in range(size - 1, 0, -1):
        path.append(back[row, path[-1]])
    path.reverse()
    return [_TEMPLATE_LABELS[s] if s < len(_TEMPLATE_LABELS) else None for s in path]
