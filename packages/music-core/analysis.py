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
from dataclasses import dataclass

import numpy as np

from music_core.notes import Note

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
        """'' for major, 'm', '7', 'maj7', 'm7', '5', 'sus2', 'sus4' or 'dim'."""
        return _QUALITIES[self.quality][1]

    def label(self, key: Key | None = None) -> str:
        """'Bbmaj7', 'F#m', 'D/F#' ... spelled for ``key``."""
        text = pitch_class_name(self.root, key) + self.suffix
        if self.bass is not None:
            text += "/" + pitch_class_name(self.bass, key)
        return text

    @property
    def harte(self) -> str:
        """Harte-style label ('A#:maj7') for mir_eval, without the bass."""
        return f"{pitch_class_name(self.root, Key(0, 'major'))}:{_QUALITIES[self.quality][2]}"


@dataclass(frozen=True)
class Analysis:
    tempo: float
    beats_per_measure: int
    downbeat: float  # a bar line in [0, bar length); bar lines at downbeat + k * bar
    key: Key | None  # the given key, else the estimate
    estimated_key: Key | None
    chords: list[Chord]


def analyze(
    notes: Sequence[Note],
    tempo: float | None,
    beats_per_measure: int = 4,
    key: Key | None = None,
    downbeat: float | None = None,
) -> Analysis:
    """Key, bar grid and chords of ``notes`` at ``tempo`` (BPM) in ``beats_per_measure``/4.

    ``key`` and ``downbeat`` (the time of any bar line) override the estimates.
    """
    if tempo is None or not (math.isfinite(tempo) and tempo > 0):
        tempo = 120.0
    notes = [n for n in notes if n.end > n.start]
    estimated = estimate_key(notes)
    period = 60.0 / tempo
    bar = beats_per_measure * period
    if downbeat is not None and math.isfinite(downbeat):
        phase = downbeat % period
        grid = _BeatGrid.build(notes, phase, period)
        downbeat %= bar
    else:
        phase = beat_phase(notes, tempo)
        grid = _BeatGrid.build(notes, phase, period)
        downbeat_beat = grid.downbeat_beat(beats_per_measure) if grid else 0
        downbeat = (phase + downbeat_beat * period) % bar
    return Analysis(
        tempo=tempo,
        beats_per_measure=beats_per_measure,
        downbeat=downbeat,
        key=key or estimated,
        estimated_key=estimated,
        chords=grid.chords() if grid else [],
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
    """Per-beat features on the grid ``phase + (first + i) * period``."""

    period: float
    phase: float
    first: int  # beat index of row 0
    chroma: np.ndarray  # (beats, 12): seconds each pitch class sounds within the beat
    bass: np.ndarray  # (beats,): pitch class of the lowest sounding note, -1 if silent
    bass_onset: np.ndarray  # (beats,): low notes starting on (near) the beat

    @classmethod
    def build(cls, notes: Sequence[Note], phase: float, period: float) -> _BeatGrid | None:
        if not notes:
            return None
        first = math.floor((min(n.start for n in notes) - phase) / period)
        last = math.floor((max(n.end for n in notes) - phase) / period)
        size = last - first + 1
        chroma = np.zeros((size, 12))
        lowest = np.full(size, 128)
        bass_onset = np.zeros(size)
        for note in notes:
            pos = (note.start - phase) / period
            nearest = round(pos)
            if note.pitch <= _BASS_MAX_PITCH and abs(pos - nearest) < 0.2 and nearest <= last:
                bass_onset[nearest - first] += 1
            b0 = math.floor(pos) - first
            b1 = math.floor((note.end - phase) / period) - first
            for b in range(max(b0, 0), min(b1, size - 1) + 1):
                beat_start = phase + (first + b) * period
                overlap = min(note.end, beat_start + period) - max(note.start, beat_start)
                if overlap > 0:
                    chroma[b, _pc(note.pitch)] += overlap
                    lowest[b] = min(lowest[b], note.pitch)
        bass = np.where(lowest < 128, lowest % 12, -1)
        return cls(period, phase, first, chroma, bass, bass_onset)

    def time(self, row: int) -> float:
        return self.phase + (self.first + row) * self.period

    def downbeat_beat(self, beats_per_measure: int) -> int:
        """Which beat of the bar (0 .. beats_per_measure-1 from ``phase``) is the downbeat."""
        size = len(self.bass)
        if beats_per_measure <= 1 or size < beats_per_measure:
            return 0
        # Chords mostly change on bar lines: harmonic novelty, i.e. how different the
        # next bar sounds from the previous one, is the main cue; bass notes help a bit.
        cum = np.vstack([np.zeros(12), np.cumsum(self.chroma, axis=0)])
        novelty = np.zeros(size)
        for row in range(1, size):
            before = cum[row] - cum[max(0, row - beats_per_measure)]
            after = cum[min(size, row + beats_per_measure)] - cum[row]
            norm = np.linalg.norm(before) * np.linalg.norm(after)
            if norm > 0:
                novelty[row] = 1 - before @ after / norm
        evidence = _zscore(novelty) + _DOWNBEAT_BASS_WEIGHT * _zscore(self.bass_onset)
        beat_of_bar = (self.first + np.arange(size)) % beats_per_measure
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
            polyphony = self.chroma[row : end + 1].sum() / ((end + 1 - row) * self.period)
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

    def _slash_bass(self, row: int, end: int, root: int, quality: str) -> int | None:
        weights = np.zeros(12)
        for b in range(row, end + 1):
            if self.bass[b] >= 0:
                weights[self.bass[b]] += self.chroma[b, self.bass[b]]
        if weights.sum() == 0:
            return None
        bass = int(np.argmax(weights))
        tones = {(root + i) % 12 for i in _QUALITIES[quality][0]}
        return bass if bass != root and bass in tones else None


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
