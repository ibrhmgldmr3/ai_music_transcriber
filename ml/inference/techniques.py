"""Playing techniques from the pitch curve: bends, vibrato and slides.

The note model says which pitch sounds when, in whole semitones. How a note was played
shows in its exact pitch over time:

* bend: the pitch rises from the fretted note and is held there. It gets there
  gradually; a new fretted note jumps.
* vibrato: the pitch swings regularly around the note, 4-9 times a second.
* slide: between two notes on one string the pitch glides through the frets in
  between instead of jumping to the next note.

Each note's pitch curve is followed with a harmonic sum. For candidate pitches around
the note (10-cent steps), the spectrum's magnitude at the first harmonics is summed, the
strongest candidate wins and a parabola refines it. Other notes sounding at the same
time are masked out. On GuitarSet's solos (the microphone recording against the
per-string pitch contours of the hexaphonic pickup) the median error is 2-4 cents.

Hammer-ons and pull-offs are not detected: the cue they leave (a softer attack than a
picked note) did not separate them from picked notes on GuitarSet, and no recording
with labeled hammer-ons and pull-offs is used yet. They can be marked in the editor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import librosa
import numpy as np
from scipy import ndimage

from music_core.notes import Note
from music_core.tab import DEFAULT_NUM_FRETS


@dataclass(frozen=True)
class TechniqueSettings:
    hop_length: int = 256  # 5.8 ms at 44.1 kHz, GuitarSet's contour rate
    n_fft: int = 2048
    harmonics: int = 6
    step: float = 0.1  # semitones between candidate pitches
    below: float = 2.5  # candidates this far below a note (one named after its bend)
    above: float = 3.5  # ... and above it (bends go up, up to a step and a half)
    mask_width: float = 0.6  # semitones around another sounding note left out
    jump: float = 1.0  # cost of a pitch change of d semitones between frames: jump * d
    attack: float = 0.015  # seconds skipped at a note's start (the pick's noise)
    release: float = 0.01  # and at its end
    weak: float = 0.2  # frames with less harmonic energy than this x the note's median
    # Bends: the pitch held at least bend_min semitones above the fretted pitch for
    # bend_hold seconds, reached in at least bend_rise seconds (15 % -> 85 %) without a
    # step of more than bend_step of the bend between two frames. Chosen on IDMT-SMT-
    # Guitar's AR and FS guitars; a 40 ms hold scored a little higher there but caught
    # GuitarSet notes whose end runs into the next note's start.
    bend_min: float = 0.75
    bend_max: float = 3.0
    bend_hold: float = 0.06
    bend_rise: float = 0.03
    # A slide climbs fret by fret: its largest step between frames is a median 31 % of
    # the way on IDMT (bends: 16 %). Without this, 47 % of IDMT's slides were bends.
    bend_step: float = 0.25
    base_window: float = 0.06  # seconds at the note's start that give its fretted pitch
    base_percentile: float = 10.0  # the start's lowest part
    base_tolerance: float = 0.4  # semitones the start may be off a fretted note
    bend_below: int = 2  # semitones below the transcribed pitch a bend may start
    # Vibrato: a regular swing of at least vibrato_depth semitones peak to peak. Chosen
    # on IDMT's AR and FS guitars (the depth of its vibrato notes: median 0.22). Slower
    # swings than 4 Hz (IDMT's slow vibrato takes, 2-3 Hz) cost more than they found.
    vibrato_rates: tuple[float, float] = (4.0, 9.0)
    vibrato_depth: float = 0.15
    vibrato_regularity: float = 0.2  # autocorrelation at the swing's period
    vibrato_min: float = 0.4  # seconds of note heard
    # Slides between two notes: the second starts where the first ends (the sound never
    # stops, and the note model ends the first note on the frame the second begins; on
    # IDMT the notes of picked changes mistaken for slides had a median 46 ms gap), the
    # pitch spends slide_glide to slide_glide_max seconds between the two notes' pitches
    # and moves slide_span semitones towards the second meanwhile. Chosen on IDMT's AR
    # and FS guitars: among the note pairs it considered, 85 % of those marked were
    # labeled slides (13 % before the gap and duration limits).
    slide_gap: float = 0.012
    slide_overlap: float = 0.05
    slide_glide: float = 0.02
    slide_glide_max: float = 0.08
    slide_span: float = 0.2  # semitones the pitch moves while between the two notes
    # Seconds searched before the earlier and after the later of the first note's end
    # and the second's start.
    slide_window: tuple[float, float] = (0.08, 0.05)
    slide_max: int = 12  # semitones
    # Seconds searched before the earlier and after the later of the first note's end
    # and the second's start.
    slide_window: tuple[float, float] = (0.08, 0.05)
    chord_tolerance: float = 0.05  # notes starting together form a chord: no slide


@dataclass
class PitchCurve:
    """Fractional MIDI pitch per frame from ``start`` (a frame index); NaN = not heard."""

    start: int
    midi: np.ndarray


class PitchTracker:
    """Harmonic-sum pitch following on one recording's magnitude spectrogram."""

    def __init__(self, y: np.ndarray, sample_rate: int, s: TechniqueSettings = TechniqueSettings()):
        self.s = s
        self.sample_rate = sample_rate
        self.frame_rate = sample_rate / s.hop_length
        # Square-root magnitudes keep quieter upper harmonics in the sum.
        self.spec = np.sqrt(np.abs(librosa.stft(y, n_fft=s.n_fft, hop_length=s.hop_length)))
        self._weights = 1.0 / np.sqrt(np.arange(1, s.harmonics + 1))
        # Semitones the recording sounds above concert pitch (A = 440 Hz): pitch curves
        # are given relative to the guitar's own tuning (see ``tune``).
        self.offset = 0.0

    def frame(self, seconds: float) -> int:
        return int(round(seconds * self.frame_rate))

    def track(
        self,
        f0: int,
        f1: int,
        low: float,
        high: float,
        others: Sequence[Note] = (),
    ) -> PitchCurve:
        """The strongest pitch in [low, high] (MIDI) in frames [f0, f1), with the pitches
        of ``others`` left out where they sound."""
        s = self.s
        f0, f1 = max(0, f0), min(self.spec.shape[1], f1)
        if f1 <= f0:
            return PitchCurve(f0, np.zeros(0))
        candidates = np.arange(low, high + 1e-9, s.step) + self.offset
        hz = librosa.midi_to_hz(candidates)
        bins = hz[:, None] * np.arange(1, s.harmonics + 1)[None, :] * s.n_fft / self.sample_rate
        inside = bins < self.spec.shape[0] - 1
        lower = np.clip(np.floor(bins).astype(int), 0, self.spec.shape[0] - 2)
        frac = (bins - lower)[..., None]
        segment = self.spec[:, f0:f1]
        values = segment[lower] * (1 - frac) + segment[lower + 1] * frac  # (C, H, T)
        score = np.einsum("cht,h->ct", values * inside[..., None], self._weights)

        for other in others:
            a, b = max(self.frame(other.start), f0), min(self.frame(other.end), f1)
            if b > a:
                masked = np.abs(candidates - self.offset - other.pitch) <= s.mask_width
                score[masked, a - f0 : b - f0] = 0.0

        cols = np.arange(score.shape[1])
        k = np.clip(_continuous_path(score, candidates, s.jump), 1, len(candidates) - 2)
        y0, y1, y2 = score[k - 1, cols], score[k, cols], score[k + 1, cols]
        curvature = y0 - 2 * y1 + y2
        shift = np.divide(
            0.5 * (y0 - y2), curvature, out=np.zeros_like(y1), where=np.abs(curvature) > 1e-12
        )
        midi = candidates[k] + np.clip(shift, -0.5, 0.5) * s.step
        strength = score.max(axis=0)
        midi[strength < s.weak * max(float(np.median(strength)), 1e-12)] = np.nan
        return PitchCurve(f0, midi - self.offset)

    def tune(self, curves: Sequence[np.ndarray], pitches: Sequence[int]) -> float:
        """Measure the recording's tuning from its notes' curves (taken at ``offset``)
        and follow it from now on. Guitars are often a fraction of a semitone off
        A = 440 Hz (in one IDMT take most strings are 35-45 cents sharp), which would
        move every fixed threshold."""
        deviations = np.concatenate(
            [c[np.abs(c - p) < 0.5] - p for c, p in zip(curves, pitches)] or [np.zeros(0)]
        )
        if len(deviations) >= 50:
            shift = float(np.median(deviations))
            self.offset += shift
            return shift
        return 0.0

    def note_curve(self, note: Note, others: Sequence[Note] = ()) -> PitchCurve:
        s = self.s
        return self.track(
            self.frame(note.start + s.attack),
            self.frame(note.end - s.release),
            note.pitch - s.below,
            note.pitch + s.above,
            others,
        )


def _continuous_path(score: np.ndarray, candidates: np.ndarray, jump: float) -> np.ndarray:
    """Candidate index per frame along the strongest path that rarely changes pitch.

    Frame by frame, the strongest candidate jumps to whatever sounds loudest (another
    string, the next note as this one fades). Here each frame's scores, relative to its
    best, are log-likelihoods, and a change of ``d`` semitones between frames costs
    ``jump * d``, so short excursions to another pitch and back don't pay off. The cost
    is linear: a squared cost made jumps cheaper in small steps and turned them into
    bend-like ramps (bend F-measure on GuitarSet players 00-01 fell from 0.26 to 0.17).
    """
    n, t = score.shape
    if t == 0:
        return np.zeros(0, dtype=int)
    peak = np.maximum(score.max(axis=0, keepdims=True), 1e-12)
    emission = np.log(np.maximum(score / peak, 1e-3))
    penalty = jump * np.abs(candidates[:, None] - candidates[None, :])  # (from, to)
    total = emission[:, 0].copy()
    back = np.zeros((t, n), dtype=int)
    for i in range(1, t):
        options = total[:, None] - penalty
        back[i] = options.argmax(axis=0)
        total = options[back[i], np.arange(n)] + emission[:, i]
    path = np.zeros(t, dtype=int)
    path[-1] = int(total.argmax())
    for i in range(t - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _smooth(midi: np.ndarray, width: int = 5) -> np.ndarray:
    """Median filter that ignores NaN frames (they stay NaN)."""
    if len(midi) == 0:
        return midi
    filled = midi.copy()
    missing = np.isnan(filled)
    if missing.all():
        return filled
    filled[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing), midi[~missing])
    out = ndimage.median_filter(filled, size=width, mode="nearest")
    out[missing] = np.nan
    return out


@dataclass(frozen=True)
class Bend:
    fretted: int  # the note the string is fretted at
    amount: float  # semitones it is bent up (rounded to a half)


def find_bend(
    midi: np.ndarray, pitch: int, frame_rate: float, s: TechniqueSettings = TechniqueSettings()
) -> Bend | None:
    """The bend of a note, or None when it isn't bent.

    ``midi`` is the note's pitch curve and ``pitch`` the note as transcribed: the fretted
    note, or the bent one when the bend comes quickly (the note model names a note by
    the pitch it hears most; a fifth of IDMT's bends came out only at a higher pitch).
    The curve then starts up to ``bend_below`` semitones lower.
    """
    curve = _smooth(midi)
    n_base = max(3, int(round(s.base_window * frame_rate)))
    hold = max(3, int(round(s.bend_hold * frame_rate)))
    if np.count_nonzero(~np.isnan(curve)) < n_base + hold:
        return None
    early = curve[:n_base]
    if np.isnan(early).all():
        return None
    # Players often start bending as they pick, and the note model starts a note a few
    # frames late: the lowest part of the start is the fretted note (on IDMT's licks the
    # median of the first 40 ms was already 0.15-0.5 semitones up).
    base = float(np.nanpercentile(early, s.base_percentile))
    fretted = int(round(base))
    if abs(base - fretted) > s.base_tolerance or not pitch - s.bend_below <= fretted <= pitch:
        return None  # starts away from any fretted note it could be bent from
    rise = np.nan_to_num(curve - fretted, nan=-np.inf)
    held = ndimage.minimum_filter1d(rise, hold, mode="nearest")
    amount = float(held.max())
    if not s.bend_min <= amount <= s.bend_max:
        return None
    if fretted + amount < pitch - 0.5:
        return None  # named after a pitch it never reaches
    # The rise that leads to the held level: from the last frame still near the fretted
    # note to the first at 85 % of the bend.
    peak = int(held.argmax())
    reached = rise[: peak + hold] > 0.85 * amount
    end = int(np.argmax(reached)) if reached.any() else peak
    low = np.flatnonzero(rise[: end + 1] <= 0.15 * amount)
    start = int(low[-1]) if len(low) else 0
    if (end - start) / frame_rate < s.bend_rise:
        return None  # a jump to a new fretted note, not a bend
    steps = np.abs(np.diff(curve[start : end + 1]))
    if len(steps) and np.nanmax(steps, initial=0.0) > s.bend_step * amount:
        return None  # climbs fret by fret: a slide
    return Bend(fretted, round(amount * 2) / 2)


def has_vibrato(
    midi: np.ndarray, frame_rate: float, s: TechniqueSettings = TechniqueSettings()
) -> bool:
    """A regular swing of the pitch at 4-9 Hz with a visible depth."""
    curve = _smooth(midi, 3)
    valid = ~np.isnan(curve)
    if np.count_nonzero(valid) < s.vibrato_min * frame_rate:
        return False
    x = np.interp(np.arange(len(curve)), np.flatnonzero(valid), curve[valid])
    # Remove the slow course of the pitch (a bend, a drift) over about a swing's length.
    trend = ndimage.uniform_filter1d(
        x, max(3, int(round(frame_rate / s.vibrato_rates[0]))), mode="nearest"
    )
    x = x - trend
    if 2 * np.sqrt(2) * x.std() < s.vibrato_depth:
        return False
    # Swing periods up to half the note: a short note shows the faster swings only.
    lo = int(frame_rate / s.vibrato_rates[1])
    hi = min(int(np.ceil(frame_rate / s.vibrato_rates[0])), len(x) // 2)
    if hi < lo:
        return False
    x = x - x.mean()
    energy = float(np.dot(x, x))
    corr = [float(np.dot(x[:-lag], x[lag:])) / energy for lag in range(lo, hi + 1)]
    return max(corr) >= s.vibrato_regularity


def glides(tracker: PitchTracker, first: Note, second: Note, others: Sequence[Note] = ()) -> bool:
    """Whether the pitch glides from one note to the next through the frets between.

    While both notes are in the analysis window, the harmonic sum can peak halfway
    between them and stay there: on GuitarSet's solos most whole-step changes showed
    such a plateau, at the very pitch a slide would pass. So the pitch must also move
    while it is between the two notes, in the direction of the change; a one-fret slide
    can't be told apart this way.
    """
    s = tracker.s
    low, high = sorted((first.pitch, second.pitch))
    curve = tracker.track(
        tracker.frame(min(first.end, second.start) - s.slide_window[0]),
        tracker.frame(max(first.end, second.start) + s.slide_window[1]),
        low - 0.7,
        high + 0.7,
        others,
    )
    midi = _smooth(curve.midi, 3)
    start, length = _longest_run((midi > low + 0.3) & (midi < high - 0.3))
    if not s.slide_glide <= length / tracker.frame_rate <= s.slide_glide_max:
        return False
    glide = midi[start : start + length]
    moved = (glide[-1] - glide[0]) * np.sign(second.pitch - first.pitch)
    return bool(moved >= s.slide_span)


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Start and length of the longest run of True."""
    best, best_start, run = 0, 0, 0
    for i, value in enumerate(mask):
        run = run + 1 if value else 0
        if run > best:
            best, best_start = run, i - run + 1
    return best_start, best


def detect_techniques(
    notes: Sequence[Note],
    y: np.ndarray,
    sample_rate: int,
    s: TechniqueSettings = TechniqueSettings(),
    tuning: Sequence[int] | None = None,
    num_frets: int = DEFAULT_NUM_FRETS,
) -> list[Note]:
    """Mark slides, bends and vibrato on positioned notes (input order is kept).

    A slide is played on one string, but the fingering optimizer, not knowing it, often
    put its two notes on two (4 in 10 of the slides found on IDMT): they are moved onto
    one, the second onto the first's string or else the other way round. A bent note
    named after its bent pitch is set back to the fretted note. ``tuning`` (open
    strings, capo included) is otherwise read from the notes' positions.
    """
    tracker = PitchTracker(y, sample_rate, s)
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch))
    result = [replace(n, technique=None, bend=None, vibrato=False) for n in notes]
    curves = []
    for note in notes:
        others = [n for n in notes if n is not note and n.end > note.start and n.start < note.end]
        curves.append(tracker.note_curve(note, others).midi)
    shift = tracker.tune(curves, [n.pitch for n in notes])
    curves = [c - shift for c in curves]
    if tuning is not None:
        open_strings = dict(enumerate(tuning))
    else:
        open_strings = {
            n.string: n.pitch - n.fret for n in notes if n.string is not None and n.fret is not None
        }

    for j, i in _slide_candidates(notes, order, s):
        first, second = notes[j], notes[i]
        around = [
            n
            for n in notes
            if n is not first
            and n is not second
            and n.end > first.end - 0.1
            and n.start < second.start + 0.1
        ]
        if not glides(tracker, first, second, around):
            continue
        moved = _one_string(result[j], result[i], result, open_strings, num_frets)
        if moved is not None:
            result[j], result[i] = moved[0], replace(moved[1], technique="slide")

    for i in order:
        note, curve = result[i], curves[i]
        bend = find_bend(curve, note.pitch, tracker.frame_rate, s)
        if bend is not None and bend.fretted != note.pitch:
            note = _refret(note, bend.fretted)
            bend = bend if note.pitch == bend.fretted else None
        result[i] = replace(
            note,
            bend=bend.amount if bend else None,
            vibrato=has_vibrato(curve, tracker.frame_rate, s),
        )
    return result


def _slide_candidates(
    notes: Sequence[Note], order: Sequence[int], s: TechniqueSettings
) -> list[tuple[int, int]]:
    """(first, second) index pairs: a note and the note that ends right before it, at
    least two semitones away (a one-fret glide can't be measured), neither in a chord."""
    starts = sorted(n.start for n in notes)
    pairs = []
    for i in order:
        second = notes[i]
        if _in_chord(second.start, starts, s.chord_tolerance):
            continue
        best, best_gap = None, np.inf
        for j in order:
            first = notes[j]
            gap = second.start - first.end
            if first.start >= second.start - 0.02 or not -s.slide_overlap <= gap <= s.slide_gap:
                continue
            if not 2 <= abs(second.pitch - first.pitch) <= s.slide_max:
                continue
            if _in_chord(first.start, starts, s.chord_tolerance):
                continue
            if abs(gap) < best_gap:
                best, best_gap = j, abs(gap)
        if best is not None:
            pairs.append((best, i))
    return pairs


def _one_string(
    first: Note,
    second: Note,
    notes: Sequence[Note],
    open_strings: dict[int, int],
    num_frets: int,
) -> tuple[Note, Note] | None:
    """Both notes on one string: the second moved to the first's string, or the first
    to the second's, where that string is free; None if neither can be."""
    if first.string is None or second.string is None:
        return None
    if first.string == second.string:
        return first, second

    def on(note: Note, string: int) -> Note | None:
        fret = note.pitch - open_strings.get(string, -100)
        if not 0 <= fret <= num_frets:
            return None
        busy = any(
            n.string == string and n is not first and n is not second
            and n.start < note.end and n.end > note.start
            for n in notes
        )  # fmt: skip
        return None if busy else replace(note, string=string, fret=fret)

    moved = on(second, first.string)
    if moved is not None:
        return first, moved
    moved = on(first, second.string)
    return (moved, second) if moved is not None else None


def _refret(note: Note, fretted: int) -> Note:
    """The note at its fretted pitch, on the same string (unchanged if it can't be)."""
    if note.fret is None:
        return replace(note, pitch=fretted)
    fret = note.fret - (note.pitch - fretted)
    return replace(note, pitch=fretted, fret=fret) if fret >= 0 else note


def _in_chord(start: float, starts: Sequence[float], tolerance: float) -> bool:
    lo = np.searchsorted(starts, start - tolerance)
    hi = np.searchsorted(starts, start + tolerance, side="right")
    return hi - lo > 1
