"""Voice mode: a sung, hummed or whistled melody -> notes -> guitar tab.

A voice sounds one note at a time, so no trained model is needed: pYIN (Mauch & Dixon,
2014) follows the fundamental and the pitch curve is cut into notes. Rounding every
frame to the nearest semitone would split notes at each vibrato swing and wherever a
held note drifts across a semitone boundary, so notes are cut only where the mean pitch
before and after a moment differs by most of a semitone (``change_threshold``): a real
change of note does that, vibrato (averaged over a period) and slow drift don't. Each
note gets the median pitch of its frames, counted from the singer's own reference
pitch, which is rarely A = 440 Hz and is measured first. The slide between two legato
notes belongs to neither: a note starts where its pitch settles and ends where it
leaves. Notes also end where the voice stops and where it re-attacks on the same pitch:
a consonant between syllables changes the spectrum and dips the voicing and loudness,
three weak cues that are added up.

python -m ml.inference.voice melody.wav --tab melody.txt
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ml.inference.predict import ProgressCallback, TranscriptionResult, _estimate_tempo
from ml.preprocessing.audio import load_audio
from music_core.analysis import refine_tempo
from music_core.notes import Note
from music_core.tab import DEFAULT_NUM_FRETS, STANDARD_TUNING, assign_tab

# Stored as the project's model_version; bump it when the method changes notes.
VOICE_VERSION = "voice-pyin@1"

Piece = tuple[int, int, int]  # start frame, end frame, MIDI pitch


@dataclass(frozen=True)
class VoiceSettings:
    sample_rate: int = 16000
    hop_length: int = 160  # 10 ms frames
    frame_length: int = 1024
    fmin: float = 65.4  # C2
    fmax: float = 2093.0  # C7: whistling goes this high
    resolution: float = 0.5  # pYIN's pitch grid in semitones; YIN refines the pitch
    chunk_seconds: float = 20.0  # pYIN runs in chunks, for progress reports
    silence_db: float = 45.0  # frames this far below the loud parts count as silence
    max_gap: float = 0.03  # shorter unvoiced gaps inside a note are bridged
    change_window: float = 0.2  # seconds of pitch averaged on each side of a cut
    change_threshold: float = 0.7  # semitones between those means that make a new note
    stable_tolerance: float = 0.5  # semitones from the note where its pitch counts as settled
    # A re-attack on one pitch: spectral change (onset strength relative to the
    # recording's strong onsets) plus dips in voicing probability and loudness that, in
    # these units, add up to 1.
    reattack_onset: float = 4.0
    reattack_prob: float = 1.5
    reattack_db: float = 20.0
    min_note: float = 0.1  # seconds; shorter pieces (glides, scoops) join a neighbor

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length


@dataclass
class PitchTrack:
    """Per frame: pitch in (fractional) MIDI or NaN, voicing probability, loudness and
    onset strength (spectral change)."""

    midi: np.ndarray
    voiced_prob: np.ndarray
    loudness_db: np.ndarray
    onset: np.ndarray


def track_pitch(
    y: np.ndarray,
    s: VoiceSettings = VoiceSettings(),
    progress: Callable[[float], None] | None = None,
) -> PitchTrack:
    """pYIN decides voicing and the rough pitch (octave errors included); YIN's finer,
    frame-by-frame estimate replaces it where both agree."""
    import librosa

    hop, n = s.hop_length, len(y)
    n_frames = 1 + n // hop  # librosa's centered frame count
    coarse = np.full(n_frames, np.nan)
    flag = np.zeros(n_frames, dtype=bool)
    prob = np.zeros(n_frames)
    chunk = max(hop, int(s.chunk_seconds * s.sample_rate) // hop * hop)
    context = s.frame_length // hop * hop * 2  # enough for pYIN's window and smoothing
    for start in range(0, n, chunk):
        lo, hi = max(0, start - context), min(n, start + chunk + context)
        f0, voiced, p = librosa.pyin(
            y[lo:hi],
            fmin=s.fmin,
            fmax=s.fmax,
            sr=s.sample_rate,
            frame_length=s.frame_length,
            hop_length=hop,
            resolution=s.resolution,
        )
        first = (start - lo) // hop
        count = min(chunk // hop, n_frames - start // hop, len(f0) - first)
        dest = slice(start // hop, start // hop + count)
        coarse[dest], flag[dest], prob[dest] = (
            f0[first : first + count],
            voiced[first : first + count],
            p[first : first + count],
        )
        if progress is not None:
            progress(min(1.0, (start + chunk) / n))

    fine = librosa.yin(
        y, fmin=s.fmin, fmax=s.fmax, sr=s.sample_rate, frame_length=s.frame_length, hop_length=hop
    )[:n_frames]
    midi = librosa.hz_to_midi(np.where(flag, coarse, np.nan))
    fine_midi = librosa.hz_to_midi(fine)
    agree = np.abs(fine_midi - midi) <= s.resolution  # NaN compares False
    midi = np.where(agree, fine_midi, midi)

    rms = librosa.feature.rms(y=y, frame_length=s.frame_length, hop_length=hop)[0][:n_frames]
    loudness = 20 * np.log10(np.maximum(rms, 1e-6))
    onset = librosa.onset.onset_strength(y=y, sr=s.sample_rate, hop_length=hop)[:n_frames]
    onset = np.pad(onset, (0, n_frames - len(onset)))
    return PitchTrack(midi=midi, voiced_prob=prob, loudness_db=loudness, onset=onset)


def estimate_tuning(midi: np.ndarray, weights: np.ndarray | None = None) -> float:
    """The singer's offset from equal temperament in semitones, in [-0.5, 0.5)."""
    ok = np.isfinite(midi)
    if not ok.any():
        return 0.0
    angle = 2 * np.pi * midi[ok]
    w = np.ones(ok.sum()) if weights is None else weights[ok]
    offset = float(np.angle(np.sum(w * np.exp(1j * angle))) / (2 * np.pi))
    return offset if offset < 0.5 else offset - 1.0


def segment_notes(track: PitchTrack, s: VoiceSettings = VoiceSettings()) -> list[Note]:
    """Pitch track -> notes with whole-semitone pitches, as sung (no octave shift)."""
    midi, loudness = track.midi, track.loudness_db
    voiced = np.isfinite(midi)
    if not voiced.any():
        return []
    loud = float(np.percentile(loudness[voiced], 95))
    voiced &= loudness > loud - s.silence_db
    midi = np.where(voiced, midi, np.nan)
    midi = midi - estimate_tuning(midi, track.voiced_prob)
    onset = track.onset / max(float(np.percentile(track.onset[voiced], 95)), 1e-6)

    shortest = max(1, int(round(s.min_note * s.frame_rate)))
    notes: list[Note] = []
    for lo, hi in _runs(voiced, bridge=int(round(s.max_gap * s.frame_rate))):
        for a, b, pitch in _absorb_short(_pieces(midi[lo:hi], shortest, s), shortest):
            a, b = _settled(midi[lo:hi], a, b, pitch, s.stable_tolerance)
            if b - a < shortest:
                continue
            span = slice(lo + a, lo + b)
            syllables = _split_reattacks(
                onset[span], track.voiced_prob[span], loudness[span], shortest, s
            )
            for c, d in syllables:
                if d - c >= shortest:
                    notes.append(_note(pitch, lo + a + c, lo + a + d, track, loud, s))
    return notes


def fit_to_range(notes: Sequence[Note], low: int, high: int) -> tuple[list[Note], int]:
    """Shift the melody by whole octaves so the most notes lie in ``[low, high]``; on a
    tie the smallest shift wins, so a melody that fits is left alone."""
    if not notes:
        return list(notes), 0
    pitches = np.array([n.pitch for n in notes])
    shifts = sorted(range(-36, 37, 12), key=abs)
    best = max(shifts, key=lambda k: int(np.sum((pitches + k >= low) & (pitches + k <= high))))
    return [replace(n, pitch=n.pitch + best) for n in notes] if best else list(notes), best


def voice_notes(
    y: np.ndarray,
    s: VoiceSettings = VoiceSettings(),
    progress: Callable[[float], None] | None = None,
) -> list[Note]:
    """Mono samples at ``s.sample_rate`` -> the sung notes."""
    return segment_notes(track_pitch(y, s, progress), s)


def transcribe_voice(
    audio_path: str | Path,
    tuning: Sequence[int] | None = None,
    num_frets: int = DEFAULT_NUM_FRETS,
    separator: Callable[[np.ndarray, int], np.ndarray] | None = None,
    progress: ProgressCallback | None = None,
    estimate_tempo: bool = True,
    s: VoiceSettings = VoiceSettings(),
) -> TranscriptionResult:
    """A recording of a voice -> notes placed on the guitar in ``tuning``.

    The melody is moved by whole octaves when that fits more of it on the fretboard
    (a whistle is usually an octave or two above the guitar); ``transpose`` says by how
    much. ``separator`` isolates the voice from a band mix first.
    """
    report = progress or (lambda fraction, stage: None)
    tuning = list(tuning or STANDARD_TUNING)
    track_start = 0.5 if separator is not None else 0.05
    report(0.0, "loading")
    y = load_audio(audio_path, s.sample_rate, mono=True, normalize=True)
    if separator is not None:
        report(0.05, "separating")
        y = separator(y, s.sample_rate)
    report(track_start, "transcribing")
    notes = voice_notes(
        y, s, lambda f: report(track_start + (0.95 - track_start) * f, "transcribing")
    )
    report(0.95, "finishing")
    notes, transpose = fit_to_range(notes, min(tuning), max(tuning) + num_frets)
    notes = assign_tab(notes, tuning, num_frets)
    tempo = _tempo(notes, y, s) if estimate_tempo else None
    return TranscriptionResult(
        notes=notes,
        duration=len(y) / s.sample_rate,
        tempo=tempo,
        tuning=tuning,
        transpose=transpose,
    )


def _tempo(notes: Sequence[Note], y: np.ndarray, s: VoiceSettings) -> float | None:
    """BPM from the note onsets (weighted by loudness), refined like the guitar's."""
    if len(notes) < 4:
        return None
    envelope = np.zeros(1 + len(y) // s.hop_length)
    for note in notes:
        frame = int(round(note.start * s.frame_rate))
        if frame < len(envelope):
            envelope[frame] += note.velocity / 127
    envelope = np.convolve(envelope, np.hanning(5), mode="same")
    tempo = _estimate_tempo({"onset": envelope[:, None]}, y, s.sample_rate, s.hop_length)
    return round(refine_tempo(notes, tempo), 2) if tempo is not None else None


def _runs(mask: np.ndarray, bridge: int = 0) -> list[tuple[int, int]]:
    """``[start, end)`` of the True runs, joining runs separated by <= ``bridge`` frames."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    runs: list[tuple[int, int]] = []
    for lo, hi in zip(edges[::2], edges[1::2]):
        if runs and lo - runs[-1][1] <= bridge:
            runs[-1] = (runs[-1][0], int(hi))
        else:
            runs.append((int(lo), int(hi)))
    return runs


def _pieces(midi: np.ndarray, shortest: int, s: VoiceSettings) -> list[Piece]:
    """Cut a voiced run where the mean pitch changes; pieces get their median semitone."""
    width = max(1, int(round(s.change_window * s.frame_rate)))
    ok = np.isfinite(midi)
    n = len(midi)
    sums = np.concatenate([[0.0], np.cumsum(np.where(ok, midi, 0.0))])
    counts = np.concatenate([[0], np.cumsum(ok)])
    t = np.arange(n + 1)
    lo, hi = np.maximum(t - width, 0), np.minimum(t + width, n)
    before = (sums[t] - sums[lo]) / np.maximum(counts[t] - counts[lo], 1)
    after = (sums[hi] - sums[t]) / np.maximum(counts[hi] - counts[t], 1)
    change = np.abs(after - before)
    change[: min(shortest, n + 1)] = 0  # no cut right at the run's edges
    change[max(0, n + 1 - shortest) :] = 0
    peak = np.zeros(n + 1, dtype=bool)
    peak[1:-1] = (change[1:-1] >= change[:-2]) & (change[1:-1] > change[2:])
    cuts = _strongest(np.where(peak, change, 0.0), s.change_threshold, shortest)
    bounds = [0, *cuts, n]
    pieces = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        frames = midi[a:b][ok[a:b]]
        if frames.size:
            pieces.append((a, b, int(np.round(np.median(frames)))))
    return pieces


def _strongest(score: np.ndarray, threshold: float, spacing: int) -> list[int]:
    """Frames where ``score`` reaches ``threshold``, strongest first, at least ``spacing``
    frames apart; returned in time order."""
    chosen: list[int] = []
    for i in np.argsort(-score):
        if score[i] < threshold:
            break
        if all(abs(int(i) - c) >= spacing for c in chosen):
            chosen.append(int(i))
    return sorted(chosen)


def _absorb_short(pieces: list[Piece], shortest: int) -> list[Piece]:
    """Merge pieces shorter than ``shortest`` frames (a glide passing a semitone, a scoop
    into a note) into the neighbor closest in pitch, shortest first."""
    pieces = _merge_equal(pieces)
    while len(pieces) > 1:
        i = min(range(len(pieces)), key=lambda k: pieces[k][1] - pieces[k][0])
        a, b, pitch = pieces[i]
        if b - a >= shortest:
            break
        left = pieces[i - 1] if i > 0 else None
        right = pieces[i + 1] if i + 1 < len(pieces) else None
        if right is None or (left is not None and abs(left[2] - pitch) <= abs(right[2] - pitch)):
            pieces[i - 1 : i + 1] = [(left[0], b, left[2])]  # type: ignore[index]
        else:
            pieces[i : i + 2] = [(a, right[1], right[2])]
        pieces = _merge_equal(pieces)
    return pieces


def _merge_equal(pieces: list[Piece]) -> list[Piece]:
    """Neighbors that share a pitch become one piece."""
    merged: list[Piece] = []
    for piece in pieces:
        if merged and merged[-1][2] == piece[2]:
            merged[-1] = (merged[-1][0], piece[1], piece[2])
        else:
            merged.append(piece)
    return merged


def _settled(midi: np.ndarray, a: int, b: int, pitch: int, tolerance: float) -> tuple[int, int]:
    """Frames ``[a, b)`` trimmed to where the pitch is within ``tolerance`` of the note."""
    near = np.flatnonzero(np.abs(midi[a:b] - pitch) <= tolerance)  # NaN compares False
    return (a + int(near[0]), a + int(near[-1]) + 1) if near.size else (a, a)


def _split_reattacks(
    onset: np.ndarray,
    voiced_prob: np.ndarray,
    loudness: np.ndarray,
    shortest: int,
    s: VoiceSettings,
) -> list[tuple[int, int]]:
    """Split a held pitch at the consonants between syllables sung on it: a spectral
    change plus dips in voicing and loudness, each alone too common inside a note."""
    width = max(1, int(round(0.15 * s.frame_rate)))
    score = (
        onset / s.reattack_onset
        + _dip(_smooth(voiced_prob), width) / s.reattack_prob
        + _dip(_smooth(loudness), width) / s.reattack_db
    )
    score[:shortest] = 0  # every syllable is at least a note long
    score[max(0, len(score) - shortest + 1) :] = 0
    cuts = [0, *_strongest(score, 1.0, shortest), len(score)]
    return list(zip(cuts[:-1], cuts[1:]))


def _smooth(x: np.ndarray) -> np.ndarray:
    return np.convolve(np.pad(x, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def _dip(x: np.ndarray, width: int) -> np.ndarray:
    """How far each frame lies below the lower of the maxima just before and after it."""
    from scipy.ndimage import maximum_filter1d

    size = width + 1
    before = maximum_filter1d(x, size, origin=width - size // 2, mode="nearest")  # x[i-width : i+1]
    after = maximum_filter1d(x, size, origin=-(size // 2), mode="nearest")  # x[i : i+width+1]
    return np.maximum(np.minimum(before, after) - x, 0.0)


def _note(
    pitch: int, start: int, end: int, track: PitchTrack, loud: float, s: VoiceSettings
) -> Note:
    peak = float(track.loudness_db[start:end].max())
    velocity = int(round(np.interp(peak, [loud - 40, loud], [30, 110])))
    return Note(
        pitch=pitch,
        start=round(start / s.frame_rate, 4),
        end=round(end / s.frame_rate, 4),
        velocity=velocity,
        confidence=round(float(np.mean(track.voiced_prob[start:end])), 3),
    )


def main(argv: list[str] | None = None) -> None:
    from music_core.midi import write_midi
    from music_core.tab import tab_to_ascii

    parser = argparse.ArgumentParser(description="Transcribe a sung or hummed melody.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--midi", type=Path, help="write a MIDI file")
    parser.add_argument("--tab", type=Path, help="write ASCII tablature")
    args = parser.parse_args(argv)

    result = transcribe_voice(args.audio)
    tempo = f"{result.tempo:.1f} BPM" if result.tempo else "unknown tempo"
    shift = f", moved {result.transpose:+d} semitones" if result.transpose else ""
    print(f"{len(result.notes)} notes, {result.duration:.1f} s, {tempo}{shift}")
    if args.midi:
        write_midi(result.notes, args.midi, tempo=result.tempo or 120.0)
        print(f"MIDI written to {args.midi}")
    if args.tab:
        args.tab.write_text(tab_to_ascii(result.notes, result.tuning), encoding="utf-8")
        print(f"Tab written to {args.tab}")


if __name__ == "__main__":
    main()
