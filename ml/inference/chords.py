"""Chords, beats and bars of a song, for a guitar chord sheet.

Chords come from BTC (``ml.models.btc``; Park et al., ISMIR 2019), trained on real
recordings with a vocabulary of 12 roots x 14 qualities (maj, min, 7, maj7, min7, sus2,
sus4, dim, aug, ...). Beats and bar lines come from Beat This! (Foscarin et al., ISMIR
2024). Both are pretrained and downloaded on first use; BTC's checkpoint is pickled, so it
is loaded only when its SHA-256 matches the published file.

The chord model votes every 93 ms; a sheet needs one chord per beat. Its probabilities
are averaged over each beat and a Viterbi path over the beats chooses the chords, paying
for every change, so a passing tone doesn't flip the chord for a beat.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ml.models import resolve_device
from ml.models.btc import BTC

BTC_URL = "https://github.com/jayg996/BTC-ISMIR19/raw/master/test/btc_model_large_voca.pt"
BTC_SHA256 = "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# BTC's class order: 14 qualities per root, then "X" (unknown) and "N" (no chord).
QUALITIES = [
    "min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7",
    "dim7", "hdim7", "sus2", "sus4",
]  # fmt: skip
LABELS = [f"{root}:{quality}" for root in ROOTS for quality in QUALITIES] + ["X", "N"]


@dataclass(frozen=True)
class ChordSegment:
    start: float
    end: float
    label: str  # Harte syntax as in mir_eval: "C:maj", "A:min7", "N" (no chord)


@dataclass
class SongAnalysis:
    chords: list[ChordSegment]
    beats: list[float]
    downbeats: list[float]


def _btc_checkpoint(cache: Path | None = None) -> Path:
    """The published large-vocabulary checkpoint, downloaded once and checked."""
    cache = cache or Path(torch.hub.get_dir()) / "btc"
    path = cache / "btc_model_large_voca.pt"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(BTC_URL, partial)
        partial.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BTC_SHA256:
        raise ValueError(f"{path} is not the published BTC checkpoint (sha256 {digest})")
    return path


class ChordRecognizer:
    """Audio -> chord probabilities per 93 ms frame (BTC, large vocabulary)."""

    sample_rate = 22050
    hop_length = 2048

    def __init__(self, device: str | None = None, checkpoint: Path | None = None):
        path = checkpoint or _btc_checkpoint()
        # A pickle, but only ever the file whose hash was checked above.
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.mean, self.std = float(state["mean"]), float(state["std"])
        self.device = resolve_device(device)
        self.model = BTC().to(self.device).eval()
        self.model.load_state_dict(state["model"])

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    def features(self, y: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalized log-magnitude CQT, ``(frames, 144)``: 6 octaves from C1, 24 bins each."""
        import librosa

        if sample_rate != self.sample_rate:
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=self.sample_rate)
        cqt = librosa.cqt(
            y, sr=self.sample_rate, n_bins=144, bins_per_octave=24, hop_length=self.hop_length
        )
        return ((np.log(np.abs(cqt) + 1e-6) - self.mean) / self.std).T.astype(np.float32)

    @torch.no_grad()
    def probabilities(
        self,
        y: np.ndarray,
        sample_rate: int,
        progress: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        """``(frames, 170)`` chord probabilities. The model sees 108-frame windows; they
        overlap by half and are cross-faded, so no frame sits at a window's blind edge."""
        x = self.features(y, sample_rate)
        n, size = len(x), self.model.timestep
        step = size // 2
        padded = np.pad(x, ((step, size + step), (0, 0)))
        weight = np.hanning(size + 2)[1:-1, None].astype(np.float32)
        total = np.zeros((len(padded), len(LABELS)), dtype=np.float32)
        norm = np.zeros((len(padded), 1), dtype=np.float32)
        starts = range(0, len(padded) - size + 1, step)
        for k, start in enumerate(starts):
            window = torch.from_numpy(padded[start : start + size])[None].to(self.device)
            probs = torch.softmax(self.model(window), dim=-1)[0].cpu().numpy()
            total[start : start + size] += probs * weight
            norm[start : start + size] += weight
            if progress is not None:
                progress((k + 1) / len(starts))
        return (total / np.maximum(norm, 1e-6))[step : step + n]


class BeatTracker:
    """Beats and downbeats (seconds) with Beat This!."""

    def __init__(self, device: str | None = None, checkpoint: str = "final0"):
        from beat_this.inference import Audio2Beats

        self.device = resolve_device(device)
        self.model = Audio2Beats(checkpoint_path=checkpoint, device=str(self.device), dbn=False)

    def __call__(self, y: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        beats, downbeats = self.model(y, sample_rate)
        return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def beat_chords(
    probs: np.ndarray,
    frame_rate: float,
    beats: Sequence[float],
    duration: float,
    change_penalty: float = 1.0,
) -> list[ChordSegment]:
    """One chord per beat from frame probabilities: the mean log-probability of each
    chord over the beat, and a Viterbi path over the beats that pays ``change_penalty``
    (in log-probability) for every change of chord. Equal neighbors are merged."""
    bounds = [0.0, *[b for b in beats if 0.0 < b < duration], duration]
    frames = np.clip((np.asarray(bounds) * frame_rate).round().astype(int), 0, len(probs))
    logp = np.log(np.maximum(probs, 1e-8))
    scores = []
    for a, b in zip(frames[:-1], frames[1:]):
        chunk = logp[a : max(b, a + 1)] if a < len(logp) else logp[-1:]
        scores.append(chunk.mean(axis=0))
    path = _viterbi(np.array(scores), change_penalty)
    segments: list[ChordSegment] = []
    for (start, end), state in zip(zip(bounds[:-1], bounds[1:]), path):
        label = LABELS[state]
        if segments and segments[-1].label == label:
            segments[-1] = ChordSegment(segments[-1].start, end, label)
        else:
            segments.append(ChordSegment(start, end, label))
    return segments


BASS_LOW, BASS_HIGH = 28, 57  # E1-A3: bass guitar, and a guitar's lowest notes
_BASS_RATE, _BASS_HOP = 22050, 512
_BASS_HARMONICS = ((0, 1.0), (12, 0.8), (19, 0.6), (24, 0.5))  # semitones above, weight


def beat_bass(
    y: np.ndarray, sample_rate: int, beats: Sequence[float], duration: float
) -> list[tuple[int | None, float]]:
    """The bass note's pitch class in each beat (as ``beat_chords`` cuts them) and the
    share of the beat's bass salience it holds; ``(None, 0)`` for a silent beat.

    Every frame's bass is the lowest pitch (E1-A3) whose harmonic sum of CQT energy is a
    peak across pitch reaching half the frame's strongest: the lowest note, not the
    loudest partial (a bass's octave is often stronger) nor the next bin down, into which
    a low note's energy spreads. The CQT follows the recording's tuning.
    """
    import librosa

    if sample_rate != _BASS_RATE:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=_BASS_RATE)
    span = BASS_HIGH - BASS_LOW + 1
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=_BASS_RATE,
            hop_length=_BASS_HOP,
            fmin=librosa.midi_to_hz(BASS_LOW),
            n_bins=span + _BASS_HARMONICS[-1][0],
            bins_per_octave=12,
            tuning=None,  # estimated from the recording
        )
    )
    salience = sum(weight * cqt[shift : shift + span] for shift, weight in _BASS_HARMONICS)
    loudness = salience.max(axis=0)
    floor = 0.05 * float(np.percentile(loudness, 95)) if loudness.size else 0.0
    padded = np.pad(salience, ((1, 1), (0, 0)))
    peak = (salience >= padded[:-2]) & (salience >= padded[2:])
    strong = peak & (salience >= 0.5 * loudness[None, :])
    lowest = np.where(strong.any(axis=0), strong.argmax(axis=0), -1)
    frame_rate = _BASS_RATE / _BASS_HOP
    bounds = [0.0, *[b for b in beats if 0.0 < b < duration], duration]
    out: list[tuple[int | None, float]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        lo, hi = int(a * frame_rate), max(int(a * frame_rate) + 1, int(b * frame_rate))
        weights = np.zeros(12)
        for frame in range(lo, min(hi, len(lowest))):
            if lowest[frame] >= 0 and loudness[frame] > floor:
                weights[(BASS_LOW + lowest[frame]) % 12] += loudness[frame]
        total = weights.sum()
        out.append(
            (int(weights.argmax()), float(weights.max() / total)) if total > 0 else (None, 0.0)
        )
    return out


# Bass notes under a chord that a chord sheet writes as a slash chord (semitones above
# the root): a third, the fifth or a seventh (D/F#, C/G, Am/G, C/B). Passing tones (the
# second, fourth, sixth) added wrong slash chords on AAM, so the chord stays as it is.
SLASH_INTERVALS = {3, 4, 7, 10, 11}


def add_bass(
    segments: Sequence[ChordSegment],
    bass: Sequence[tuple[int | None, float]],
    beats: Sequence[float],
    duration: float,
    min_beats: int = 2,
    min_share: float = 0.5,
) -> list[ChordSegment]:
    """Slash chords: within each chord, a run of at least ``min_beats`` beats over one
    bass note other than the root (in ``SLASH_INTERVALS``, held by at least ``min_share``
    of each beat's bass) becomes ``root:quality/interval`` (Harte). A bass changing every
    beat (an alternating root-fifth bass) leaves the chord as it is."""
    bounds = [0.0, *[b for b in beats if 0.0 < b < duration], duration]
    out: list[ChordSegment] = []
    for segment in segments:
        root = _root(segment.label)
        rows = [
            i for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))
            if a >= segment.start - 1e-6 and b <= segment.end + 1e-6
        ]  # fmt: skip
        if root is None or not rows:
            out.append(segment)
            continue
        notes = [bass[i][0] if bass[i][1] >= min_share else None for i in rows]
        runs: list[tuple[int, int, int | None]] = []  # first row, last row, bass interval
        for k, pc in zip(rows, notes):
            interval = None if pc is None else (pc - root) % 12
            if runs and runs[-1][2] == interval:
                runs[-1] = (runs[-1][0], k, interval)
            else:
                runs.append((k, k, interval))
        # Short runs and the root (or an unclear bass) don't make a slash chord.
        labels = [
            f"{segment.label}/{_INTERVAL_NAMES[i]}"
            if i in SLASH_INTERVALS and last - first + 1 >= min_beats
            else segment.label
            for first, last, i in runs
        ]
        for (first, last, _), label in zip(runs, labels):
            start = max(segment.start, bounds[first])
            end = min(segment.end, bounds[last + 1])
            if out and out[-1].label == label and abs(out[-1].end - start) < 1e-6:
                out[-1] = ChordSegment(out[-1].start, end, label)
            else:
                out.append(ChordSegment(start, end, label))
    return out


_INTERVAL_NAMES = {
    1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
}  # fmt: skip


def _root(label: str) -> int | None:
    if label in ("N", "X"):
        return None
    return ROOTS.index(label.split(":")[0])


def beat_scores(
    probs: np.ndarray, frame_rate: float, beats: Sequence[float], duration: float, top: int = 6
) -> list[dict]:
    """Each beat's ``top`` labels and their mean log-probability, for fusing with a
    transcription's notes (``music_core.analysis.analyze``'s ``chord_scores``); the beats
    are cut as in ``beat_chords``. Small enough to store with a project."""
    bounds = [0.0, *[b for b in beats if 0.0 < b < duration], duration]
    frames = np.clip((np.asarray(bounds) * frame_rate).round().astype(int), 0, len(probs))
    logp = np.log(np.maximum(probs, 1e-8))
    out = []
    for start, a, b in zip(bounds[:-1], frames[:-1], frames[1:]):
        mean = (logp[a : max(b, a + 1)] if a < len(logp) else logp[-1:]).mean(axis=0)
        best = np.argsort(mean)[::-1][:top]
        out.append(
            {
                "start": round(float(start), 3),
                "top": [[LABELS[i], round(float(mean[i]), 3)] for i in best if LABELS[i] != "X"],
            }
        )
    return out


def frame_chords(probs: np.ndarray, frame_rate: float) -> list[ChordSegment]:
    """The model's own per-frame choice (no beats), merged: for comparison."""
    best = probs.argmax(axis=1)
    segments: list[ChordSegment] = []
    for i, state in enumerate(best):
        start, end = i / frame_rate, (i + 1) / frame_rate
        if segments and segments[-1].label == LABELS[state]:
            segments[-1] = ChordSegment(segments[-1].start, end, LABELS[state])
        else:
            segments.append(ChordSegment(start, end, LABELS[state]))
    return segments


def _viterbi(scores: np.ndarray, change_penalty: float) -> list[int]:
    """Best state sequence for per-step ``scores`` (log-probabilities) when changing
    state costs ``change_penalty``."""
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
