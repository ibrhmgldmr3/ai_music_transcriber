"""Data augmentation.

Two levels:

* **Offline variants** (``apply_variant``): pitch shift, tempo change, noise, reverb, EQ and
  gain rendered once by ``scripts/prepare_dataset.py``. Note labels are transformed together
  with the audio (pitch shift moves pitches/frets, time stretch rescales onsets/offsets).
* **On-the-fly** (``FeatureAugmenter``): cheap gain offset and SpecAugment-style masking on
  log-spectrogram crops during training.

Both are used for the training split only - never for validation or test data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from music_core.notes import Note
from music_core.tab import STANDARD_TUNING

# --------------------------------------------------------------------------- waveform


def random_gain(y: np.ndarray, rng: np.random.Generator, max_db: float = 6.0) -> np.ndarray:
    gain_db = rng.uniform(-max_db, max_db)
    return y * (10.0 ** (gain_db / 20.0))


def add_noise(
    y: np.ndarray,
    rng: np.random.Generator,
    snr_db_range: Sequence[float] = (20.0, 40.0),
) -> np.ndarray:
    """Add white noise at a random signal-to-noise ratio."""
    signal_power = float(np.mean(y**2)) + 1e-12
    snr_db = rng.uniform(*snr_db_range)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return y + rng.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(y.dtype)


def add_reverb(
    y: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    decay_range: Sequence[float] = (0.2, 0.8),
    wet_range: Sequence[float] = (0.1, 0.35),
) -> np.ndarray:
    """Convolve with a synthetic exponentially decaying noise impulse response."""
    from scipy.signal import fftconvolve

    rt60 = rng.uniform(*decay_range)
    length = int(rt60 * sample_rate)
    t = np.arange(length) / sample_rate
    impulse = rng.normal(0.0, 1.0, length) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    impulse /= np.sqrt(np.sum(impulse**2)) + 1e-12
    wet = fftconvolve(y, impulse)[: len(y)]
    mix = rng.uniform(*wet_range)
    return ((1.0 - mix) * y + mix * wet).astype(y.dtype)


def random_eq(
    y: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    max_db: float = 6.0,
) -> np.ndarray:
    """Random low/high shelf-like tilt: boost or cut the lows and highs independently."""
    from scipy.signal import butter, sosfilt

    low_cut = rng.uniform(150.0, 400.0)
    high_cut = rng.uniform(2000.0, 5000.0)
    low = sosfilt(butter(2, low_cut, "lowpass", fs=sample_rate, output="sos"), y)
    high = sosfilt(butter(2, high_cut, "highpass", fs=sample_rate, output="sos"), y)
    low_gain = 10.0 ** (rng.uniform(-max_db, max_db) / 20.0) - 1.0
    high_gain = 10.0 ** (rng.uniform(-max_db, max_db) / 20.0) - 1.0
    return (y + low_gain * low + high_gain * high).astype(y.dtype)


def distortion(
    y: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    hard: bool = False,
    drive_db: Sequence[float] = (12.0, 30.0),
    tone_hz: Sequence[float] = (2500.0, 6000.0),
) -> np.ndarray:
    """Amp-style drive: boost, waveshape (tanh overdrive or hard-clip distortion), then a
    cabinet-like low-pass. Output keeps the input's peak level."""
    from scipy.signal import butter, sosfilt

    driven = y * 10.0 ** (rng.uniform(*drive_db) / 20.0)
    shaped = np.clip(driven, -1.0, 1.0) if hard else np.tanh(driven)
    cabinet = butter(2, rng.uniform(*tone_hz), "lowpass", fs=sample_rate, output="sos")
    out = sosfilt(cabinet, shaped)
    peak_in, peak_out = float(np.max(np.abs(y))), float(np.max(np.abs(out)))
    if peak_out > 0:
        out *= peak_in / peak_out
    return out.astype(y.dtype)


def phone_quality(y: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Phone / voice-memo sound: 300-3400 Hz band, 8 kHz sampling, light compression."""
    import librosa
    from scipy.signal import butter, sosfilt

    # Compress first: the band limit must come last, since waveshaping adds harmonics.
    peak = float(np.max(np.abs(y))) or 1.0
    squashed = np.tanh(rng.uniform(1.5, 3.0) * y / peak) * peak  # automatic gain control
    band = butter(4, [300.0, 3400.0], "bandpass", fs=sample_rate, output="sos")
    narrow = librosa.resample(
        librosa.resample(sosfilt(band, squashed), orig_sr=sample_rate, target_sr=8000),
        orig_sr=8000,
        target_sr=sample_rate,
    )[: len(y)]
    return np.pad(narrow, (0, len(y) - len(narrow))).astype(y.dtype)


def echo(
    y: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    delay_s: Sequence[float] = (0.15, 0.45),
    feedback: Sequence[float] = (0.2, 0.5),
    mix: Sequence[float] = (0.2, 0.4),
) -> np.ndarray:
    """Feedback delay (echo pedal)."""
    from scipy.signal import lfilter

    d = int(rng.uniform(*delay_s) * sample_rate)
    fb = rng.uniform(*feedback)
    # wet[n] = y[n - d] + fb * wet[n - d]
    b = np.zeros(d + 1)
    b[d] = 1.0
    a = np.zeros(d + 1)
    a[0], a[d] = 1.0, -fb
    wet = lfilter(b, a, y)
    return (y + rng.uniform(*mix) * wet).astype(y.dtype)


def pitch_shift(y: np.ndarray, sample_rate: int, n_steps: float) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """rate > 1 speeds up (shorter audio), rate < 1 slows down."""
    import librosa

    return librosa.effects.time_stretch(y, rate=rate)


# ------------------------------------------------------------------------------ labels


def shift_notes(
    notes: Sequence[Note],
    n_steps: int,
    tuning: Sequence[int] = STANDARD_TUNING,
    num_frets: int = 20,
) -> list[Note]:
    """Transpose notes; string positions move by the same number of frets when possible."""
    shifted = []
    for note in notes:
        pitch = note.pitch + n_steps
        string, fret = note.string, note.fret
        if string is not None and fret is not None:
            fret += n_steps
            if not (0 <= fret <= num_frets and 0 <= string < len(tuning)):
                string, fret = None, None
        shifted.append(replace(note, pitch=pitch, string=string, fret=fret))
    return shifted


def stretch_notes(notes: Sequence[Note], rate: float) -> list[Note]:
    return [replace(n, start=n.start / rate, end=n.end / rate) for n in notes]


def apply_variant(
    y: np.ndarray,
    notes: Sequence[Note],
    variant: dict[str, Any],
    sample_rate: int,
    rng: np.random.Generator,
    tuning: Sequence[int] = STANDARD_TUNING,
    num_frets: int = 20,
) -> tuple[np.ndarray, list[Note]]:
    """Apply one offline augmentation variant (see ``offline_augmentation`` in base.yaml).

    Effects run in signal-chain order: pitch/tempo, guitar effects (distortion, echo),
    room (reverb), tone (eq), recording device (phone), noise and gain. ``reverb`` may
    be ``true`` or a dict of ``add_reverb`` keyword arguments.
    """
    notes = list(notes)
    if variant.get("pitch_shift"):
        steps = int(variant["pitch_shift"])
        y = pitch_shift(y, sample_rate, steps)
        notes = shift_notes(notes, steps, tuning, num_frets)
    if variant.get("time_stretch"):
        rate = float(variant["time_stretch"])
        y = time_stretch(y, rate)
        notes = stretch_notes(notes, rate)
    if variant.get("distortion"):
        y = distortion(y, sample_rate, rng, hard=variant["distortion"] == "hard")
    if variant.get("echo"):
        y = echo(y, sample_rate, rng)
    if variant.get("reverb"):
        options = variant["reverb"] if isinstance(variant["reverb"], dict) else {}
        y = add_reverb(y, sample_rate, rng, **options)
    if variant.get("eq"):
        y = random_eq(y, sample_rate, rng)
    if variant.get("phone"):
        y = phone_quality(y, sample_rate, rng)
    if variant.get("noise_snr_db"):
        y = add_noise(y, rng, variant["noise_snr_db"])
    if variant.get("gain_db"):
        y = random_gain(y, rng, float(variant["gain_db"]))
    return y.astype(np.float32), notes


# ----------------------------------------------------------------------------- features


def spec_augment(
    spec: np.ndarray,
    rng: np.random.Generator,
    freq_masks: int = 1,
    freq_mask_width: int = 8,
    time_masks: int = 2,
    time_mask_width: int = 16,
) -> np.ndarray:
    """SpecAugment-style frequency / time masking on a ``(bins, frames)`` array (in place)."""
    n_bins, n_frames = spec.shape
    fill = float(spec.mean())
    for _ in range(freq_masks):
        width = int(rng.integers(0, freq_mask_width + 1))
        if 0 < width < n_bins:
            start = int(rng.integers(0, n_bins - width + 1))
            spec[start : start + width, :] = fill
    for _ in range(time_masks):
        width = int(rng.integers(0, time_mask_width + 1))
        if 0 < width < n_frames:
            start = int(rng.integers(0, n_frames - width + 1))
            spec[:, start : start + width] = fill
    return spec


class FeatureAugmenter:
    """Random gain (as a log-domain offset), optional noise and spectrogram masking."""

    def __init__(
        self,
        gain_db: float = 6.0,
        noise_std: float = 0.0,
        freq_masks: int = 1,
        freq_mask_width: int = 8,
        time_masks: int = 2,
        time_mask_width: int = 16,
    ):
        self.gain_db = gain_db
        self.noise_std = noise_std
        self.freq_masks = freq_masks
        self.freq_mask_width = freq_mask_width
        self.time_masks = time_masks
        self.time_mask_width = time_mask_width

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> FeatureAugmenter | None:
        if not cfg or not cfg.get("enabled", False):
            return None
        return cls(**{k: v for k, v in cfg.items() if k != "enabled"})

    def __call__(self, features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = features.astype(np.float32, copy=True)
        if self.gain_db:
            # log(g * S) = log(S) + log(g)
            x += rng.uniform(-self.gain_db, self.gain_db) / 20.0 * np.log(10.0)
        if self.noise_std:
            x += rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
        return spec_augment(
            x,
            rng,
            freq_masks=self.freq_masks,
            freq_mask_width=self.freq_mask_width,
            time_masks=self.time_masks,
            time_mask_width=self.time_mask_width,
        )
