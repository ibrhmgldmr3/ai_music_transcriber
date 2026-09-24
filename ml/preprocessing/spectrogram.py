"""Time-frequency features (CQT / mel) used as model input."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np


def compute_cqt(
    y: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_bins: int,
    bins_per_octave: int,
    fmin_midi: int,
) -> np.ndarray:
    """Magnitude constant-Q transform, shape ``(n_bins, n_frames)``."""
    cqt = librosa.cqt(
        y,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=librosa.midi_to_hz(fmin_midi),
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    return np.abs(cqt)


def compute_mel(
    y: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 30.0,
) -> np.ndarray:
    """Magnitude mel spectrogram, shape ``(n_mels, n_frames)``."""
    return librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        power=1.0,
    )


def log_compress(spec: np.ndarray, offset: float = 1e-6) -> np.ndarray:
    return np.log(spec + offset)


def compute_features(y: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Log-magnitude features as configured in ``cfg['features']``: float32 ``(bins, frames)``."""
    audio, features = cfg["audio"], cfg["features"]
    if features["type"] == "cqt":
        spec = compute_cqt(
            y,
            sample_rate=audio["sample_rate"],
            hop_length=audio["hop_length"],
            n_bins=features["n_bins"],
            bins_per_octave=features["bins_per_octave"],
            fmin_midi=features["fmin_midi"],
        )
    elif features["type"] == "mel":
        spec = compute_mel(
            y,
            sample_rate=audio["sample_rate"],
            hop_length=audio["hop_length"],
            n_fft=features["n_fft"],
            n_mels=features["n_mels"],
            fmin=features.get("fmin", 30.0),
        )
    else:
        raise ValueError(f"Unknown feature type {features['type']!r} (expected 'cqt' or 'mel')")
    return log_compress(spec, features.get("log_offset", 1e-6)).astype(np.float32)
