"""Audio loading and basic waveform utilities."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    path: str | Path,
    sample_rate: int = 22050,
    mono: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Load and resample an audio file. Returns float32 samples (mono: shape ``(n,)``)."""
    y, _ = librosa.load(str(path), sr=sample_rate, mono=mono)
    if normalize:
        y = normalize_peak(y)
    return y.astype(np.float32)


def normalize_peak(y: np.ndarray, peak: float = 0.95, eps: float = 1e-8) -> np.ndarray:
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    return y if max_abs < eps else y * (peak / max_abs)


def trim_silence(y: np.ndarray, top_db: float = 60.0) -> tuple[np.ndarray, int]:
    """Trim leading/trailing silence. Returns the trimmed signal and the start offset (samples)."""
    trimmed, index = librosa.effects.trim(y, top_db=top_db)
    return trimmed, int(index[0])


def duration_seconds(y: np.ndarray, sample_rate: int) -> float:
    return y.shape[-1] / sample_rate
