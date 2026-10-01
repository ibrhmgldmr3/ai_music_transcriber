"""Strums of an accompaniment in audio, for the strumming pattern (music_core.rhythm).

In a song, the accompaniment (the guitar, piano and other stems Demucs separates, without
vocals, bass and drums) is strummed or comped; its onsets, with their strength, are the
strums. A transcription has better ones: its chords (``music_core.rhythm.strums_from_notes``).
"""

from __future__ import annotations

import numpy as np

_HOP = 256


def audio_strums(y: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    """(time, strength) of the onsets in ``y``: spectral flux peaks (librosa), the
    strength relative to the strongest."""
    import librosa

    envelope = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=_HOP)
    if not envelope.size or envelope.max() <= 0:
        return []
    peaks = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=sample_rate, hop_length=_HOP, units="frames"
    )
    times = librosa.frames_to_time(peaks, sr=sample_rate, hop_length=_HOP)
    strength = envelope[peaks] / envelope.max()
    return [(round(float(t), 3), round(float(s), 3)) for t, s in zip(times, strength)]
