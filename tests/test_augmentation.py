import numpy as np
import pytest

pytest.importorskip("scipy")

from ml.preprocessing.augmentation import (  # noqa: E402
    apply_variant,
    distortion,
    echo,
    phone_quality,
)
from music_core.notes import Note  # noqa: E402

SR = 22050


def tone(freq: float = 220.0, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def band_energy(y: np.ndarray, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum())


@pytest.mark.parametrize("hard", [False, True])
def test_distortion_adds_harmonics_and_keeps_peak(hard):
    y = tone()
    out = distortion(y, SR, np.random.default_rng(0), hard=hard)
    assert np.max(np.abs(out)) == pytest.approx(np.max(np.abs(y)), rel=1e-3)
    # A pure 220 Hz tone gains odd harmonics (660 Hz, 1100 Hz, ...).
    assert band_energy(out, 600, 700) > 100 * band_energy(y, 600, 700) + 1e-6


def test_phone_removes_highs():
    y = tone(5000) + tone(1000)
    out = phone_quality(y, SR, np.random.default_rng(0))
    assert out.shape == y.shape
    assert band_energy(out, 4500, 5500) < 1e-3 * band_energy(out, 900, 1100)


def test_echo_repeats_after_the_delay():
    y = np.zeros(SR, dtype=np.float32)
    y[0] = 1.0
    out = echo(
        y, SR, np.random.default_rng(0), delay_s=(0.25, 0.25), feedback=(0.5, 0.5), mix=(0.5, 0.5)
    )
    delay = int(0.25 * SR)
    assert out[delay] == pytest.approx(0.5) and out[2 * delay] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "variant", [{"distortion": "soft"}, {"echo": True}, {"phone": True}, {"reverb": True}]
)
def test_effects_leave_labels_untouched(variant):
    notes = [Note(57, 0.1, 0.6, string=1, fret=12)]
    out, labels = apply_variant(tone(), notes, variant, SR, np.random.default_rng(0))
    assert labels == notes and out.shape == tone().shape and np.isfinite(out).all()
