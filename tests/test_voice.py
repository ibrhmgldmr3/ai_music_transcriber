import numpy as np
import pytest

from ml.inference.voice import (
    VoiceSettings,
    estimate_tuning,
    fit_to_range,
    transcribe_voice,
    voice_notes,
)
from music_core.notes import Note

SR = VoiceSettings().sample_rate


def sing(
    melody: list[tuple[int | None, float]],
    detune: float = 0.0,
    vibrato: float = 0.3,
    glide: float = 0.06,
    seed: int = 0,
) -> tuple[np.ndarray, list[Note]]:
    """A voice-like tone: harmonics, vibrato, glides between legato notes and a short dip
    in loudness where a note repeats the previous pitch. ``None`` is a rest.
    Returns the samples and the notes as sung (before ``detune``)."""
    rng = np.random.default_rng(seed)
    times = np.cumsum([0.0] + [d for _, d in melody])
    n = int(times[-1] * SR)
    t = np.arange(n) / SR
    pitch = np.zeros(n)
    amp = np.zeros(n)
    notes = []
    for i, ((p, _), start, end) in enumerate(zip(melody, times[:-1], times[1:])):
        seg = (t >= start) & (t < end)
        if p is None:
            continue
        pitch[seg] = p
        amp[seg] = 1.0
        notes.append(Note(p, float(start), float(end)))
        previous = melody[i - 1][0] if i else None
        if previous is not None and previous != p:  # legato glide into the note
            ramp = (t >= start - glide / 2) & (t < start + glide / 2)
            pitch[ramp] = previous + (p - previous) * (t[ramp] - (start - glide / 2)) / glide
        if previous == p:  # re-attack: a quick dip in loudness
            dip = np.abs(t - start) < 0.04
            amp[dip] = 0.08
    attack = np.convolve(amp, np.ones(int(0.02 * SR)) / int(0.02 * SR), mode="same")
    midi = pitch + detune + vibrato * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(440 * 2 ** ((midi - 69) / 12)) / SR
    tone = sum(np.sin(k * phase) / k for k in range(1, 11))
    y = attack * tone + 0.003 * rng.standard_normal(n)
    return (0.3 * y / np.abs(y).max()).astype(np.float32), notes


MELODY = [(60, 0.5), (62, 0.4), (64, 0.4), (64, 0.4), (None, 0.3), (67, 0.6), (65, 0.3), (64, 0.8)]


def assert_matches(estimated: list[Note], reference: list[Note], tolerance: float = 0.05):
    assert [n.pitch for n in estimated] == [n.pitch for n in reference]
    for est, ref in zip(estimated, reference):
        assert abs(est.start - ref.start) <= tolerance, (est, ref)
        assert abs(est.end - ref.end) <= 0.1, (est, ref)


def test_sung_melody_becomes_its_notes():
    y, reference = sing(MELODY)
    assert_matches(voice_notes(y), reference)


def test_a_singer_off_equal_temperament_is_counted_from_their_own_tuning():
    y, reference = sing(MELODY, detune=0.4)  # 40 cents sharp: rounding would give +1
    assert_matches(voice_notes(y), reference)


def test_estimate_tuning_wraps_around_the_semitone():
    midi = np.array([60.45, 62.47, 64.52, np.nan, 65.55])
    assert estimate_tuning(midi) == pytest.approx(0.5, abs=0.06) or estimate_tuning(
        midi
    ) == pytest.approx(-0.5, abs=0.06)
    assert estimate_tuning(np.array([59.8, 61.8, 63.8])) == pytest.approx(-0.2, abs=1e-6)


def test_silence_gives_no_notes():
    assert voice_notes(np.zeros(SR, dtype=np.float32)) == []


def test_fit_to_range_moves_a_whistle_down_and_leaves_a_fitting_melody_alone():
    whistle = [Note(p, 0, 1) for p in (86, 88, 91, 93)]
    moved, shift = fit_to_range(whistle, 40, 84)
    assert shift == -12 and [n.pitch for n in moved] == [74, 76, 79, 81]
    melody = [Note(p, 0, 1) for p in (40, 64, 67, 84)]
    assert fit_to_range(melody, 40, 84) == (melody, 0)
    # One note too high: an octave down makes all of them playable.
    assert fit_to_range([Note(p, 0, 1) for p in (60, 64, 86)], 40, 84)[1] == -12
    assert fit_to_range([], 40, 84) == ([], 0)


def test_transcribe_voice_places_the_notes_on_the_guitar(tmp_path):
    import soundfile as sf

    y, reference = sing(MELODY)
    path = tmp_path / "melody.wav"
    sf.write(path, y, SR)
    stages = []
    result = transcribe_voice(
        path, tuning=[38, 43, 48, 53, 57, 62], progress=lambda f, stage: stages.append(stage)
    )
    assert [n.pitch for n in result.notes] == [n.pitch for n in reference]
    assert all(n.string is not None and n.fret is not None for n in result.notes)
    assert result.tuning == [38, 43, 48, 53, 57, 62] and result.transpose == 0
    assert result.duration == pytest.approx(len(y) / SR)
    assert stages[0] == "loading" and "transcribing" in stages and stages[-1] == "finishing"
