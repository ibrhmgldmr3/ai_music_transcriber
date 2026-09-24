import pytest

from music_core.notes import Note, hz_to_midi, midi_to_hz, midi_to_name, name_to_midi
from music_core.timing import quantize_notes


def test_midi_names_round_trip():
    assert midi_to_name(60) == "C4"
    assert midi_to_name(40) == "E2"
    assert name_to_midi("A4") == 69
    assert name_to_midi("Eb3") == 51
    for midi in range(21, 109):
        assert name_to_midi(midi_to_name(midi)) == midi


def test_frequency_conversion():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert hz_to_midi(midi_to_hz(52)) == pytest.approx(52)


def test_note_validation_and_dict():
    with pytest.raises(ValueError):
        Note(60, start=1.0, end=0.5)
    note = Note.from_dict({"pitch": 64, "start": 0.0, "end": 0.5, "unknown": 1})
    assert note.duration == 0.5
    assert note.to_dict()["confidence"] is None


def test_quantize_notes_snaps_to_grid():
    (note,) = quantize_notes([Note(60, 0.13, 0.26)], tempo=120, subdivision=4)  # grid 0.125 s
    assert note.start == pytest.approx(0.125)
    assert note.end == pytest.approx(0.25)
