import io

import pytest

from music_core.midi import midi_to_notes, notes_to_midi_bytes
from music_core.notes import Note


def test_midi_round_trip():
    notes = [Note(60, 0.0, 0.5, velocity=100), Note(64, 0.5, 1.0, velocity=90), Note(67, 0.5, 1.25)]
    data = notes_to_midi_bytes(notes, tempo=120.0)
    decoded = midi_to_notes(io.BytesIO(data))

    assert [n.pitch for n in decoded] == [60, 64, 67]
    for original, restored in zip(notes, decoded):
        assert restored.start == pytest.approx(original.start, abs=1e-3)
        assert restored.end == pytest.approx(original.end, abs=1e-3)
        assert restored.velocity == original.velocity
