import io

import mido
import pytest

from music_core.midi import BEND_RANGE, midi_to_notes, notes_to_midi_bytes
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


def _messages(notes):
    return [
        m
        for track in mido.MidiFile(file=io.BytesIO(notes_to_midi_bytes(notes))).tracks
        for m in track
    ]


def test_notes_without_techniques_stay_on_one_channel():
    messages = _messages(
        [Note(64, 0.0, 0.5, string=4, fret=5), Note(52, 0.0, 0.5, string=2, fret=2)]
    )
    assert {m.channel for m in messages if m.type == "note_on"} == {0}
    assert not any(m.type in ("pitchwheel", "control_change") for m in messages)


def test_a_bend_is_a_pitch_bend_curve_on_its_strings_channel():
    notes = [Note(64, 0.0, 1.0, string=4, fret=5, bend=2.0), Note(52, 0.0, 1.0, string=2, fret=2)]
    messages = _messages(notes)
    channel = {m.note: m.channel for m in messages if m.type == "note_on"}
    assert channel[64] != channel[52]
    # The bend range is set to an octave on every channel used.
    rpn = [(m.channel, m.control, m.value) for m in messages if m.type == "control_change"]
    assert (channel[64], 6, BEND_RANGE) in rpn and (channel[52], 6, BEND_RANGE) in rpn
    bends = [m.pitch for m in messages if m.type == "pitchwheel" and m.channel == channel[64]]
    assert max(bends) == round(2.0 / BEND_RANGE * 8191)  # a whole step up, held
    assert bends[-1] == 0  # reset when the note ends
    assert all(
        m.pitch == 0 for m in messages if m.type == "pitchwheel" and m.channel == channel[52]
    )


def test_vibrato_swings_both_ways():
    bends = [
        m.pitch
        for m in _messages([Note(64, 0.0, 1.0, string=4, fret=5, vibrato=True)])
        if m.type == "pitchwheel"
    ]
    assert min(bends) < 0 < max(bends)


def test_a_slide_glides_into_the_next_note_without_a_gap():
    notes = [
        Note(64, 0.0, 0.4, string=4, fret=5),
        Note(67, 0.5, 1.0, string=4, fret=8, technique="slide"),
    ]
    events, now = [], 0.0
    for message in mido.MidiFile(file=io.BytesIO(notes_to_midi_bytes(notes, tempo=120.0))):
        now += message.time
        if message.type in ("note_on", "note_off", "pitchwheel"):
            events.append((round(now, 3), message))
    off = next(t for t, m in events if m.type == "note_off" and m.note == 64)
    assert off == pytest.approx(0.5, abs=0.01)  # held until the slide lands
    before = [m.pitch for t, m in events if m.type == "pitchwheel" and t < 0.5]
    assert max(before) == pytest.approx(3 / BEND_RANGE * 8191, abs=300)  # glides up 3 frets
    landing = [m for t, m in events if t == pytest.approx(0.5, abs=0.002)]
    order = [m.type for m in landing]
    assert order.index("note_on") > order.index("pitchwheel")  # the new note starts unbent
    assert [m.pitch for m in landing if m.type == "pitchwheel"][-1] == 0


def test_hammer_ons_are_softer_and_read_back():
    notes = [
        Note(64, 0.0, 0.5, velocity=100, string=4, fret=5),
        Note(66, 0.5, 1.0, velocity=100, string=4, fret=7, technique="hammer_on"),
    ]
    decoded = midi_to_notes(io.BytesIO(notes_to_midi_bytes(notes)))
    assert [(n.pitch, n.velocity) for n in decoded] == [(64, 100), (66, 70)]
