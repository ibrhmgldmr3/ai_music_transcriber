import numpy as np
import pytest

from ml.inference.techniques import (
    Bend,
    PitchTracker,
    TechniqueSettings,
    detect_techniques,
    find_bend,
    has_vibrato,
)
from music_core.notes import Note

SR = 44100


def tone(midi: np.ndarray, decay: float = 1.5) -> np.ndarray:
    """A plucked-string-like tone following a per-sample MIDI pitch curve."""
    hz = 440.0 * 2.0 ** ((midi - 69) / 12)
    phase = 2 * np.pi * np.cumsum(hz) / SR
    t = np.arange(len(midi)) / SR
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 9))
    return (harmonics * np.exp(-decay * t)).astype(np.float32)


def curve(*segments: tuple[float, float, float]) -> np.ndarray:
    """Pitch segments (seconds, from, to) joined end to end, linear within each."""
    return np.concatenate([np.linspace(a, b, int(seconds * SR)) for seconds, a, b in segments])


def note_curve(y: np.ndarray, note: Note) -> np.ndarray:
    return PitchTracker(y, SR).note_curve(note).midi


def test_steady_note_has_no_technique():
    y = tone(curve((0.6, 64, 64)))
    midi = note_curve(y, Note(64, 0.0, 0.6))
    assert np.nanmedian(np.abs(midi - 64)) < 0.05
    assert find_bend(midi, 64, SR / 256) is None
    assert not has_vibrato(midi, SR / 256)


def test_whole_step_bend():
    y = tone(curve((0.15, 64, 64), (0.08, 64, 66), (0.3, 66, 66)))
    assert find_bend(note_curve(y, Note(64, 0.0, 0.53)), 64, SR / 256) == Bend(64, 2.0)


def test_jump_to_another_note_is_not_a_bend():
    y = tone(curve((0.2, 64, 64), (0.3, 66, 66)))
    assert find_bend(note_curve(y, Note(64, 0.0, 0.5)), 64, SR / 256) is None


def test_vibrato():
    t = np.arange(int(0.8 * SR)) / SR
    y = tone(64 + 0.3 * np.sin(2 * np.pi * 6 * t))
    midi = note_curve(y, Note(64, 0.0, 0.8))
    assert has_vibrato(midi, SR / 256)
    assert find_bend(midi, 64, SR / 256) is None


def test_other_notes_are_masked():
    # A louder G4 on another string doesn't pass for a bend of the E4.
    y = tone(curve((0.6, 64, 64))) + 2 * tone(curve((0.6, 67, 67)))
    tracker = PitchTracker(y, SR)
    midi = tracker.note_curve(Note(64, 0.0, 0.6), [Note(67, 0.0, 0.6)]).midi
    assert np.nanmedian(np.abs(midi - 64)) < 0.1
    assert find_bend(midi, 64, SR / 256) is None


@pytest.mark.parametrize("glide, expected", [(0.06, "slide"), (0.0, None)])
def test_slide_between_notes_on_one_string(glide, expected):
    segments = [(0.4, 64, 64), (glide, 64, 67), (0.4, 67, 67)]
    y = tone(curve(*[s for s in segments if s[0] > 0]), decay=0.5)
    # The note model ends the first note where the second begins: halfway through.
    change = 0.4 + glide / 2
    notes = [
        Note(64, 0.0, change, string=4, fret=5),
        Note(67, change, 0.8 + glide, string=4, fret=8),
    ]
    first, second = detect_techniques(notes, y, SR)
    assert first.technique is None
    assert second.technique == expected


def test_a_picked_change_is_not_a_slide():
    # The same glide, but the second note starts 50 ms after the first ends.
    y = tone(curve((0.4, 64, 64), (0.06, 64, 67), (0.4, 67, 67)), decay=0.5)
    notes = [Note(64, 0.0, 0.4, string=4, fret=5), Note(67, 0.45, 0.86, string=4, fret=8)]
    assert [n.technique for n in detect_techniques(notes, y, SR)] == [None, None]


def test_a_slide_is_moved_onto_one_string():
    y = tone(curve((0.4, 64, 64), (0.06, 64, 67), (0.4, 67, 67)), decay=0.5)
    notes = [Note(64, 0.0, 0.43, string=4, fret=5), Note(67, 0.43, 0.86, string=3, fret=12)]
    first, second = detect_techniques(notes, y, SR, tuning=(40, 45, 50, 55, 59, 64))
    assert (first.string, first.fret) == (4, 5)
    assert (second.string, second.fret, second.technique) == (4, 8, "slide")


def test_a_semitone_glide_is_not_a_slide():
    y = tone(curve((0.4, 64, 64), (0.06, 64, 65), (0.4, 65, 65)), decay=0.5)
    notes = [Note(64, 0.0, 0.43, string=4, fret=5), Note(65, 0.43, 0.86, string=4, fret=6)]
    assert [n.technique for n in detect_techniques(notes, y, SR)] == [None, None]


def test_a_note_named_after_its_bend_is_set_back_to_the_fretted_note():
    # A quick half-step bend held: the note model hears mostly F4 (65).
    y = tone(curve((0.05, 64, 64), (0.05, 64, 65), (0.5, 65, 65)))
    (note,) = detect_techniques([Note(65, 0.0, 0.6, string=4, fret=6)], y, SR)
    assert (note.pitch, note.fret, note.bend) == (64, 5, 1.0)


def test_a_note_cut_where_the_bend_starts_is_read_on():
    # The note model ends the E4 where its pitch leaves E (0.17 s), before the bend is
    # held; the string still sounds bent after that.
    y = tone(curve((0.12, 64, 64), (0.08, 64, 66), (0.3, 66, 66)))
    cut = Note(64, 0.0, 0.17, string=4, fret=5)
    (note,) = detect_techniques([cut], y, SR, TechniqueSettings(bend_extend=0.0))
    assert note.bend is None
    (note,) = detect_techniques([cut], y, SR)
    assert (note.pitch, note.bend) == (64, 2.0)
    assert note.end == pytest.approx(0.37)


def test_a_note_is_not_read_into_the_next_one():
    # A picked D4 -> E4: the E4 isn't the D4's bend.
    y = np.concatenate([tone(curve((0.15, 62, 62))), tone(curve((0.3, 64, 64)))])
    notes = [Note(62, 0.0, 0.12, string=3, fret=0), Note(64, 0.15, 0.45, string=4, fret=5)]
    assert [n.bend for n in detect_techniques(notes, y, SR)] == [None, None]
