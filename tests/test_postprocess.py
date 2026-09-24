import numpy as np
import pytest

from ml.evaluation.metrics import frame_metrics, tab_metrics, tab_note_accuracy
from ml.inference.postprocess import assign_positions_from_tab, decode_notes, tab_position_probs
from music_core.notes import Note

TUNING = (40, 45, 50, 55, 59, 64)


def test_decode_single_note_with_confidence():
    frame = np.zeros((100, 5))
    onset = np.zeros_like(frame)
    frame[10:30, 2] = 0.9
    onset[10, 2] = 0.7
    (note,) = decode_notes(frame, onset, frame_rate=10.0, min_midi=60)
    assert note.pitch == 62
    assert (note.start, note.end) == (pytest.approx(1.0), pytest.approx(3.0))
    assert note.confidence == pytest.approx(0.8)


def test_repeated_onset_splits_note():
    frame = np.zeros((50, 1))
    onset = np.zeros_like(frame)
    frame[10:40, 0] = 0.9
    onset[[10, 25], 0] = 0.9
    notes = decode_notes(frame, onset, frame_rate=10.0, min_midi=40)
    assert [(n.start, n.end) for n in notes] == [(1.0, 2.5), (2.5, 4.0)]


def test_offset_ends_note_early():
    frame = np.zeros((50, 1))
    onset, offset = np.zeros_like(frame), np.zeros_like(frame)
    frame[10:40, 0] = 0.9
    onset[10, 0] = 0.9
    offset[20, 0] = 0.9
    (note,) = decode_notes(frame, onset, offset, frame_rate=10.0, min_midi=40)
    assert note.end == pytest.approx(2.0)


def test_tab_head_probabilities_choose_position():
    # On its own the optimizer plays E4 on the B string (fret 5); the model says G9.
    tab_probs = np.full((10, 6, 22), 0.01)
    tab_probs[:, 3, 10] = 0.9  # string 3 (G), fret 9
    (note,) = assign_positions_from_tab(
        [Note(64, 0.0, 1.0)], tab_probs, frame_rate=10.0, tuning=TUNING, num_frets=20
    )
    assert (note.string, note.fret) == (3, 9)


def test_tab_position_probs_are_normalized_over_candidates():
    tab_probs = np.zeros((10, 6, 22))
    tab_probs[:, 4, 6] = 0.6  # B string, fret 5
    tab_probs[:, 3, 10] = 0.2  # G string, fret 9
    (probs,) = tab_position_probs(
        [Note(64, 0.0, 1.0)], tab_probs, frame_rate=10.0, tuning=TUNING, num_frets=20
    )
    assert set(probs) == {(5, 0), (4, 5), (3, 9), (2, 14), (1, 19)}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs[(4, 5)] == pytest.approx(0.75) and probs[(3, 9)] == pytest.approx(0.25)


def test_frame_metrics():
    assert frame_metrics(np.eye(4, dtype=bool), np.eye(4, dtype=bool))["f1"] == 1.0
    m = frame_metrics(np.array([[1, 1]]), np.array([[1, 0]]))
    assert (m["precision"], m["recall"]) == (0.5, 1.0)


def test_tab_metrics_right_pitch_wrong_string():
    pred = np.array([[0, 1, 0, 0, 0, 0]])  # A string open -> A2 (45)
    target = np.array([[6, 0, 0, 0, 0, 0]])  # low E fret 5 -> A2 (45)
    m = tab_metrics(pred, target, tuning=TUNING, min_midi=40, n_pitches=49)
    assert m["pitch_f1"] == 1.0 and m["f1"] == 0.0 and m["tdr"] == 0.0


def test_tab_note_accuracy():
    reference = [Note(45, 0.0, 1.0, string=0, fret=5), Note(50, 1.0, 2.0, string=2, fret=0)]
    estimated = [Note(45, 0.01, 1.0, string=1, fret=0), Note(50, 1.02, 2.0, string=2, fret=0)]
    result = tab_note_accuracy(reference, estimated)
    assert result == {"accuracy": 0.5, "matched": 2.0}
