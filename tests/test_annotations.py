import json

import numpy as np

from ml.preprocessing.annotations import (
    array_to_notes,
    load_guitarset_jams,
    notes_to_array,
    notes_to_targets,
)
from ml.preprocessing.augmentation import shift_notes, stretch_notes
from music_core.notes import Note


def test_load_guitarset_jams(tmp_path):
    jam = {
        "annotations": [
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": "0"},
                "data": [{"time": 0.5, "duration": 0.5, "value": 40.1, "confidence": None}],
            },
            {
                "namespace": "note_midi",
                "annotation_metadata": {"data_source": "5"},
                "data": [{"time": 1.0, "duration": 0.25, "value": 67.0, "confidence": None}],
            },
            {"namespace": "pitch_contour", "data": []},
        ]
    }
    path = tmp_path / "track.jams"
    path.write_text(json.dumps(jam), encoding="utf-8")

    notes = load_guitarset_jams(path)
    assert [(n.pitch, n.string, n.fret) for n in notes] == [(40, 0, 0), (67, 5, 3)]


def test_notes_to_targets():
    notes = [Note(60, 0.0, 1.0, string=2, fret=10)]
    targets = notes_to_targets(
        notes, n_frames=20, sample_rate=10, hop_length=1, min_midi=50, max_midi=70
    )  # 10 frames per second
    p = 60 - 50
    assert targets["frame"].shape == (20, 21)
    assert targets["frame"][:10, p].all() and not targets["frame"][10:, p].any()
    assert targets["onset"][0, p] == 1 and targets["onset"][1:, p].sum() == 0
    assert targets["offset"][10, p] == 1 and targets["offset"][:, p].sum() == 1
    assert (targets["tab"][:10, 2] == 11).all() and (targets["tab"][10:, 2] == 0).all()


def test_note_array_round_trip():
    notes = [Note(60, 0.0, 1.0), Note(64, 0.5, 1.5, string=4, fret=5)]
    assert array_to_notes(notes_to_array(notes)) == notes
    assert notes_to_array([]).shape == (0, 6)


def test_label_transforms_for_offline_augmentation():
    (shifted,) = shift_notes([Note(64, 0.0, 1.0, string=4, fret=5)], n_steps=-2)
    assert (shifted.pitch, shifted.string, shifted.fret) == (62, 4, 3)
    (out_of_range,) = shift_notes([Note(59, 0.0, 1.0, string=4, fret=0)], n_steps=-2)
    assert out_of_range.string is None and out_of_range.pitch == 57
    (stretched,) = stretch_notes([Note(60, 1.0, 2.0)], rate=0.5)  # half speed
    assert (stretched.start, stretched.end) == (2.0, 4.0)
    assert np.isclose(stretched.duration, 2.0)
