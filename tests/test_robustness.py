"""Regression tests for input-handling bugs in the ML pipeline and music_core."""

import io
from pathlib import Path

import numpy as np
import pytest

from ml.config import load_config, parse_overrides
from ml.datasets.guitarset import GuitarSetTrack, player_splits
from ml.inference.postprocess import decode_notes
from music_core.midi import midi_to_notes, notes_to_midi_bytes
from music_core.notes import Note


def _tracks():
    return [GuitarSetTrack(f"0{i}_BN1-129-Eb_comp", Path("a"), Path("b")) for i in range(6)]


def test_player_ids_from_cli_overrides_are_normalized():
    # YAML turns "04" into the integer 4; the split must still find player "04".
    val_player = parse_overrides(["dataset.val_player=04"])["dataset"]["val_player"]
    assert val_player == 4
    splits = player_splits(_tracks(), val_player, "05")
    assert len(splits["val"]) == 1 and len(splits["test"]) == 1 and len(splits["train"]) == 4


def test_invalid_player_split_is_rejected():
    with pytest.raises(ValueError):
        player_splits(_tracks(), "05", 5)
    with pytest.raises(ValueError):
        player_splits(_tracks(), "09", "05")


def test_config_inheritance_cycle_is_reported(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nx: 1\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\ny: 2\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("tempo", [0.0, -10.0, float("nan"), float("inf")])
def test_midi_export_survives_unusable_tempo(tempo):
    data = notes_to_midi_bytes([Note(60, 0.0, 0.5)], tempo=tempo)
    (note,) = midi_to_notes(io.BytesIO(data))
    assert note.pitch == 60 and note.end == pytest.approx(0.5, abs=1e-3)


def test_decode_accepts_plain_lists():
    frame = np.zeros((20, 2))
    frame[5:15, 1] = 0.9
    onset = np.zeros_like(frame)
    onset[5, 1] = 0.9
    (note,) = decode_notes(frame.tolist(), onset.tolist(), frame_rate=10.0, min_midi=40)
    assert note.pitch == 41
