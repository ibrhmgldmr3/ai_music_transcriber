"""EGDB / Guitar-TECHS loading, splits and preprocessing helpers."""

from pathlib import Path

import mido
import numpy as np
import pytest

from ml.datasets import egdb, guitar_techs
from ml.datasets.dataset import split_into_parts
from ml.preprocessing.annotations import load_string_midi, notes_to_array
from music_core.notes import Note

TUNING = [40, 45, 50, 55, 59, 64]


def write_string_midi(
    path: Path, notes_by_string: dict[int, list[tuple[int, float, float]]], names: str
) -> None:
    """Type-1 MIDI at 100 BPM; string 1 = high e on channel 0, track "1" or "e"."""
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=600000, time=0)])
    midi.tracks.append(tempo)
    for number, notes in notes_by_string.items():
        track = mido.MidiTrack()
        track.name = {"digits": str(number), "letters": "eBGDAE"[number - 1]}.get(names, "x")
        events = []
        for pitch, start, end in notes:
            events += [(start, "note_on", pitch, 90), (end, "note_off", pitch, 0)]
        now = 0
        for time, kind, pitch, velocity in sorted(events):
            tick = mido.second2tick(time, 480, 600000)
            track.append(
                mido.Message(
                    kind, note=pitch, velocity=velocity, channel=number - 1, time=round(tick) - now
                )
            )
            now = round(tick)
        midi.tracks.append(track)
    midi.save(str(path))


@pytest.mark.parametrize("names", ["digits", "letters", "none"])
def test_string_midi_positions(tmp_path, names):
    path = tmp_path / "labels.mid"
    # high e string, 5th fret; low E string, 3rd fret; a pitch the B string can't play.
    write_string_midi(
        tmp_path / "labels.mid",
        {1: [(69, 0.5, 1.0)], 6: [(43, 1.0, 2.0)], 2: [(50, 2.0, 2.5)]},
        names,
    )
    notes = load_string_midi(path, TUNING, string_from="channel" if names == "none" else "track")
    by_pitch = {n.pitch: n for n in notes}
    assert (by_pitch[69].string, by_pitch[69].fret) == (5, 5)
    assert (by_pitch[43].string, by_pitch[43].fret) == (0, 3)
    assert by_pitch[50].string is None and by_pitch[50].fret is None
    assert by_pitch[43].start == pytest.approx(1.0, abs=1e-3)
    assert by_pitch[43].end == pytest.approx(2.0, abs=1e-3)


def test_split_into_parts_keeps_onsets_where_notes_start():
    frame_rate = 10.0
    notes = [Note(60, 0.5, 2.5), Note(62, 2.2, 2.8), Note(64, 4.0, 5.0)]
    n_frames = 60
    onset = np.zeros((n_frames, 3), dtype=np.float32)
    onset[[5, 22, 40], [0, 1, 2]] = 1
    track = {
        "features": np.random.default_rng(0).random((8, n_frames)).astype(np.float16),
        "onset": onset,
        "frame": np.ones((n_frames, 3), dtype=np.float32),
        "notes": notes_to_array(notes),
    }
    parts = split_into_parts(track, part_frames=20, frame_rate=frame_rate)
    assert [p["features"].shape[1] for p in parts] == [20, 20, 20]
    assert sum(p["onset"].sum() for p in parts) == 3  # no onsets invented at the cuts
    first, second, third = (p["notes"] for p in parts)
    assert first[:, 0].tolist() == [60] and first[0, 2] == pytest.approx(2.0)  # clipped at the cut
    assert second[:, 0].tolist() == [62] and second[0, 1] == pytest.approx(0.2)
    assert third[:, 0].tolist() == [64] and third[0, 1] == pytest.approx(0.0)


def test_egdb_splits(tmp_path):
    for folder in ["audio_label", *(f"audio_{t}" for t in egdb.TRAIN_TONES + egdb.TEST_TONES)]:
        (tmp_path / folder).mkdir()
    for clip in (1, 200, 201, 215, 216, 240):
        (tmp_path / "audio_label" / f"{clip}.midi").touch()
        for tone in egdb.TRAIN_TONES + egdb.TEST_TONES:
            (tmp_path / f"audio_{tone}" / f"{clip}.wav").touch()
    splits = egdb.split_tracks(egdb.find_tracks(tmp_path))
    assert "egdb_001_DI" in splits["train"] and "egdb_200_Mesa" in splits["train"]
    assert not any(t.endswith(("JCjazz", "Plexi")) for t in splits["train"] + splits["val"])
    assert splits["val"] == ["egdb_201_DI", "egdb_201_Marshall", "egdb_215_DI", "egdb_215_Marshall"]
    assert len(splits["test"]) == 2 * 6  # clips 216 and 240, every tone


def test_guitar_techs_splits_hold_out_the_third_player(tmp_path):
    for section, takes in (("P1_scales", 12), ("P1_chords", 3), ("P3_music", 2)):
        (tmp_path / section / "midi").mkdir(parents=True)
        for folder, prefix, ext in guitar_techs.VERSIONS.values():
            (tmp_path / section / folder).mkdir(parents=True)
            for take in range(1, takes + 1):
                (tmp_path / section / folder / f"{prefix}_{take:02d}{ext}").touch()
        for take in range(1, takes + 1):
            (tmp_path / section / "midi" / f"midi_{take:02d}.mid").touch()

    tracks = guitar_techs.find_tracks(tmp_path)
    splits = guitar_techs.split_tracks(tracks)
    assert len(tracks) == (12 + 3 + 2) * 4
    assert (
        all(t.startswith("gtechs_P3_music_") for t in splits["test"]) and len(splits["test"]) == 8
    )
    # Takes sort as chords 01-03, scales 01-12: the tenth is scales_07.
    assert splits["val"] == [f"gtechs_P1_scales_07_{v}" for v in guitar_techs.VERSIONS]
    assert len(splits["train"]) == 14 * 4


def test_offset_between_recordings():
    sr = 22050
    rng = np.random.default_rng(1)
    clicks = np.zeros(sr * 3)
    clicks[rng.integers(0, sr * 2, 25)] = 1.0
    signal = np.convolve(clicks, rng.standard_normal(400) * np.exp(-np.arange(400) / 80), "same")
    for delay in (0.03, -0.045):
        shift = int(round(delay * sr))
        other = np.roll(signal, shift)
        assert guitar_techs.estimate_offset(signal, other, sr) == pytest.approx(delay, abs=0.007)


def test_mix_splits_prefixes_and_thins_validation(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from mix_splits import main as mix

    for name, n_val in (("a", 10), ("b", 3)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "train.txt").write_text("t1\nt2\n", encoding="utf-8")
        (folder / "val.txt").write_text("".join(f"v{i}\n" for i in range(n_val)), encoding="utf-8")
    out = tmp_path / "mixed"
    mix(["--out", str(out), "--max-val", "5", f"a={tmp_path / 'a'}", f"b={tmp_path / 'b'}"])
    assert (out / "train.txt").read_text(encoding="utf-8").split() == [
        "a/t1",
        "a/t2",
        "b/t1",
        "b/t2",
    ]
    val = (out / "val.txt").read_text(encoding="utf-8").split()
    assert val == ["a/v0", "a/v2", "a/v4", "a/v6", "a/v8", "b/v0", "b/v1", "b/v2"]
