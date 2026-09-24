"""GuitarSet (Xi et al., ISMIR 2018): file layout, preprocessing and player-based splits.

Expected layout under ``raw_dir`` (as produced by ``scripts/download_dataset.py``)::

    annotation/00_BN1-129-Eb_comp.jams
    audio_mono-mic/00_BN1-129-Eb_comp_mic.wav
    ...
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.config import tab_shape
from ml.preprocessing.annotations import load_guitarset_jams, notes_to_array, notes_to_targets
from ml.preprocessing.audio import load_audio
from ml.preprocessing.augmentation import apply_variant
from ml.preprocessing.spectrogram import compute_features
from music_core.notes import Note

ZENODO_URL = "https://zenodo.org/records/3371780/files/{archive}.zip?download=1"
ANNOTATION_DIR = "annotation"
# audio_type -> (directory / archive name, file suffix)
AUDIO_TYPES: dict[str, tuple[str, str]] = {
    "mic": ("audio_mono-mic", "_mic.wav"),
    "mix": ("audio_mono-pickup_mix", "_mix.wav"),
    "hex_cln": ("audio_hex-pickup_debleeded", "_hex_cln.wav"),
    "hex": ("audio_hex-pickup_original", "_hex.wav"),
}
PLAYERS = ("00", "01", "02", "03", "04", "05")


@dataclass(frozen=True)
class GuitarSetTrack:
    track_id: str  # e.g. "00_BN1-129-Eb_comp"
    audio_path: Path
    jams_path: Path

    @property
    def player(self) -> str:
        return self.track_id.split("_", 1)[0]

    @property
    def mode(self) -> str:
        """'comp' (accompaniment) or 'solo'."""
        return self.track_id.rsplit("_", 1)[-1]


def find_tracks(raw_dir: str | Path, audio_type: str = "mic") -> list[GuitarSetTrack]:
    """Pair every annotation file with its audio file of the requested type."""
    if audio_type not in AUDIO_TYPES:
        raise ValueError(f"Unknown audio_type {audio_type!r}; choose from {sorted(AUDIO_TYPES)}")
    raw_dir = Path(raw_dir)
    audio_dir, suffix = AUDIO_TYPES[audio_type]
    audio_index = {p.name: p for p in (raw_dir / audio_dir).rglob(f"*{suffix}")}

    tracks = []
    for jams_path in sorted((raw_dir / ANNOTATION_DIR).rglob("*.jams")):
        audio_path = audio_index.get(f"{jams_path.stem}{suffix}")
        if audio_path is not None:
            tracks.append(GuitarSetTrack(jams_path.stem, audio_path, jams_path))
    return tracks


def load_track(track: GuitarSetTrack, cfg: dict[str, Any]) -> tuple[np.ndarray, list[Note]]:
    audio = cfg["audio"]
    y = load_audio(track.audio_path, audio["sample_rate"], mono=True, normalize=audio["normalize"])
    return y, load_guitarset_jams(track.jams_path, tuning=cfg["tab"]["tuning"])


def preprocess_track(
    track: GuitarSetTrack,
    cfg: dict[str, Any],
    variant: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Features + targets for one track (optionally an offline augmentation variant)."""
    y, notes = load_track(track, cfg)
    if variant is not None:
        tab = cfg["tab"]
        y, notes = apply_variant(
            y,
            notes,
            variant,
            sample_rate=cfg["audio"]["sample_rate"],
            rng=rng or np.random.default_rng(),
            tuning=tab["tuning"],
            num_frets=tab["num_frets"],
        )
    return preprocess_audio(y, notes, cfg)


def preprocess_audio(
    y: np.ndarray, notes: list[Note], cfg: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Features + frame targets + reference notes, ready for ``np.savez_compressed``."""
    audio, labels, tab = cfg["audio"], cfg["labels"], cfg["tab"]
    features = compute_features(y, cfg)
    num_strings, _ = tab_shape(cfg)
    targets = notes_to_targets(
        notes,
        n_frames=features.shape[1],
        sample_rate=audio["sample_rate"],
        hop_length=audio["hop_length"],
        min_midi=labels["min_midi"],
        max_midi=labels["max_midi"],
        num_strings=num_strings,
        num_frets=tab["num_frets"],
        onset_frames=labels.get("onset_frames", 1),
        offset_frames=labels.get("offset_frames", 1),
    )
    return {
        "features": features.astype(np.float16),
        "onset": targets["onset"],
        "frame": targets["frame"],
        "offset": targets["offset"],
        "tab": targets["tab"],
        "notes": notes_to_array(notes),
    }


def player_splits(
    tracks: Iterable[GuitarSetTrack],
    val_player: str = "04",
    test_player: str = "05",
) -> dict[str, list[str]]:
    """Leave-players-out split so the model is evaluated on unseen guitarists."""
    # "--set dataset.val_player=04" arrives from YAML as the integer 4.
    val_player, test_player = _player_id(val_player), _player_id(test_player)
    if val_player == test_player:
        raise ValueError(f"Validation and test player must differ (both {val_player})")
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for track in tracks:
        if track.player == test_player:
            splits["test"].append(track.track_id)
        elif track.player == val_player:
            splits["val"].append(track.track_id)
        else:
            splits["train"].append(track.track_id)
    return splits


def _player_id(player: str | int) -> str:
    player_id = str(player).strip().zfill(2)
    if player_id not in PLAYERS:
        raise ValueError(f"Unknown GuitarSet player {player!r}; expected one of {PLAYERS}")
    return player_id


def subset_per_player(tracks: list[GuitarSetTrack], per_player: int) -> list[GuitarSetTrack]:
    """Small development subset (V0): the first ``per_player`` tracks of every player."""
    counts: dict[str, int] = {}
    subset = []
    for track in tracks:
        if counts.get(track.player, 0) < per_player:
            counts[track.player] = counts.get(track.player, 0) + 1
            subset.append(track)
    return subset
