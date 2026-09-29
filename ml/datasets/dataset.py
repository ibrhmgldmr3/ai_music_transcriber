"""PyTorch dataset over preprocessed ``.npz`` tracks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.preprocessing.augmentation import FeatureAugmenter

TARGET_KEYS = ("onset", "frame", "offset", "tab")
PITCH_KEYS = ("onset", "frame", "offset")


def extend_pitch_range(
    track: dict[str, np.ndarray], n_pitches: int, min_midi: int, sample_rate: int, hop_length: int
) -> dict[str, np.ndarray]:
    """Grow the piano-roll targets of ``track`` downwards to ``n_pitches`` rows.

    Preprocessed files keep the range they were made with (E2-E6). A model with a lower
    ``min_midi`` (for lowered tunings) gets the missing bottom rows rasterized from the
    track's notes here, so the data needn't be preprocessed again.
    """
    stored = track["frame"].shape[1]
    missing = n_pitches - stored
    if missing <= 0:
        if missing < 0:
            raise ValueError(f"Targets have {stored} pitches, the model only {n_pitches}")
        return track
    from ml.preprocessing.annotations import array_to_notes, notes_to_targets

    low = [n for n in array_to_notes(track["notes"]) if min_midi <= n.pitch < min_midi + missing]
    extra = notes_to_targets(
        low,
        n_frames=track["frame"].shape[0],
        sample_rate=sample_rate,
        hop_length=hop_length,
        min_midi=min_midi,
        max_midi=min_midi + missing - 1,
    )
    extended = dict(track)
    for key in PITCH_KEYS:
        if key in track:
            extended[key] = np.concatenate(
                [extra[key].astype(track[key].dtype), track[key]], axis=1
            )
    return extended


def split_into_parts(
    track: dict[str, np.ndarray], part_frames: int, frame_rate: float
) -> list[dict[str, np.ndarray]]:
    """Cut a preprocessed track into consecutive pieces of about ``part_frames`` frames.

    Targets are sliced, not recomputed, so a note crossing a cut keeps its single onset in
    the piece where it starts. Reference notes go to the piece they start in.
    """
    n_frames = track["features"].shape[1]
    n_parts = max(1, round(n_frames / part_frames))
    bounds = np.linspace(0, n_frames, n_parts + 1).round().astype(int)
    parts = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        part = {"features": track["features"][:, start:end]}
        for key in TARGET_KEYS:
            if key in track:
                part[key] = track[key][start:end]
        if "notes" in track:
            notes = track["notes"].copy()  # columns: pitch, start, end, velocity, string, fret
            t0, t1 = start / frame_rate, end / frame_rate
            notes = notes[(notes[:, 1] >= t0) & (notes[:, 1] < t1)]
            notes[:, 1] -= t0
            notes[:, 2] = np.minimum(notes[:, 2] - t0, t1 - t0)
            part["notes"] = notes
        parts.append(part)
    return parts


def read_split(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


class TranscriptionDataset(Dataset):
    """Frame-level transcription dataset.

    Each ``.npz`` file holds ``features`` (bins, T), ``onset`` / ``frame`` / ``offset``
    (T, pitches) and optionally ``tab`` (T, strings). With ``segment_frames`` set, random
    fixed-length crops are returned (padded if the track is shorter); otherwise whole tracks.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        segment_frames: int | None = None,
        augmenter: FeatureAugmenter | None = None,
        repeats: int = 1,
        preload: bool = True,
        pitch_range: tuple[int, int, int, int] | None = None,
    ):
        """``pitch_range`` = (n_pitches, min_midi, sample_rate, hop_length) of the model,
        when it covers lower pitches than the files (see ``extend_pitch_range``)."""
        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError("TranscriptionDataset needs at least one file")
        self.segment_frames = segment_frames
        self.augmenter = augmenter
        self.repeats = max(1, repeats)
        self.pitch_range = pitch_range
        self._cache = [self._load(f) for f in self.files] if preload else None

    @classmethod
    def from_split(
        cls, processed_dir: str | Path, split_file: str | Path, **kwargs: Any
    ) -> TranscriptionDataset:
        files = [Path(processed_dir) / f"{name}.npz" for name in read_split(split_file)]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} processed files missing (e.g. {missing[0]}). "
                "Run scripts/prepare_dataset.py first."
            )
        return cls(files, **kwargs)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        with np.load(path) as data:
            track = {key: data[key] for key in data.files}
        if self.pitch_range is not None:
            track = extend_pitch_range(track, *self.pitch_range)
        return track

    def __len__(self) -> int:
        return len(self.files) * self.repeats

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index = index % len(self.files)
        track = (
            self._cache[file_index]
            if self._cache is not None
            else self._load(self.files[file_index])
        )
        features, targets = self._crop(track)

        if self.augmenter is not None:
            # Seed from torch so every DataLoader worker draws different augmentations.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            features = self.augmenter(features, rng)

        item: dict[str, Any] = {
            "features": torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            "name": self.files[file_index].stem,
        }
        for key, value in targets.items():
            dtype = np.int64 if key == "tab" else np.float32
            item[key] = torch.from_numpy(value.astype(dtype))
        return item

    def _crop(self, track: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        features = track["features"]
        targets = {key: track[key] for key in TARGET_KEYS if key in track}
        length = self.segment_frames
        if not length:
            return features, targets

        n_frames = features.shape[1]
        start = (
            int(torch.randint(0, n_frames - length + 1, (1,)).item()) if n_frames > length else 0
        )
        window = slice(start, start + length)
        features = features[:, window]
        targets = {key: value[window] for key, value in targets.items()}

        pad = length - features.shape[1]
        if pad > 0:
            fill = float(features.min()) if features.size else 0.0
            features = np.pad(features, ((0, 0), (0, pad)), constant_values=fill)
            targets = {key: np.pad(value, ((0, pad), (0, 0))) for key, value in targets.items()}
        return features, targets
