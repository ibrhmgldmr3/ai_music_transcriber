"""Compute features + frame targets for a dataset and write its splits.

    python scripts/prepare_dataset.py                          # GuitarSet (V1)
    python scripts/prepare_dataset.py --subset 5               # 5 tracks per player (V0, dev)
    python scripts/prepare_dataset.py --augment                # + offline variants (V2, train only)
    python scripts/prepare_dataset.py --set features.type=cqt  # CQT instead of log-mel
    python scripts/prepare_dataset.py --dataset egdb           # electric guitar, amp tones
    python scripts/prepare_dataset.py --dataset egdb_aligned   # aligned timing (align_egdb.py)
    python scripts/prepare_dataset.py --dataset guitar_techs   # electric guitar, room mics

GuitarSet splits: players 00-03 -> train, 04 -> val, 05 -> test (ml/configs/guitar.yaml).
Augmented variants are named ``<track_id>__<variant>`` and only listed in train.txt.
EGDB and Guitar-TECHS go to ``ml/data/processed/<dataset>`` and ``ml/data/splits/<dataset>``
with the splits described in their modules; long training and validation takes are
cut into ``<track_id>__partNN`` pieces, so every dataset is sampled by duration, not by
file, and validation never runs a many-minute take through the model at once.
Combine datasets for training with ``scripts/mix_splits.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ml.config import load_config, parse_overrides
from ml.datasets import egdb, guitar_techs
from ml.datasets.dataset import split_into_parts
from ml.datasets.guitarset import (
    find_tracks,
    player_splits,
    preprocess_audio,
    preprocess_track,
    subset_per_player,
)
from ml.preprocessing.augmentation import apply_variant

EXTRA_DATASETS = {"egdb": egdb, "egdb_aligned": egdb.ALIGNED, "guitar_techs": guitar_techs}
PART_SECONDS = 30.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="ml/configs/guitar.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dataset", default="guitarset", choices=["guitarset", *EXTRA_DATASETS])
    parser.add_argument("--subset", type=int, metavar="N", help="only N tracks per player")
    parser.add_argument("--augment", action="store_true", help="render offline augmentations")
    parser.add_argument(
        "--variants", nargs="*", metavar="NAME", help="only these offline variants (with --augment)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.overrides))
    if args.dataset == "guitarset":
        prepare_guitarset(cfg, args)
    else:
        prepare_extra(cfg, args.dataset, args.overwrite, offline_variants(cfg, args))


def offline_variants(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """The configured offline variants to render (all, or those named by ``--variants``)."""
    augmentation = cfg.get("offline_augmentation") or {}
    if not (args.augment or augmentation.get("enabled")):
        return []
    variants = augmentation.get("variants", [])
    if args.variants:
        unknown = set(args.variants) - {v["name"] for v in variants}
        if unknown:
            sys.exit(f"Unknown variants: {', '.join(sorted(unknown))}")
        variants = [v for v in variants if v["name"] in args.variants]
    return variants


def prepare_guitarset(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    paths, dataset = cfg["paths"], cfg["dataset"]
    raw_dir = Path(paths["raw_dir"])
    processed_dir, splits_dir = Path(paths["processed_dir"]), Path(paths["splits_dir"])

    tracks = find_tracks(raw_dir, dataset["audio_type"])
    if not tracks:
        sys.exit(f"No GuitarSet tracks found in {raw_dir}. Run scripts/download_dataset.py first.")
    if args.subset:
        tracks = subset_per_player(tracks, args.subset)

    splits = player_splits(tracks, dataset["val_player"], dataset["test_player"])
    train_ids = set(splits["train"])
    variants = offline_variants(cfg, args)
    rng = np.random.default_rng(cfg.get("seed", 42))

    processed_dir.mkdir(parents=True, exist_ok=True)
    augmented_ids: list[str] = []
    for track in tqdm(tracks, desc="preprocessing"):
        jobs = [(track.track_id, None)]
        if track.track_id in train_ids:  # never augment validation / test data
            jobs += [(f"{track.track_id}__{v['name']}", v) for v in variants]
        for output_id, variant in jobs:
            output = processed_dir / f"{output_id}.npz"
            if variant is not None:
                augmented_ids.append(output_id)
            if output.exists() and not args.overwrite:
                continue
            np.savez_compressed(output, **preprocess_track(track, cfg, variant, rng))

    splits["train"] += augmented_ids
    write_splits(splits, splits_dir)
    print(f"Features in {processed_dir}, splits in {splits_dir}")


def prepare_extra(
    cfg: dict[str, Any], name: str, overwrite: bool, variants: list[dict[str, Any]]
) -> None:
    """``variants`` are rendered for the training tracks the dataset module calls
    ``augmentable`` (all, without that function), as ``<track_id>__<variant>``."""
    module = EXTRA_DATASETS[name]
    data_root = Path(cfg["paths"]["raw_dir"]).parent.parent  # ml/data
    raw_dir = data_root / "raw" / name
    processed_dir, splits_dir = data_root / "processed" / name, data_root / "splits" / name

    tracks = module.find_tracks(raw_dir)
    if not tracks:
        sys.exit(f"No {name} tracks found in {raw_dir}.")
    splits = module.split_tracks(tracks)
    train_ids = set(splits["train"])
    cut_ids = train_ids | set(splits.get("val", []))  # test tracks stay whole
    augmentable = getattr(module, "augmentable", lambda track: True)
    frame_rate = cfg["audio"]["sample_rate"] / cfg["audio"]["hop_length"]
    part_frames = int(PART_SECONDS * frame_rate)
    rng = np.random.default_rng(cfg.get("seed", 42))

    processed_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}  # track id -> the files it became
    for track in tqdm(tracks, desc=f"preprocessing {name}"):
        jobs: list[tuple[str, dict[str, Any] | None]] = [(track.track_id, None)]
        if track.track_id in train_ids and augmentable(track):
            jobs += [(f"{track.track_id}__{v['name']}", v) for v in variants]
        written[track.track_id] = []
        loaded = None
        for output_id, variant in jobs:
            existing = sorted(processed_dir.glob(f"{output_id}.npz")) + sorted(
                processed_dir.glob(f"{output_id}__part*.npz")
            )
            if existing and not overwrite:
                written[track.track_id] += [p.stem for p in existing]
                continue
            loaded = loaded or module.load_track(track, cfg)
            y, notes = loaded
            if variant is not None:
                tab = cfg["tab"]
                y, notes = apply_variant(
                    y, notes, variant, cfg["audio"]["sample_rate"], rng,
                    tab["tuning"], tab["num_frets"],
                )  # fmt: skip
            data = preprocess_audio(y, notes, cfg)
            if track.track_id in cut_ids and data["features"].shape[1] > 1.5 * part_frames:
                parts = split_into_parts(data, part_frames, frame_rate)
                ids = [f"{output_id}__part{i:02d}" for i in range(len(parts))]
            else:
                parts, ids = [data], [output_id]
            for part_id, part in zip(ids, parts):
                np.savez_compressed(processed_dir / f"{part_id}.npz", **part)
            written[track.track_id] += ids

    write_splits(
        {split: [i for t in ids for i in written[t]] for split, ids in splits.items()}, splits_dir
    )
    print(f"Features in {processed_dir}, splits in {splits_dir}")


def write_splits(splits: dict[str, list[str]], splits_dir: Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        (splits_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        print(f"{split:<5} {len(ids):5d} files")


if __name__ == "__main__":
    main()
