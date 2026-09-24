"""Compute features + frame targets for every GuitarSet track and write player splits.

    python scripts/prepare_dataset.py                          # full dataset (V1)
    python scripts/prepare_dataset.py --subset 5               # 5 tracks per player (V0, dev)
    python scripts/prepare_dataset.py --augment                # + offline variants (V2, train only)
    python scripts/prepare_dataset.py --set features.type=cqt  # CQT instead of log-mel

Splits: players 00-03 -> train, 04 -> val, 05 -> test (see ml/configs/guitar.yaml).
Augmented variants are named ``<track_id>__<variant>`` and only listed in train.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.config import load_config, parse_overrides
from ml.datasets.guitarset import find_tracks, player_splits, preprocess_track, subset_per_player


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="ml/configs/guitar.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--subset", type=int, metavar="N", help="only N tracks per player")
    parser.add_argument("--augment", action="store_true", help="render offline augmentations")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.overrides))
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
    augmentation = cfg.get("offline_augmentation") or {}
    variants = (
        augmentation.get("variants", []) if (args.augment or augmentation.get("enabled")) else []
    )
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
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (splits_dir / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        print(f"{name:<5} {len(ids):4d} tracks")
    print(f"Features in {processed_dir}, splits in {splits_dir}")


if __name__ == "__main__":
    main()
