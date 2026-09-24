"""Grid-search the tab optimizer's cost weights on annotated notes.

The optimizer gets the ground-truth notes of the validation split without their
string/fret and is scored on how often it picks the annotated position. The defaults
in ``music_core.tab.TabCostWeights`` came from this script on GuitarSet.

python scripts/tune_tab.py --split val
python scripts/tune_tab.py --split val --set paths.splits_dir=ml/data/splits/guitarset_full
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict, replace
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ml.config import load_config, parse_overrides
from ml.datasets.dataset import read_split
from ml.preprocessing.annotations import array_to_notes
from music_core.notes import Note
from music_core.tab import TabCostWeights, assign_tab

GRID = {
    "fret_height": [0.0, 0.1, 0.3, 0.6],
    "preferred_fret": [0, 3, 5, 7],
    "open_string": [0.0, 0.5, 1.0, 2.0],
    "span": [0.5, 1.0, 2.0],
    "position_change": [0.5, 1.0, 2.0],
    "string_change": [0.0, 0.3, 1.0],
}


def load_notes(processed_dir: Path, names: list[str]) -> list[list[Note]]:
    tracks = []
    for name in names:
        if "__" in name:  # skip offline-augmented variants
            continue
        with np.load(processed_dir / f"{name}.npz") as data:
            tracks.append([n for n in array_to_notes(data["notes"]) if n.string is not None])
    return tracks


def accuracy(
    tracks: list[list[Note]], tuning, num_frets: int, weights: TabCostWeights, tolerance: float
) -> float:
    correct = total = 0
    for reference in tracks:
        stripped = [replace(n, string=None, fret=None) for n in reference]
        placed = assign_tab(stripped, tuning, num_frets, tolerance, weights)
        correct += sum(p.string == r.string and p.fret == r.fret for p, r in zip(placed, reference))
        total += len(reference)
    return correct / total if total else 0.0


def _score(params: dict, tracks, tuning, num_frets, tolerance) -> tuple[float, dict]:
    weights = TabCostWeights(**params)
    return accuracy(tracks, tuning, num_frets, weights, tolerance), params


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="ml/configs/guitar.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.overrides))
    paths, tab = cfg["paths"], cfg["tab"]
    names = read_split(Path(paths["splits_dir"]) / f"{args.split}.txt")
    tracks = load_notes(Path(paths["processed_dir"]), names)
    tuning, num_frets = tuple(tab["tuning"]), tab["num_frets"]
    print(f"{len(tracks)} tracks, {sum(map(len, tracks))} notes")

    baseline = accuracy(tracks, tuning, num_frets, TabCostWeights(), 0.05)
    print(f"current defaults: accuracy={baseline:.3f}")

    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    score = partial(_score, tracks=tracks, tuning=tuning, num_frets=num_frets, tolerance=0.05)
    with Pool(args.workers) as pool:
        results = sorted(pool.map(score, combos, chunksize=8), key=lambda r: -r[0])

    print(f"\nTop of {len(combos)} combinations:")
    for acc, params in results[:5]:
        print(f"  accuracy={acc:.3f}  {params}")

    best = TabCostWeights(**results[0][1])
    print("\nchord_tolerance with the best weights:")
    for tolerance in (0.02, 0.03, 0.05, 0.08):
        print(f"  {tolerance}: {accuracy(tracks, tuning, num_frets, best, tolerance):.3f}")
    print(f"\nBest weights: {asdict(best)}")


if __name__ == "__main__":
    main()
