"""Combine the splits of several prepared datasets into one set of split files.

    python scripts/mix_splits.py --out ml/data/splits/mixed \\
        guitarset=ml/data/splits/guitarset_fx egdb=ml/data/splits/egdb \\
        guitar_techs=ml/data/splits/guitar_techs

Each ``NAME=SPLITS_DIR`` adds that directory's train/val/test files, prefixed with
``NAME/`` so they resolve under ``ml/data/processed``. ``--max-val`` evenly thins each
source's validation list so no dataset dominates checkpoint selection. Train with
``--set paths.processed_dir=ml/data/processed paths.splits_dir=ml/data/splits/mixed``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.datasets.dataset import read_split

SPLITS = ("train", "val", "test")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-val", type=int, default=60, help="validation files per source (0: all)"
    )
    parser.add_argument("sources", nargs="+", metavar="NAME=SPLITS_DIR")
    args = parser.parse_args(argv)

    mixed: dict[str, list[str]] = {split: [] for split in SPLITS}
    for source in args.sources:
        name, _, directory = source.partition("=")
        if not name or not directory:
            parser.error(f"expected NAME=SPLITS_DIR, got {source!r}")
        for split in SPLITS:
            path = Path(directory) / f"{split}.txt"
            if path.exists():
                names = read_split(path)
                if split == "val" and 0 < args.max_val < len(names):
                    step = len(names) / args.max_val
                    names = [names[int(i * step)] for i in range(args.max_val)]
                mixed[split] += [f"{name}/{n}" for n in names]
                print(f"{name:<14} {split:<5} {len(names):5d}")

    args.out.mkdir(parents=True, exist_ok=True)
    for split, names in mixed.items():
        (args.out / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        print(f"{'mixed':<14} {split:<5} {len(names):5d}")


if __name__ == "__main__":
    main()
