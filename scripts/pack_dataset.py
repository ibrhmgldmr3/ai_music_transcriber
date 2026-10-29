"""Zip a config's preprocessed tracks for training elsewhere (e.g. Colab via Google Drive).

    python scripts/pack_dataset.py --config ml/configs/guitar_contour.yaml

writes ``<processed_dir name>.zip`` holding ``<processed_dir name>/<track>.npz`` for every
track in the config's train / val / test splits. The ``.npz`` files are already
compressed, so they are stored as they are. The splits themselves are in git.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from ml.config import load_config, parse_overrides
from ml.datasets.dataset import read_split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="ml/configs/guitar_contour.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, help="zip file (default: <processed dir>.zip)")
    args = parser.parse_args(argv)

    paths = load_config(args.config, parse_overrides(args.overrides))["paths"]
    processed_dir, splits_dir = Path(paths["processed_dir"]), Path(paths["splits_dir"])
    names = sorted(
        {
            name
            for split in ("train", "val", "test")
            for name in read_split(splits_dir / f"{split}.txt")
        }
    )
    files = [processed_dir / f"{name}.npz" for name in names]
    missing = [f for f in files if not f.exists()]
    if missing:
        sys.exit(f"{len(missing)} files missing (e.g. {missing[0]}); run prepare_dataset.py first")

    out = args.out or Path(f"{processed_dir.name}.zip")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:
        for f in files:
            archive.write(f, f"{processed_dir.name}/{f.name}")
    size = out.stat().st_size / 1e9
    print(f"{len(files)} tracks -> {out} ({size:.2f} GB)")


if __name__ == "__main__":
    main()
