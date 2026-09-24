"""Download GuitarSet (Zenodo record 3371780) into ml/data/raw/guitarset.

python scripts/download_dataset.py                    # annotations + mono mic audio (~0.7 GB)
python scripts/download_dataset.py --audio mic mix    # more audio variants
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from ml.datasets.guitarset import ANNOTATION_DIR, AUDIO_TYPES, ZENODO_URL

TIMEOUT_SECONDS = 60  # per socket operation, so a stalled connection fails instead of hanging


def download(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS download: {url}")
    partial = destination.with_suffix(destination.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response,
        partial.open("wb") as out,
    ):
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    if total is not None and partial.stat().st_size != total:
        raise OSError(f"Incomplete download of {destination.name}")
    partial.replace(destination)


def extract(archive: Path, target: Path) -> None:
    """Extract into a temporary folder first, so an interrupted run is never mistaken for
    a finished one, and refuse members that would land outside the target (zip slip)."""
    staging = target.with_name(target.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    root = staging.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in {archive.name}: {member}")
        zf.extractall(staging)
    # Flatten "<name>/<name>/..." if the archive contains a top-level folder.
    nested = staging / target.name
    source = nested if nested.is_dir() else staging
    shutil.rmtree(target, ignore_errors=True)
    source.replace(target)
    shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("ml/data/raw/guitarset"))
    parser.add_argument("--audio", nargs="+", default=["mic"], choices=sorted(AUDIO_TYPES))
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    archives = [ANNOTATION_DIR] + [AUDIO_TYPES[a][0] for a in args.audio]
    for name in archives:
        target = args.out / name
        if target.is_dir() and any(target.iterdir()):
            print(f"[skip] {target} already exists")
            continue
        archive = args.out / f"{name}.zip"
        if not archive.exists():
            try:
                download(ZENODO_URL.format(archive=name), archive)
            except OSError as exc:
                sys.exit(f"Download of {name}.zip failed: {exc}")
        print(f"Extracting {archive.name} ...")
        extract(archive, target)
        if not args.keep_archives:
            archive.unlink()
    print(f"GuitarSet ready in {args.out}")


if __name__ == "__main__":
    main()
