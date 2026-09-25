"""Download and unpack Guitar-TECHS (ICASSP 2025, CC BY 4.0) from Zenodo, ~4.1 GB.

    python scripts/download_guitar_techs.py --out ml/data/raw/guitar_techs

The nine archives download in parallel (a single Zenodo connection is slow) and are
unpacked next to themselves, skipping macOS metadata.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RECORD = "https://zenodo.org/records/14963133/files/{}.zip?download=1"
ARCHIVES = [
    f"{player}_{section}"
    for player in ("P1", "P2")
    for section in ("chords", "scales", "singlenotes", "techniques")
] + ["P3_music"]


def fetch(name: str, out: Path) -> Path:
    target = out / f"{name}.zip"
    if target.exists() and zipfile.is_zipfile(target):
        return target
    tmp = target.with_suffix(".zip.part")
    with urllib.request.urlopen(RECORD.format(name), timeout=300) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    tmp.replace(target)
    print(f"downloaded {target.name}", flush=True)
    return target


def unpack(archive: Path, out: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        members = [
            n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith(".DS_Store")
        ]
        for member in members:  # refuse paths that would leave the output folder
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise ValueError(f"unsafe path in {archive.name}: {member}")
        z.extractall(out, members=members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("ml/data/raw/guitar_techs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(len(ARCHIVES)) as pool:
        archives = list(pool.map(lambda name: fetch(name, args.out), ARCHIVES))
    for archive in archives:
        unpack(archive, args.out)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
