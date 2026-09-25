"""Download EGDB (Chen et al., ICASSP 2022) from its public Google Drive folder.

240 electric guitar clips: direct input, five amplifier renderings and MIDI labels with
the string of every note, plus real recordings ("RealData"). The README in the folder
defines the split: clips 216-240 for testing, JCjazz and Plexi as unseen test tones.

    python scripts/download_egdb.py --out ml/data/raw/egdb
"""

from __future__ import annotations

import argparse
import html
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_FOLDER = "1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_"
LISTING = "https://drive.google.com/embeddedfolderview?id={}"
DOWNLOAD = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
_ENTRY = re.compile(
    r'<a href="https://drive\.google\.com/(file/d|drive/folders)/([\w-]+)[^"]*"[^>]*>.*?'
    r'<div class="flip-entry-title">(.*?)</div>',
    re.S,
)


def _get(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def list_folder(folder_id: str, prefix: Path = Path()) -> list[tuple[Path, str]]:
    """(relative path, file id) of every file under a public folder, recursively."""
    page = _get(LISTING.format(folder_id)).decode("utf-8")
    files: list[tuple[Path, str]] = []
    for kind, item_id, title in _ENTRY.findall(page):
        name = html.unescape(title).strip()
        if name.startswith(".") or "/" in name or "\\" in name or name in ("", ".."):
            continue  # .DS_Store, and nothing that could escape the output folder
        if kind == "drive/folders":
            files += list_folder(item_id, prefix / name)
        else:
            files.append((prefix / name, item_id))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("ml/data/raw/egdb"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    files = list_folder(ROOT_FOLDER)
    print(f"{len(files)} files")

    def fetch(item: tuple[Path, str]) -> None:
        relative, file_id = item
        target = args.out / relative
        if target.exists() and target.stat().st_size > 0:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _get(DOWNLOAD.format(file_id))
        if data[:15].lower().startswith(b"<!doctype html"):
            raise RuntimeError(f"Drive returned a web page instead of {relative}")
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)

    with ThreadPoolExecutor(args.workers) as pool:
        for done, _ in enumerate(pool.map(fetch, files), start=1):
            if done % 100 == 0:
                print(f"{done}/{len(files)}", flush=True)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
