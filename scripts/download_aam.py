"""Download a sample of the AAM dataset (Artificial Audio Multitracks, CC BY 4.0).

AAM (Ostermann et al., 2023) has 3000 generated band songs whose annotations (chords,
beats, bars, key, tempo) are exact. The audio comes in 14.7 GB zips of 1000 mixes; this
reads single mixes out of the remote zip with HTTP range requests, so an evaluation
sample costs about 15 MB per song instead of the whole archive.

    https://zenodo.org/records/5794629

python scripts/download_aam.py --count 70
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/5794629/files"
ANNOTATIONS = "0001-1000-annotations-v1.1.0.zip"
MIXES = "0001-1000-audio-mixes.zip"


class RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a URL through HTTP range requests (1 MB blocks)."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos = url, block, 0
        head = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}))
        self.size = int(head.headers["Content-Range"].rsplit("/", 1)[1])
        self.cache: dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * self.block
            end = min(self.size, start + self.block) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            self.cache = {index: urllib.request.urlopen(request).read()}  # keep one block
        return self.cache[index]

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            index, offset = divmod(self.pos, self.block)
            chunk = self._block(index)[offset : offset + end - self.pos]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download a sample of AAM mixes.")
    parser.add_argument("--out", type=Path, default=Path("ml/data/raw/aam"))
    parser.add_argument("--count", type=int, default=70, help="songs, spread over 0001-1000")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    annotations = args.out / ANNOTATIONS
    if not annotations.exists():
        urllib.request.urlretrieve(f"{RECORD}/{ANNOTATIONS}/content", annotations)

    mixes = zipfile.ZipFile(RemoteFile(f"{RECORD}/{MIXES}/content"))
    members = sorted(m for m in mixes.namelist() if not m.endswith("/"))
    step = max(1, len(members) // args.count)
    audio = args.out / "mixes"
    audio.mkdir(exist_ok=True)
    for member in members[::step][: args.count]:
        target = audio / Path(member).name
        if target.exists():
            continue
        target.write_bytes(mixes.read(member))
        print(target.name, f"{target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
