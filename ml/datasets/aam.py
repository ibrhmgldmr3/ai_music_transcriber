"""AAM (Ostermann et al., 2023): generated band songs with exact chord, beat, bar, key,
tempo and per-instrument note annotations (CC BY 4.0).

``scripts/download_aam.py`` fetches a sample of the mixes into ml/data/raw/aam/mixes; the
annotations come in one archive. Each song has:

* ``<id>_beatinfo.arff``: every beat's time, bar number, beat in the bar and chord
  (major/minor only: "Fmaj", "A#min", "N.C.");
* ``<id>_segments.arff``: sections with their tempo, key ("Fmaj") and the instrument
  each generator (melody, chords, bass line, drums) plays;
* ``<id>_onsets.arff``: at every onset of any instrument, the MIDI pitches each
  instrument is sounding ("[+41]" a note starting, "[41]" one held).
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AAM = Path("ml/data/raw/aam")
ANNOTATIONS = "0001-1000-annotations-v1.1.0.zip"


@dataclass
class AAMSong:
    song: str  # "0001"
    audio: Path
    beats: list[float]
    downbeats: list[float]
    chord_times: list[float]
    chords: list[str]  # Harte, per beat: "F:maj", "A#:min", "N"
    sections: list[tuple[float, float, str]] = field(default_factory=list)  # start, BPM, key
    bass: list[tuple[float, int | None]] = field(default_factory=list)  # time, lowest bass pitch

    @property
    def duration(self) -> float:
        return (
            float(self.beats[-1] + (self.beats[-1] - self.beats[-2]))
            if len(self.beats) > 1
            else 0.0
        )

    def bass_at(self, a: float, b: float) -> int | None:
        """The bass line's pitch sounding longest in [a, b), or None if it rests."""
        times = [t for t, _ in self.bass]
        if not times:
            return None
        weights: dict[int, float] = {}
        k = max(0, int(np.searchsorted(times, a, side="right")) - 1)
        while k < len(self.bass) and self.bass[k][0] < b:
            start = max(a, self.bass[k][0])
            end = min(b, self.bass[k + 1][0] if k + 1 < len(self.bass) else b)
            pitch = self.bass[k][1]
            if pitch is not None and end > start:
                weights[pitch] = weights.get(pitch, 0.0) + end - start
            k += 1
        return max(weights, key=weights.get) if weights else None


def harte(name: str) -> str:
    """AAM's "A#maj" / "Amin" / "N.C." -> "A#:maj" / "A:min" / "N"."""
    match = re.fullmatch(r"([A-G]#?)(maj|min)", name)
    return f"{match.group(1)}:{match.group(2)}" if match else "N"


def _rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if line[:1].isdigit():
            rows.append([cell.strip().strip("'") for cell in re.split(r",(?![^\[]*\])", line)])
    return rows


def songs(root: Path = AAM) -> Iterator[AAMSong]:
    """The downloaded songs; ones whose beat times go backwards are skipped."""
    labels = zipfile.ZipFile(root / ANNOTATIONS)
    for audio in sorted((root / "mixes").glob("*_mix.flac")):
        song = audio.name[:4]
        rows = _rows(labels.read(f"{song}_beatinfo.arff").decode("utf-8"))
        times = [float(r[0]) for r in rows]
        if any(b <= a for a, b in zip(times, times[1:])):
            continue
        sections = [
            (float(r[0]), float(r[2]), r[3])
            for r in _rows(labels.read(f"{song}_segments.arff").decode("utf-8"))
        ]
        yield AAMSong(
            song=song,
            audio=audio,
            beats=times,
            downbeats=[float(r[0]) for r in rows if r[2] == "1"],
            chord_times=times,
            chords=[harte(r[3]) for r in rows],
            sections=sections,
            bass=_bass_line(labels, song),
        )


def _bass_line(labels: zipfile.ZipFile, song: str) -> list[tuple[float, int | None]]:
    """(time, lowest pitch the bass-line instrument sounds from then on) at every onset."""
    segments = _rows(labels.read(f"{song}_segments.arff").decode("utf-8"))
    text = labels.read(f"{song}_onsets.arff").decode("utf-8")
    names = [
        line.split("'")[1].removeprefix("Onset events of ")
        for line in text.splitlines()
        if line.startswith("@ATTRIBUTE")
    ][1:]
    # Which instrument plays the bass line in each section.
    bass_by_section = []
    for row in segments:
        instruments = row[4].strip("[]").split(",")
        generators = row[5].strip("[]").split(",")
        player = next((i for i, g in zip(instruments, generators) if g == "BassLine"), None)
        bass_by_section.append((float(row[0]), player))
    out: list[tuple[float, int | None]] = []
    for row in _rows(text):
        time = float(row[0])
        player = next((p for start, p in reversed(bass_by_section) if start <= time + 1e-6), None)
        pitches = []
        if player in names:
            cell = row[1 + names.index(player)]
            pitches = [int(p.lstrip("+")) for p in cell.strip("[]").split(",") if p.strip()]
        out.append((time, min(pitches) if pitches else None))
    return out
