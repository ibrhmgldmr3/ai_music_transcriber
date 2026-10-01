"""Key changes in songs: one key for the whole song vs. ``music_core.analysis.segment_keys``.

AAM songs change key between sections (three to four times per song in the sample,
mostly between a major key and its parallel minor), and every section's key is
annotated; GuitarSet's comp takes (``--dataset guitarset``) keep one key, so there every
key change found is a false one. The keys are found from the song mode's chords (BTC,
one chord per beat, cached by scripts/benchmark_chords.py) over the found bar lines, as
in the app.
Scores, weighted by time: the exact key, and MIREX's weighted score (a fifth away 0.5,
relative 0.3, parallel 0.2).

    python scripts/benchmark_chords.py --dataset aam      # fills the cache first
    python scripts/benchmark_keys.py --penalty 1 2 4 8
    python scripts/benchmark_keys.py --dataset guitarset
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np

from ml.datasets.aam import songs
from ml.inference.chords import beat_chords
from ml.inference.song import _symbols
from music_core.analysis import Key, KeySpan, estimate_key, segment_keys
from music_core.guitar_chords import QUALITIES
from music_core.notes import Note

FRAME_RATE = 22050 / 2048  # ChordRecognizer
_ROOTS = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9,
          "A#": 10, "B": 11}  # fmt: skip
_FLATS = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10}
GUITARSET = Path("ml/data/raw/guitarset")


def parse_key(name: str) -> Key | None:
    match = re.fullmatch(r"([A-G]#?)(maj|min)", name)
    if not match:
        return None
    return Key(_ROOTS[match.group(1)], "major" if match.group(2) == "maj" else "minor")


def chord_tones(probs: np.ndarray, beats: np.ndarray, duration: float) -> list[Note]:
    segments = beat_chords(probs, FRAME_RATE, beats, duration, 1.0)
    return [
        Note(48 + (symbol.root + interval) % 12, segment.start, segment.end)
        for segment, symbol in _symbols(segments)
        for interval in QUALITIES[symbol.quality][0]
    ]


def weighted(reference: Key, estimate: Key) -> float:
    if reference == estimate:
        return 1.0
    if reference.mode == estimate.mode and (estimate.tonic - reference.tonic) % 12 in (5, 7):
        return 0.5
    relative = (reference.tonic + (9 if reference.mode == "major" else 3)) % 12
    if reference.mode != estimate.mode and estimate.tonic == relative:
        return 0.3
    if reference.mode != estimate.mode and estimate.tonic == reference.tonic:
        return 0.2
    return 0.0


def score(reference: list[KeySpan], estimate: list[KeySpan], end: float) -> tuple[float, float]:
    """Time-weighted exact and MIREX scores over [0, end)."""
    cuts = sorted({0.0, end, *[s.start for s in reference + estimate if 0 < s.start < end]})
    exact = mirex = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = (a + b) / 2
        ref = next((s.key for s in reference if s.start <= mid < s.end), None)
        est = next((s.key for s in estimate if s.start <= mid < s.end), None)
        if ref is None or est is None:
            continue
        exact += (b - a) * (ref == est)
        mirex += (b - a) * weighted(ref, est)
    return exact / end, mirex / end


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--penalty", type=float, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--dataset", choices=["aam", "guitarset"], default="aam")
    parser.add_argument("--cache", type=Path, default=Path("ml/data/interim/chords"))
    args = parser.parse_args(argv)

    results: dict[str, list[tuple[float, float]]] = {"one key": []}
    changes: dict[str, list[int]] = {"reference": [], "one key": []}
    for audio, reference in references(args.dataset):
        cached = args.cache / args.dataset / f"{audio.stem}.npz"
        if not cached.exists():
            continue
        data = np.load(cached)
        duration = float(data["duration"])
        reference = [replace(s, end=min(s.end, duration)) for s in reference]
        tones = chord_tones(data["probs"], data["beats"], duration)
        changes["reference"].append(sum(a.key != b.key for a, b in zip(reference, reference[1:])))
        whole = estimate_key(tones)
        results["one key"].append(score(reference, [KeySpan(0.0, duration, whole)], duration))
        changes["one key"].append(0)
        for penalty in args.penalty:
            name = f"segments, penalty {penalty:g}"
            spans = segment_keys(tones, list(data["downbeats"]), penalty)
            results.setdefault(name, []).append(score(reference, spans, duration))
            changes.setdefault(name, []).append(max(0, len(spans) - 1))

    songs_ = len(results["one key"])
    print(f"{songs_} songs, {np.mean(changes['reference']):.1f} key changes each")
    print(f"{'method':<24} {'exact':>6} {'MIREX':>6} {'changes':>8} {'with a change':>14}")
    for name, rows in results.items():
        exact, mirex = np.mean(rows, axis=0)
        moved = np.mean(np.array(changes[name]) > 0)
        print(
            f"{name:<24} {exact:>6.3f} {mirex:>6.3f} {np.mean(changes[name]):>8.1f} {moved:>14.2f}"
        )


def references(dataset: str) -> Iterator[tuple[Path, list[KeySpan]]]:
    """(audio, keys over time): AAM's sections, or GuitarSet's one key per comp take."""
    if dataset == "aam":
        for song in songs():
            sections = [(t, parse_key(k)) for t, _, k in song.sections]
            yield (
                song.audio,
                [
                    KeySpan(t, sections[i + 1][0] if i + 1 < len(sections) else 1e9, k)
                    for i, (t, k) in enumerate(sections)
                    if k is not None
                ],
            )
        return
    for jams in sorted((GUITARSET / "annotation").glob("*_comp.jams")):
        data = json.loads(jams.read_text(encoding="utf-8"))
        value = next(a for a in data["annotations"] if a["namespace"] == "key_mode")["data"]
        tonic, mode = value[0]["value"].split(":")
        key = Key(_ROOTS.get(tonic, _FLATS.get(tonic)), mode)
        yield GUITARSET / "audio_mono-mic" / f"{jams.stem}_mic.wav", [KeySpan(0.0, 1e9, key)]


if __name__ == "__main__":
    main()
