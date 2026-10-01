"""Chord, beat and bar recognition (ml.inference.chords) against annotated recordings.

* ``aam``: generated band songs from AAM (scripts/download_aam.py) with exact chord (major /
  minor), beat and bar annotations.
* ``guitarset``: the accompaniment ("comp") takes of GuitarSet, real acoustic guitar,
  scored against the lead-sheet chords (with sevenths) and the performed beats.

Chords are scored with mir_eval over time (root, major/minor, major/minor with the bass,
sevenths); beats and bar lines (downbeats) with the beat F-measure (70 ms). Model outputs
are cached, so decoding settings can be compared quickly. ``--bass`` adds slash chords
from the mix's bass as song mode writes them (``ml.inference.song``'s thresholds; song
mode does this only when the song has a bass, which GuitarSet's takes don't).

python scripts/benchmark_chords.py --dataset aam
python scripts/benchmark_chords.py --dataset guitarset --penalty 0 1 2 4
python scripts/benchmark_chords.py --dataset guitarset --reference performed --bass
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.inference.chords import (
    BeatTracker,
    ChordRecognizer,
    add_bass,
    beat_bass,
    beat_chords,
    frame_chords,
)
from ml.inference.song import SLASH_BEATS, SLASH_SHARE

AAM = Path("ml/data/raw/aam")
GUITARSET = Path("ml/data/raw/guitarset")


def aam_songs() -> Iterator[tuple[str, Path, dict]]:
    """(id, audio, reference) for the downloaded AAM mixes."""
    labels = zipfile.ZipFile(AAM / "0001-1000-annotations-v1.1.0.zip")
    for audio in sorted((AAM / "mixes").glob("*_mix.flac")):
        song = audio.name[:4]
        text = labels.read(f"{song}_beatinfo.arff").decode("utf-8")
        rows = [line.split(",") for line in text.splitlines() if line[:1].isdigit()]
        times = [float(r[0]) for r in rows]
        if any(b <= a for a, b in zip(times, times[1:])):
            print(f"{song}: beat times go backwards in the annotation, skipped")
            continue
        chords = [_harte(r[3].strip().strip("'")) for r in rows]
        downbeats = [float(r[0]) for r in rows if r[2].strip() == "1"]
        yield song, audio, {"beats": times, "downbeats": downbeats, "chords": (times, chords)}


def guitarset_songs(
    players: set[str] | None = None, performed: bool = False
) -> Iterator[tuple[str, Path, dict]]:
    """(id, audio, reference) for the GuitarSet comp takes: the lead-sheet chords, or
    with ``performed`` the chords as played (with inversions: "C:maj/5")."""
    for jams in sorted((GUITARSET / "annotation").glob("*_comp.jams")):
        song = jams.stem
        if players and song[:2] not in players:
            continue
        data = json.loads(jams.read_text(encoding="utf-8"))
        chords = [a for a in data["annotations"] if a["namespace"] == "chord"][int(performed)][
            "data"
        ]
        beats = next(a for a in data["annotations"] if a["namespace"] == "beat_position")["data"]
        audio = GUITARSET / "audio_mono-mic" / f"{song}_mic.wav"
        yield (
            song,
            audio,
            {
                "beats": [b["time"] for b in beats],
                "downbeats": [b["time"] for b in beats if b["value"]["position"] == 1],
                "chords": ([c["time"] for c in chords], [c["value"] for c in chords]),
            },
        )


def _harte(name: str) -> str:
    """AAM's "A#maj" / "Amin" / "N.C." -> "A#:maj" / "A:min" / "N"."""
    match = re.fullmatch(r"([A-G]#?)(maj|min)", name)
    return f"{match.group(1)}:{match.group(2)}" if match else "N"


def analyze(
    audio: Path, cache: Path, chords: ChordRecognizer, beats: BeatTracker, bass: bool = False
) -> dict:
    """Chord probabilities, beats and bar lines of a recording, and with ``bass`` the
    bass note of every beat (cached)."""
    import librosa

    cached = cache / f"{audio.stem}.npz"
    result: dict = {}
    if cached.exists():
        data = np.load(cached)
        result = {k: data[k] for k in data.files} | {"duration": float(data["duration"])}
    if result and (not bass or "bass_pc" in result):
        return result
    y, sr = librosa.load(audio, sr=44100, mono=True)
    if not result:
        found, downs = beats(y, sr)
        result = {
            "probs": chords.probabilities(y, sr),
            "beats": found,
            "downbeats": downs,
            "duration": len(y) / sr,
        }
    if bass:
        notes = beat_bass(y, sr, result["beats"], result["duration"])
        result["bass_pc"] = np.array([-1 if pc is None else pc for pc, _ in notes])
        result["bass_share"] = np.array([share for _, share in notes])
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cached, **result)
    return result


def score(
    reference: dict, result: dict, frame_rate: float, penalty: float | None, bass: bool = False
) -> dict:
    import mir_eval

    duration = result["duration"]
    if penalty is None:
        segments = frame_chords(result["probs"], frame_rate)
    else:
        segments = beat_chords(result["probs"], frame_rate, result["beats"], duration, penalty)
        if bass:
            notes = [
                (None if pc < 0 else int(pc), float(share))
                for pc, share in zip(result["bass_pc"], result["bass_share"])
            ]
            segments = add_bass(
                segments,
                notes,
                result["beats"],
                duration,
                min_beats=SLASH_BEATS,
                min_share=SLASH_SHARE,
            )
    starts, labels = reference["chords"]
    ends = [*starts[1:], duration]
    ref = np.array([[a, max(b, a + 1e-3)] for a, b in zip(starts, ends)])
    est = np.array([[s.start, s.end] for s in segments])
    chord = mir_eval.chord.evaluate(ref, labels, est, [s.label for s in segments])
    return {
        "root": chord["root"],
        "majmin": chord["majmin"],
        "majmin_inv": chord["majmin_inv"],
        "sevenths": chord["sevenths"],
        "beat_f": mir_eval.beat.f_measure(np.array(reference["beats"]), result["beats"]),
        "downbeat_f": mir_eval.beat.f_measure(
            np.array(reference["downbeats"]), result["downbeats"]
        ),
        "changes_per_minute": 60 * (len(segments) - 1) / duration,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chord, beat and bar recognition benchmark.")
    parser.add_argument("--dataset", choices=["aam", "guitarset"], default="aam")
    parser.add_argument("--players", nargs="*", help="GuitarSet players, e.g. 05")
    parser.add_argument(
        "--penalty",
        type=float,
        nargs="*",
        default=[1.0],
        help="chord change penalties to compare (beat decoding); frames are always shown",
    )
    parser.add_argument(
        "--reference",
        choices=["leadsheet", "performed"],
        default="leadsheet",
        help="GuitarSet: the lead sheet's chords or the chords as played (with inversions)",
    )
    parser.add_argument(
        "--bass", action="store_true", help="also with slash chords from the bass (add_bass)"
    )
    parser.add_argument("--cache", type=Path, default=Path("ml/data/interim/chords"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_chords.json"))
    args = parser.parse_args(argv)

    songs = (
        list(aam_songs())
        if args.dataset == "aam"
        else list(
            guitarset_songs(
                set(args.players) if args.players else None, args.reference == "performed"
            )
        )
    )
    chords, beats = ChordRecognizer(), BeatTracker()
    cache = args.cache / args.dataset
    results = [
        (reference, analyze(audio, cache, chords, beats, args.bass))
        for _, audio, reference in tqdm(songs, desc=args.dataset)
    ]

    table: dict[str, dict[str, float]] = {}
    for penalty in [None, *args.penalty]:
        for bass in (False, True) if args.bass and penalty is not None else (False,):
            name = "frames" if penalty is None else f"beats, penalty {penalty:g}"
            name += " + bass" if bass else ""
            rows = [score(ref, res, chords.frame_rate, penalty, bass) for ref, res in results]
            table[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"{len(results)} recordings")
    print(
        f"{'decoding':<29} {'root':>6} {'majmin':>7} {'+inv':>6} {'7ths':>6}"
        f" {'beat F':>7} {'bar F':>6} {'chg/min':>8}"
    )
    for name, m in table.items():
        print(
            f"{name:<29} {m['root']:>6.3f} {m['majmin']:>7.3f} {m['majmin_inv']:>6.3f}"
            f" {m['sevenths']:>6.3f} {m['beat_f']:>7.3f} {m['downbeat_f']:>6.3f}"
            f" {m['changes_per_minute']:>8.1f}"
        )
    args.output.write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
