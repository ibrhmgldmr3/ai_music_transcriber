"""Strumming patterns (music_core.rhythm) against GuitarSet's strummed accompaniments.

GuitarSet annotates every string, so its strums and their directions are known
(ml/datasets/strums.py); takes with at least 20 strums count as strummed. Scores:

* direction: the pendulum rule (even eighth or sixteenth down, odd up) on the reference
  strums and beats;
* strums found (onset F-measure, 50 ms) and their sixteenths (F-measure of the struck
  sixteenths, on the tracked beats of scripts/benchmark_chords.py's cache), from the
  audio's onsets (``ml.inference.rhythm.audio_strums``) and, with ``--notes``, from the
  chords of a transcription (cached note files of ``--notes-cache``);
* pattern: the song's pattern (the sixteenths struck in most bars) against the
  reference's: F-measure of its struck sixteenths, and how often it is exactly right.

    python scripts/benchmark_strums.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.datasets.strums import beats_of, strums
from ml.inference.rhythm import audio_strums
from music_core.notes import Note
from music_core.rhythm import strum_rhythm, strums_from_notes
from music_core.timing import BeatGrid, downbeat_phase, tracked_meter

GUITARSET = Path("ml/data/raw/guitarset")
CHORD_CACHE = Path("ml/data/interim/chords/guitarset")


def sixteenths(times: list[float], grid: BeatGrid) -> set[int]:
    return {int(k) for k in np.round(np.asarray(grid.position(np.array(times))) * 4)}


def f1(reference: set, estimate: set) -> float:
    if not reference and not estimate:
        return 1.0
    hits = len(reference & estimate)
    return 2 * hits / (len(reference) + len(estimate)) if hits else 0.0


def main(argv: list[str] | None = None) -> None:
    import librosa
    import mir_eval

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--players", nargs="*", help="only these players, e.g. 05")
    parser.add_argument("--notes-cache", type=Path, help="<take>.json transcriptions")
    args = parser.parse_args(argv)

    direction = Counter()
    found: dict[str, list[tuple[float, float, bool]]] = {"audio": [], "notes": []}
    takes = 0
    for jams in tqdm(sorted((GUITARSET / "annotation").glob("*_comp.jams"))):
        if args.players and jams.stem[:2] not in args.players:
            continue
        reference = strums(jams)
        if len(reference) < 20:
            continue
        takes += 1
        style = jams.stem.split("_")[1].split("-")[0][:-1]
        ref_beats = beats_of(jams)
        ref_grid = BeatGrid.tracked(ref_beats, 4, 0, 0.0, ref_beats[-1] + 2)
        ref_pairs = [(s.time, float(s.strings)) for s in reference]
        rhythm = strum_rhythm(ref_pairs, ref_grid)
        for s in reference:
            if s.down is not None:
                slot = round(float(ref_grid.position(s.time)) * rhythm.per_beat)
                direction[(style, "n")] += 1
                direction[(style, "ok")] += rhythm.down(slot) == s.down

        cache = CHORD_CACHE / f"{jams.stem}_mic.npz"
        if not cache.exists():
            continue
        tracked = np.load(cache)
        meter = tracked_meter(tracked["beats"], tracked["downbeats"])
        phase = downbeat_phase(tracked["beats"], tracked["downbeats"], meter)
        grid = BeatGrid.tracked(tracked["beats"], meter, phase, 0.0, float(tracked["duration"]))
        ref_times = [s.time for s in reference]
        ref_rhythm = strum_rhythm(ref_pairs, grid)
        estimates = {}
        y, sr = librosa.load(GUITARSET / "audio_mono-mic" / f"{jams.stem}_mic.wav", sr=22050)
        estimates["audio"] = audio_strums(y, sr)
        if args.notes_cache and (args.notes_cache / f"{jams.stem}.json").exists():
            data = json.loads((args.notes_cache / f"{jams.stem}.json").read_text())
            estimates["notes"] = strums_from_notes([Note.from_dict(n) for n in data["notes"]])
        for name, pairs in estimates.items():
            est_rhythm = strum_rhythm(pairs, grid)
            if est_rhythm is None:
                found[name].append((0.0, 0.0, 0.0, False))
                continue
            strong = [t for (t, s) in pairs if s >= 0.3 * np.median([s for _, s in pairs])]
            onset = mir_eval.onset.f_measure(np.array(ref_times), np.array(strong), window=0.05)[0]
            slots = f1(sixteenths(ref_times, grid), sixteenths(strong, grid))
            est, ref = _at16(est_rhythm), _at16(ref_rhythm)
            pattern_f = f1({i for i, h in enumerate(ref) if h}, {i for i, h in enumerate(est) if h})
            found[name].append((onset, slots, pattern_f, est == ref))

    n = sum(v for k, v in direction.items() if k[1] == "n")
    ok = sum(v for k, v in direction.items() if k[1] == "ok")
    print(f"{takes} strummed takes")
    styles = sorted({k[0] for k in direction})
    print("pendulum rule: " + f"{ok / n:.3f} of {n} strums; " + ", ".join(
        f"{s} {direction[(s, 'ok')] / direction[(s, 'n')]:.2f}" for s in styles
    ))  # fmt: skip
    for name, rows in found.items():
        if rows:
            onset, slots, pattern_f, same = np.mean(np.array(rows, dtype=float), axis=0)
            print(
                f"{name:>6}: strums F {onset:.3f}, sixteenths F {slots:.3f}, "
                f"pattern slots F {pattern_f:.3f}, pattern exact {same:.2f}"
            )


def _at16(rhythm) -> tuple[bool, ...]:
    """The main pattern on sixteenths, so eighth and sixteenth patterns compare."""
    if rhythm.per_beat == 4:
        return rhythm.pattern
    out = []
    for hit in rhythm.pattern:
        out += [hit, False]
    return tuple(out)


if __name__ == "__main__":
    main()
