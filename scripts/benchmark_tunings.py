"""How models cope with lowered tunings and a capo, simulated on the GuitarSet test split.

Shifting a recording by k semitones is what retuning every string by k (or a capo at
fret k) does to it: the notes move by k while their strings and frets stay the same.
Each condition shifts the audio and the reference pitches, hands the matching open
strings to the model and scores notes and positions like the robustness benchmark.

python scripts/benchmark_tunings.py \
    --model v8=ml/checkpoints/guitar_v8/best.pt+ml/checkpoints/guitar_v7/best.pt \
    --model v9=ml/checkpoints/guitar_v9/best.pt+ml/checkpoints/guitar_v7/best.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.datasets.dataset import read_split
from ml.datasets.guitarset import find_tracks, load_track
from ml.evaluation.metrics import note_metrics, tab_note_accuracy
from ml.inference.predict import Predictor
from ml.preprocessing.augmentation import pitch_shift
from ml.preprocessing.spectrogram import compute_features
from music_core.tab import STANDARD_TUNING

# Semitones every string is moved by.
CONDITIONS = {
    "standard": 0,
    "half step down": -1,
    "whole step down": -2,
    "two steps down": -4,
    "capo 2": 2,
}


def benchmark(predictor: Predictor, track_ids: list[str]) -> dict[str, dict[str, float]]:
    """Per condition: mean note F1, tab accuracy (of detected notes) and the share of
    reference notes that are fully right (detected and on the annotated string/fret)."""
    cfg = predictor.cfg
    sample_rate = cfg["audio"]["sample_rate"]
    wanted = set(track_ids)
    tracks = [
        t for t in find_tracks(cfg["paths"]["raw_dir"], cfg["dataset"]["audio_type"])
        if t.track_id in wanted
    ]  # fmt: skip
    if not tracks:
        raise FileNotFoundError("No recordings of the split found; check paths.raw_dir")

    totals = {
        name: {"f1": [], "correct": 0.0, "matched": 0.0, "reference": 0} for name in CONDITIONS
    }
    for track in tqdm(tracks, desc="tunings"):
        original, original_reference = load_track(track, cfg)
        placed_reference = sum(n.string is not None for n in original_reference)
        for name, shift in CONDITIONS.items():
            audio = pitch_shift(original, sample_rate, shift) if shift else original
            reference = [dataclasses.replace(n, pitch=n.pitch + shift) for n in original_reference]
            tuning = [pitch + shift for pitch in STANDARD_TUNING]
            probs = predictor.predict_features(compute_features(audio, cfg))
            estimated = predictor.decode(probs, tuning)
            accuracy = tab_note_accuracy(reference, estimated)
            t = totals[name]
            t["f1"].append(note_metrics(reference, estimated)["f1"])
            t["correct"] += accuracy["accuracy"] * accuracy["matched"]
            t["matched"] += accuracy["matched"]
            t["reference"] += placed_reference

    return {
        name: {
            "note_f1": float(np.mean(t["f1"])),
            "tab_accuracy": t["correct"] / t["matched"] if t["matched"] else 0.0,
            "fully_correct": t["correct"] / t["reference"] if t["reference"] else 0.0,
        }
        for name, t in totals.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Transcription in other tunings and with a capo.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=NOTES[+TAB]",
        help="a notes checkpoint, optionally with a tab checkpoint; repeat to compare",
    )
    parser.add_argument(
        "--split", type=Path, default=Path("ml/data/splits/guitarset_full/test.txt")
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_tunings.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    track_ids = [n for n in read_split(args.split) if "__" not in n]
    results: dict[str, dict[str, dict[str, float]]] = {}
    for spec in args.model:
        name, _, paths = spec.partition("=")
        notes, _, tab = paths.partition("+")
        predictor = Predictor.from_checkpoint(
            Path(notes), device=args.device, tab_checkpoint=Path(tab) if tab else None
        )
        results[name] = benchmark(predictor, track_ids)

    for name, by_condition in results.items():
        print(f"\n{name}")
        print(f"{'condition':<16} {'note F1':>8} {'tab acc':>8} {'fully correct':>14}")
        for condition, m in by_condition.items():
            print(
                f"{condition:<16} {m['note_f1']:>8.3f} {m['tab_accuracy']:>8.3f}"
                f" {m['fully_correct']:>14.3f}"
            )
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
