"""Robustness benchmark: how transcription holds up on degraded or effected recordings.

Each recording of a split is degraded (phone, noise, big room, overdrive, distortion,
echo) and run through the full pipeline. None of the conditions changes pitch or
timing, so the annotations stay valid. Every (recording, condition) pair uses a fixed
random seed, so different models are compared on identical audio.

python scripts/benchmark_robustness.py --checkpoint ml/checkpoints/guitar/best.pt \
    --tab-checkpoint ml/checkpoints/guitar_tab/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ml.config import load_config
from ml.datasets.dataset import read_split
from ml.datasets.guitarset import find_tracks, load_track
from ml.evaluation.metrics import note_metrics, tab_note_accuracy
from ml.inference.predict import Predictor
from ml.preprocessing.augmentation import apply_variant
from ml.preprocessing.spectrogram import compute_features

CONDITIONS: dict[str, dict[str, Any]] = {
    "clean": {},
    "phone": {"phone": True},
    "noise 10 dB": {"noise_snr_db": [10.0, 10.0]},
    "big room": {"reverb": {"decay_range": [1.2, 1.8], "wet_range": [0.4, 0.5]}},
    "overdrive": {"distortion": "soft"},
    "distortion": {"distortion": "hard"},
    "echo": {"echo": True},
}


def benchmark(
    predictor: Predictor,
    cfg: dict[str, Any],
    track_ids: list[str],
    conditions: dict[str, dict[str, Any]] = CONDITIONS,
    seed: int = 1234,
) -> dict[str, dict[str, float]]:
    """Per condition: mean note F1, tab accuracy (of detected notes) and the share of
    reference notes that are fully right (detected and on the annotated string/fret)."""
    tab = cfg["tab"]
    sample_rate = cfg["audio"]["sample_rate"]
    wanted = set(track_ids)
    tracks = [
        t for t in find_tracks(cfg["paths"]["raw_dir"], cfg["dataset"]["audio_type"])
        if t.track_id in wanted
    ]  # fmt: skip
    if not tracks:
        raise FileNotFoundError("No recordings of the split found; check paths.raw_dir")

    totals = {
        name: {"f1": [], "correct": 0.0, "matched": 0.0, "reference": 0} for name in conditions
    }
    for i, track in enumerate(tqdm(tracks, desc="robustness")):
        clean, reference = load_track(track, cfg)
        placed_reference = sum(n.string is not None for n in reference)
        for j, (name, variant) in enumerate(conditions.items()):
            rng = np.random.default_rng([seed, i, j])
            audio, _ = apply_variant(
                clean, reference, variant, sample_rate, rng, tab["tuning"], tab["num_frets"]
            )
            probs = predictor.predict_features(compute_features(audio, cfg))
            estimated = predictor.decode(probs)
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
    parser = argparse.ArgumentParser(description="Robustness benchmark on degraded audio.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tab-checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--config", type=Path, help="take data paths from this config instead")
    parser.add_argument("--output", type=Path, help="results JSON (default: next to checkpoint)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    predictor = Predictor.from_checkpoint(
        args.checkpoint, device=args.device, tab_checkpoint=args.tab_checkpoint
    )
    cfg = predictor.cfg
    if args.config:
        cfg = {**cfg, "paths": load_config(args.config)["paths"]}
    track_ids = [n for n in read_split(Path(cfg["paths"]["splits_dir"]) / f"{args.split}.txt")]
    results = benchmark(predictor, cfg, [n for n in track_ids if "__" not in n])

    print(f"{'condition':<14} {'note F1':>8} {'tab acc':>8} {'fully correct':>14}")
    for name, m in results.items():
        print(
            f"{name:<14} {m['note_f1']:>8.3f} {m['tab_accuracy']:>8.3f} {m['fully_correct']:>14.3f}"
        )
    output = args.output or args.checkpoint.with_name(f"robustness_{args.split}.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {output}")


if __name__ == "__main__":
    main()
