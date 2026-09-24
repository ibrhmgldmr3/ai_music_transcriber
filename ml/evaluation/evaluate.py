"""Evaluate a checkpoint on a dataset split (never augmented).

python scripts/evaluate.py --checkpoint ml/checkpoints/guitar/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ml.config import load_config, num_pitches
from ml.datasets.dataset import read_split
from ml.evaluation.metrics import (
    frame_metrics,
    mean_confidence,
    note_metrics,
    tab_metrics,
    tab_note_accuracy,
)
from ml.inference.predict import Predictor
from ml.preprocessing.annotations import array_to_notes


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def evaluate_files(
    predictor: Predictor, files: Sequence[Path]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Per-track metrics and their mean over tracks."""
    cfg = predictor.cfg
    threshold = cfg["inference"]["frame_threshold"]
    rows: list[dict[str, Any]] = []
    for path in tqdm(files, desc="evaluating"):
        with np.load(path) as data:
            track = {key: data[key] for key in data.files}
        probs = predictor.predict_features(track["features"].astype(np.float32))
        estimated = predictor.decode(probs)
        reference = array_to_notes(track["notes"])

        row: dict[str, Any] = {"track": path.stem, "n_notes": len(estimated)}
        row.update(_prefixed("frame_", frame_metrics(probs["frame"] >= threshold, track["frame"])))
        row.update(_prefixed("note_", note_metrics(reference, estimated)))
        row.update(_prefixed("note_offset_", note_metrics(reference, estimated, offset_ratio=0.2)))
        row["tab_note_accuracy"] = tab_note_accuracy(reference, estimated)["accuracy"]
        row["mean_confidence"] = mean_confidence(estimated) or 0.0
        if "tab" in probs and "tab" in track:
            metrics = tab_metrics(
                probs["tab"].argmax(-1),
                track["tab"],
                tuning=cfg["tab"]["tuning"],
                min_midi=cfg["labels"]["min_midi"],
                n_pitches=num_pitches(cfg),
            )
            row.update(_prefixed("tab_head_", metrics))
        rows.append(row)

    keys = [k for k in rows[0] if k != "track"] if rows else []
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return summary, rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tab-checkpoint", type=Path, help="separate model whose tab head places the notes"
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=Path, help="take data paths from this config instead")
    parser.add_argument("--output", type=Path, help="results JSON (default: next to checkpoint)")
    parser.add_argument("--plot", action="store_true", help="save a piano roll of the first track")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    predictor = Predictor.from_checkpoint(
        args.checkpoint, device=args.device, tab_checkpoint=args.tab_checkpoint
    )
    paths = (load_config(args.config) if args.config else predictor.cfg)["paths"]
    names = read_split(Path(paths["splits_dir"]) / f"{args.split}.txt")
    if args.split == "train":
        names = [n for n in names if "__" not in n]  # skip offline-augmented variants
    files = [Path(paths["processed_dir"]) / f"{name}.npz" for name in names]

    summary, rows = evaluate_files(predictor, files)
    output = args.output or args.checkpoint.with_name(f"eval_{args.split}.json")
    output.write_text(json.dumps({"summary": summary, "tracks": rows}, indent=2), encoding="utf-8")

    for key, value in summary.items():
        print(f"{key:<28} {value:.4f}")
    print(f"\nResults written to {output}")

    if args.plot and files:
        from ml.evaluation.visualization import plot_pianoroll, save_figure

        with np.load(files[0]) as data:
            features, target = data["features"].astype(np.float32), data["frame"]
        probs = predictor.predict_features(features)
        threshold = predictor.cfg["inference"]["frame_threshold"]
        fig = plot_pianoroll(
            probs["frame"] >= threshold,
            target,
            min_midi=predictor.cfg["labels"]["min_midi"],
            frame_rate=predictor.frame_rate,
            title=files[0].stem,
        )
        figure_path = output.with_suffix(".png")
        save_figure(fig, figure_path)
        print(f"Piano roll saved to {figure_path}")


if __name__ == "__main__":
    main()
