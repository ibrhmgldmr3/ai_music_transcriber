"""Tune the note decoding thresholds of a checkpoint on validation data.

A model's onset/frame probabilities shift with its training data, so thresholds that
suit one model can cost another recall. The grid is scored by note F1 averaged over the
validation sources (GuitarSet, EGDB, ...), each counting equally.

    python scripts/tune_thresholds.py --checkpoint ml/checkpoints/guitar_v6/best.pt
    python scripts/tune_thresholds.py --checkpoint ml/checkpoints/guitar_v6/best.pt --write

``--write`` stores the best thresholds in the checkpoint's config (``inference``), which
the Predictor, the API and the evaluation scripts read.
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ml.datasets.dataset import read_split
from ml.evaluation.metrics import note_metrics
from ml.inference.postprocess import decode_notes
from ml.inference.predict import Predictor
from ml.preprocessing.annotations import array_to_notes

ONSET = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
FRAME = [0.3, 0.35, 0.4, 0.45, 0.5]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("ml/data/processed"))
    parser.add_argument("--split", type=Path, default=Path("ml/data/splits/mixed/val.txt"))
    parser.add_argument("--write", action="store_true", help="save the best thresholds")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    inference, labels = predictor.cfg["inference"], predictor.cfg["labels"]
    tracks: list[tuple[str, dict[str, np.ndarray], list]] = []
    for name in tqdm(read_split(args.split), desc="predicting"):
        with np.load(args.processed_dir / f"{name}.npz") as data:
            probs = predictor.predict_features(data["features"].astype(np.float32))
            reference = array_to_notes(data["notes"])
        source = name.split("/", 1)[0] if "/" in name else "all"
        tracks.append((source, probs, reference))

    def score(onset: float, frame: float) -> dict[str, float]:
        by_source: dict[str, list[float]] = defaultdict(list)
        for source, probs, reference in tracks:
            estimated = decode_notes(
                probs["frame"],
                probs.get("onset"),
                probs.get("offset"),
                frame_rate=predictor.frame_rate,
                min_midi=labels["min_midi"],
                onset_threshold=onset,
                frame_threshold=frame,
                offset_threshold=inference.get("offset_threshold", 0.5),
                min_duration=inference["min_note_duration"],
            )
            by_source[source].append(note_metrics(reference, estimated)["f1"])
        return {source: float(np.mean(values)) for source, values in by_source.items()}

    current = (inference["onset_threshold"], inference["frame_threshold"])
    results = {pair: score(*pair) for pair in itertools.product(ONSET, FRAME)}
    results.setdefault(current, score(*current))
    sources = sorted(next(iter(results.values())))
    ranked = sorted(results.items(), key=lambda kv: -np.mean(list(kv[1].values())))

    print(f"{'onset':>6} {'frame':>6} {'mean':>7} " + " ".join(f"{s:>13}" for s in sources))
    for (onset, frame), by_source in ranked[:8] + [(current, results[current])]:
        mark = "  <- current" if (onset, frame) == current else ""
        values = " ".join(f"{by_source[s]:13.3f}" for s in sources)
        print(f"{onset:6.2f} {frame:6.2f} {np.mean(list(by_source.values())):7.3f} {values}{mark}")

    (best_onset, best_frame), _ = ranked[0]
    if args.write:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        checkpoint["config"]["inference"]["onset_threshold"] = best_onset
        checkpoint["config"]["inference"]["frame_threshold"] = best_frame
        torch.save(checkpoint, args.checkpoint)
        print(
            f"\nWrote onset_threshold={best_onset} frame_threshold={best_frame}"
            f" to {args.checkpoint}"
        )


if __name__ == "__main__":
    main()
