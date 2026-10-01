"""Compare models on every prepared test set, broken down by recording condition.

    python scripts/benchmark_datasets.py \\
        --model v4v5=ml/checkpoints/guitar_v4/best.pt+ml/checkpoints/guitar_v5/best.pt \\
        --model v8v7=ml/checkpoints/guitar_v8/best.pt+ml/checkpoints/guitar_v7/best.pt

``NAME=NOTES[+TAB]``: the note model and optionally the tab model (as in the API).
Groups: GuitarSet player 05; EGDB unseen clips per amplifier (JCjazz and Plexi never
trained on); Guitar-TECHS unseen player per microphone.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ml.datasets.dataset import read_split
from ml.evaluation.evaluate import evaluate_files
from ml.inference.predict import Predictor

DATA = Path("ml/data")
# (group label, processed dir, split file, suffix of the track id that picks the group)
TEST_SETS = [
    ("GuitarSet (akustik, mikrofon)", "guitarset", "guitarset_full/test", None),
    *[
        (f"EGDB {tone}", "egdb", "egdb/test", f"_{tone}")
        for tone in ("DI", "Marshall", "Ftwin", "Mesa", "JCjazz", "Plexi")
    ],
    # The same unseen clips with their onsets aligned (scripts/align_egdb.py), clips that
    # don't match their audio left out.
    *[
        (f"EGDB hizalı {tone}", "egdb_aligned", "egdb_aligned/test", f"_{tone}")
        for tone in ("DI", "Marshall", "Ftwin", "Mesa", "JCjazz", "Plexi")
    ],
    *[
        (f"Guitar-TECHS {version}", "guitar_techs", "guitar_techs/test", f"_{version}")
        for version in ("di", "amp", "ego", "exo")
    ],
]
METRICS = ("note_f1", "note_offset_f1", "tab_note_accuracy")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", action="append", required=True, metavar="NAME=NOTES[+TAB]")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmark_datasets.json"))
    args = parser.parse_args(argv)

    results: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for spec in args.model:
        name, _, paths = spec.partition("=")
        notes_ckpt, _, tab_ckpt = paths.partition("+")
        predictor = Predictor.from_checkpoint(
            notes_ckpt, device=args.device, tab_checkpoint=tab_ckpt or None
        )
        for label, processed, split, suffix in TEST_SETS:
            split_file = DATA / "splits" / f"{split}.txt"
            if not split_file.exists():
                continue
            names = [n for n in read_split(split_file) if suffix is None or n.endswith(suffix)]
            files = [DATA / "processed" / processed / f"{n}.npz" for n in names]
            if not files:
                continue
            summary, _ = evaluate_files(predictor, files)
            results[label][name] = {m: summary[m] for m in METRICS} | {"tracks": len(files)}

    models = [spec.partition("=")[0] for spec in args.model]
    for metric in METRICS:
        print(f"\n{metric}")
        print(f"{'':34s}" + "".join(f"{m:>12s}" for m in models))
        for label, by_model in results.items():
            row = "".join(f"{by_model.get(m, {}).get(metric, np.nan):12.3f}" for m in models)
            print(f"{label:34s}{row}")
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
