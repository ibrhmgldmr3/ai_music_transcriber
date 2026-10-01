"""Align EGDB's note timing to the audio, for training the note model on EGDB.

EGDB's labels mostly carry the tablature's timing (ml/datasets/egdb.py). A note model
trained on precisely timed labels (``--checkpoint``, which never saw EGDB's note labels)
transcribes every clip's direct input; each labeled onset moves to the model's onset of
its pitch within 60 ms (``egdb.align_notes``), and the labels are written to
ml/data/interim/egdb_aligned/<clip>.npz. Clips whose labels match the transcription
poorly (onset F1 below ``--min-f1`` at 50 ms) don't match their audio and are left out;
kept.txt lists the rest.

    python scripts/align_egdb.py --checkpoint ml/checkpoints/guitar_v9/best.pt
    python scripts/prepare_dataset.py --dataset egdb_aligned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.datasets import egdb
from ml.inference.predict import Predictor
from ml.preprocessing.annotations import load_string_midi, notes_to_array
from music_core.notes import Note
from music_core.tab import STANDARD_TUNING

RAW = Path("ml/data/raw/egdb")


def onset_f1(reference: list[Note], estimate: list[Note], tolerance: float = 0.05) -> float:
    import mir_eval

    def arrays(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
        intervals = np.array([[n.start, max(n.end, n.start + 1e-3)] for n in notes]).reshape(-1, 2)
        return intervals, mir_eval.util.midi_to_hz(np.array([n.pitch for n in notes], float))

    if not reference or not estimate:
        return 0.0
    ref, est = arrays(reference), arrays(estimate)
    return mir_eval.transcription.precision_recall_f1_overlap(
        ref[0], ref[1], est[0], est[1], onset_tolerance=tolerance, offset_ratio=None
    )[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--min-f1", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=egdb.ALIGNED_DIR)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    labels = sorted(RAW.glob("audio_label/*.midi"), key=lambda p: int(p.stem))
    report: dict[int, dict[str, float]] = {}
    shifts: list[float] = []
    for label in tqdm(labels, desc="aligning"):
        clip = int(label.stem)
        audio = RAW / "audio_DI" / f"{clip}.wav"
        if not audio.exists():
            continue
        notes = load_string_midi(label, STANDARD_TUNING, string_from="channel")
        transcribed = predictor.transcribe(audio, estimate_tempo=False).notes
        aligned, moved = egdb.align_notes(notes, transcribed)
        np.savez_compressed(args.output / f"{clip}.npz", notes=notes_to_array(aligned))
        shifts += [a.start - n.start for a, n in zip(aligned, notes) if a.start != n.start]
        report[clip] = {
            "notes": len(notes),
            "moved": moved,
            "f1": onset_f1(notes, transcribed),
            "f1_25ms": onset_f1(notes, transcribed, 0.025),
        }

    kept = [c for c, row in report.items() if row["notes"] >= 10 and row["f1"] >= args.min_f1]
    (args.output / "kept.txt").write_text("\n".join(map(str, kept)) + "\n", encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    shifts_ms = 1000 * np.abs(shifts)
    print(f"{len(report)} clips, {len(kept)} kept; left out: {sorted(set(report) - set(kept))}")
    print(
        f"labels moved to a transcribed onset: {np.mean([r['moved'] for r in report.values()]):.2f}"
    )
    print(f"moves: median {np.median(shifts_ms):.0f} ms, > 25 ms {np.mean(shifts_ms > 25):.2f}")
    print(
        "onset F1 of the original labels (kept clips): "
        f"{np.mean([report[c]['f1'] for c in kept]):.3f} at 50 ms, "
        f"{np.mean([report[c]['f1_25ms'] for c in kept]):.3f} at 25 ms"
    )


if __name__ == "__main__":
    main()
