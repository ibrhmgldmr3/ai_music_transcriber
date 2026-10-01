"""Beat and bar grid of guitar recordings: the notes' fixed-tempo grid vs. beat tracking.

* ``notes``: the editor's grid until now; one tempo from the transcription's onsets
  (``refine_tempo``), its phase and bar lines from the notes (``music_core.analysis``),
  4/4 as the app assumes.
* ``tracked``: Beat This! on the audio (``ml.inference.chords.BeatTracker``), its beats
  and bar lines as they come, so the grid can follow a changing tempo.

GuitarSet was recorded to a click, so a second condition, ``rubato``, time-stretches
every four bars by a random factor (0.88-1.12, drifting like a player without a click)
and warps the reference beats with it. Scores: beat and bar line (downbeat) F-measure
(mir_eval, 70 ms). Beat This! was trained on GuitarSet among others, so its scores on
the steady takes are optimistic; the warped ones less so.

    python scripts/benchmark_grid.py --players 05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.inference.chords import BeatTracker
from ml.inference.predict import Predictor
from ml.preprocessing.audio import load_audio
from music_core.analysis import analyze

GUITARSET = Path("ml/data/raw/guitarset")
SAMPLE_RATE = 44100


def references(players: set[str]) -> list[tuple[str, Path, np.ndarray, np.ndarray]]:
    out = []
    for jams in sorted((GUITARSET / "annotation").glob("*.jams")):
        if jams.stem[:2] not in players:
            continue
        data = json.loads(jams.read_text(encoding="utf-8"))
        beats = next(a for a in data["annotations"] if a["namespace"] == "beat_position")["data"]
        times = np.array([b["time"] for b in beats])
        downs = np.array([b["time"] for b in beats if b["value"]["position"] == 1])
        out.append((jams.stem, GUITARSET / "audio_mono-mic" / f"{jams.stem}_mic.wav", times, downs))
    return out


def warp(
    y: np.ndarray, beats: np.ndarray, downbeats: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-stretch every 4-bar stretch by a drifting factor; the beats move with it."""
    import librosa

    cuts = [0.0, *downbeats[4::4], len(y) / SAMPLE_RATE]
    rate, pieces, new_cuts = 1.0, [], [0.0]
    for a, b in zip(cuts[:-1], cuts[1:]):
        rate = float(np.clip(rate + rng.normal(0, 0.06), 0.88, 1.12))
        chunk = y[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)]
        stretched = librosa.effects.time_stretch(chunk, rate=rate) if len(chunk) > 2048 else chunk
        pieces.append(stretched)
        new_cuts.append(new_cuts[-1] + len(stretched) / SAMPLE_RATE)
    mapping = lambda t: np.interp(t, cuts, new_cuts)  # noqa: E731
    return np.concatenate(pieces), mapping(beats), mapping(downbeats)


def notes_grid(
    predictor: Predictor, y: np.ndarray, duration: float, beats_per_measure: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    result = predictor.transcribe_signal(y)
    analysis = analyze(result.notes, result.tempo, beats_per_measure)
    period = 60.0 / analysis.tempo
    index = np.arange(np.ceil(-analysis.downbeat / period), (duration - analysis.downbeat) / period)
    beats = analysis.downbeat + index * period
    return beats, beats[index % beats_per_measure == 0]


def main(argv: list[str] | None = None) -> None:
    import mir_eval

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--players", nargs="*", default=["05"])
    parser.add_argument("--checkpoint", default="ml/checkpoints/guitar_v9/best.pt")
    parser.add_argument("--tab-checkpoint", default="ml/checkpoints/guitar_v7/best.pt")
    parser.add_argument("--output", type=Path, default=Path("benchmark_grid.json"))
    args = parser.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.checkpoint, tab_checkpoint=args.tab_checkpoint)
    tracker = BeatTracker()
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for song, audio, ref_beats, ref_downs in tqdm(references(set(args.players))):
        y = load_audio(audio, SAMPLE_RATE, mono=True, normalize=True)
        for condition in ("steady", "rubato"):
            if condition == "rubato":
                y_c, beats_c, downs_c = warp(y, ref_beats, ref_downs, rng)
            else:
                y_c, beats_c, downs_c = y, ref_beats, ref_downs
            duration = len(y_c) / SAMPLE_RATE
            estimates = {
                "notes": notes_grid(predictor, y_c, duration),
                "tracked": tracker(y_c, SAMPLE_RATE),
            }
            for method, (beats, downs) in estimates.items():
                rows.append(
                    {
                        "song": song,
                        "style": "solo" if song.endswith("solo") else "comp",
                        "condition": condition,
                        "method": method,
                        "beat_f": mir_eval.beat.f_measure(beats_c, np.asarray(beats)),
                        "downbeat_f": mir_eval.beat.f_measure(downs_c, np.asarray(downs)),
                    }
                )

    print(f"{len(rows) // 4} recordings")
    print(f"{'condition':<8} {'style':<5} {'method':<8} {'beat F':>7} {'bar F':>6}")
    summary = {}
    for condition in ("steady", "rubato"):
        for style in ("comp", "solo"):
            for method in ("notes", "tracked"):
                sel = [
                    r for r in rows
                    if r["condition"] == condition and r["style"] == style and r["method"] == method
                ]  # fmt: skip
                beat = float(np.mean([r["beat_f"] for r in sel]))
                down = float(np.mean([r["downbeat_f"] for r in sel]))
                summary[f"{condition}/{style}/{method}"] = {"beat_f": beat, "downbeat_f": down}
                print(f"{condition:<8} {style:<5} {method:<8} {beat:>7.3f} {down:>6.3f}")
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), "utf-8")


if __name__ == "__main__":
    main()
