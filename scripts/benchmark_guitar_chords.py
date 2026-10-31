"""Guitar projects' chords (BTC + the transcription's notes) on GuitarSet's accompaniments.

What the app does for a guitar recording: the note model's notes (``--checkpoint``,
``--tab-checkpoint``), the tracked beats and BTC's per-beat chord scores fused in
``music_core.analysis.analyze``. Beats and BTC's probabilities come from the cache that
``scripts/benchmark_chords.py --dataset guitarset`` fills, the note model's probabilities
from ``scripts/benchmark_techniques.py``'s (filled here when missing).

Scored over time with mir_eval against the lead sheet (root, major/minor, sevenths) and
against the chords as played, which name the tensions too (9ths, 11ths, 13ths). A
tension is a ninth (2 semitones above the root), eleventh (5) or thirteenth (9) on top
of a chord with a third (a sixth counts only above a seventh); where the root is right,
each tension the estimate names is scored against the played chord's.

    python scripts/benchmark_guitar_chords.py --players 00 01 02 03
    python scripts/benchmark_guitar_chords.py --players 05
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_chords import guitarset_songs  # noqa: E402

from ml.inference.chords import beat_scores  # noqa: E402
from music_core.analysis import analyze  # noqa: E402

BTC_CACHE = Path("ml/data/interim/chords/guitarset")
NOTES_CACHE = Path("ml/data/interim/techniques/guitarset")
BTC_FRAME_RATE = 22050 / 2048  # ChordRecognizer
STEP = 0.01  # seconds between the frames the tensions are compared on
TENSIONS = {2: "9", 5: "11", 9: "13"}


def model_probs(song: str, audio: Path, args: argparse.Namespace) -> dict[str, np.ndarray]:
    cache = NOTES_CACHE / f"{song}.npz"
    if cache.exists():
        with np.load(cache) as data:
            return dict(data)
    from ml.preprocessing.audio import load_audio
    from ml.preprocessing.spectrogram import compute_features

    predictor = load_predictor(args)
    y = load_audio(audio, predictor.cfg["audio"]["sample_rate"])
    probs = predictor.predict_features(compute_features(y, predictor.cfg))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **probs)
    return probs


def load_predictor(args: argparse.Namespace):
    if not hasattr(load_predictor, "predictor"):
        from ml.inference.predict import Predictor

        load_predictor.predictor = Predictor.from_checkpoint(
            args.checkpoint, tab_checkpoint=args.tab_checkpoint
        )
    return load_predictor.predictor


def tensions(label: str) -> tuple[int | None, set[int]]:
    """Root and tensions (semitones above the root, see the module doc) of a Harte label."""
    import mir_eval

    if label in ("N", "X"):
        return None, set()
    root, bitmap, _ = mir_eval.chord.encode(label, reduce_extended_chords=True)
    present = {i for i in range(12) if bitmap[i]}
    if not present & {3, 4}:
        return int(root), set()  # sus and power chords: no third to add tensions to
    found = {i for i in (2, 5) if i in present}
    if 9 in present and present & {10, 11}:
        found.add(9)
    return int(root), found


def sample(starts: list[float], ends: list[float], labels: list[str], times: np.ndarray):
    index = np.searchsorted(np.asarray(starts), times, side="right") - 1
    out = []
    for t, i in zip(times, index):
        out.append(labels[i] if i >= 0 and t < ends[i] else "N")
    return out


def main(argv: list[str] | None = None) -> None:
    import mir_eval

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--players", nargs="*", help="GuitarSet players, e.g. 05")
    parser.add_argument("--checkpoint", default="ml/checkpoints/guitar_v10/best.pt")
    parser.add_argument("--tab-checkpoint", default="ml/checkpoints/guitar_v7/best.pt")
    args = parser.parse_args(argv)
    players = set(args.players) if args.players else None

    leadsheet = {song: ref for song, _, ref in guitarset_songs(players)}
    sums: Counter = Counter()
    counts: Counter = Counter()
    for song, audio, performed in tqdm(list(guitarset_songs(players, performed=True))):
        btc_file = BTC_CACHE / f"{audio.stem}.npz"
        if not btc_file.exists():
            print(f"{song}: no beats/BTC cache, run benchmark_chords.py --dataset guitarset")
            continue
        with np.load(btc_file) as data:
            btc = {k: data[k] for k in data.files}
        duration = float(btc["duration"])
        notes = load_predictor(args).decode(model_probs(song, audio, args))
        result = analyze(
            notes,
            None,
            beats=btc["beats"],
            downbeats=btc["downbeats"],
            end=duration,
            chord_scores=beat_scores(btc["probs"], BTC_FRAME_RATE, btc["beats"], duration),
        )
        est = [(c.start, c.end, c.harte) for c in result.chords]
        est_iv = np.array([[a, max(b, a + 1e-3)] for a, b, _ in est]) if est else None
        est_labels = [label for _, _, label in est]

        for name, ref in (("leadsheet", leadsheet[song]), ("performed", performed)):
            starts, labels = ref["chords"]
            ends = [*starts[1:], duration]
            ref_iv = np.array([[a, max(b, a + 1e-3)] for a, b in zip(starts, ends)])
            if est_iv is None:
                continue
            scores = mir_eval.chord.evaluate(ref_iv, labels, est_iv, est_labels)
            for metric in ("root", "majmin", "sevenths"):
                sums[name, metric] += scores[metric]
            counts[name] += 1

        starts, labels = performed["chords"]
        times = np.arange(0, duration, STEP)
        ref_frames = sample(list(starts), [*starts[1:], duration], labels, times)
        est_frames = sample([a for a, _, _ in est], [b for _, b, _ in est], est_labels, times)
        for ref_label, est_label in zip(ref_frames, est_frames):
            ref_root, ref_t = tensions(ref_label)
            est_root, est_t = tensions(est_label)
            if ref_root is None:
                continue
            counts["frames"] += 1
            counts["ref tension frames"] += bool(ref_t)
            if est_root != ref_root:
                continue
            counts["root right"] += 1
            for i, name in TENSIONS.items():
                counts[name, "ref"] += i in ref_t
                counts[name, "est"] += i in est_t
                counts[name, "hit"] += i in ref_t and i in est_t
            counts["any", "ref"] += bool(ref_t)
            counts["any", "est"] += bool(est_t)
            counts["any", "hit"] += bool(ref_t) and ref_t == est_t

    for name in ("leadsheet", "performed"):
        if counts[name]:
            row = "  ".join(
                f"{m} {sums[name, m] / counts[name]:.3f}" for m in ("root", "majmin", "sevenths")
            )
            print(f"{name:10s} {counts[name]} takes  {row}")
    frames = max(counts["frames"], 1)
    print(
        f"played chords with a tension: {counts['ref tension frames'] / frames:.3f} of the"
        f" time; root right {counts['root right'] / frames:.3f}"
    )
    for name in [*TENSIONS.values(), "any"]:
        ref, est, hit = counts[name, "ref"], counts[name, "est"], counts[name, "hit"]
        p, r = hit / max(est, 1), hit / max(ref, 1)
        print(
            f"tension {name:4s} (root right) ref {ref * STEP:7.1f} s  est {est * STEP:7.1f} s"
            f"  P {p:.3f} R {r:.3f} F {2 * p * r / max(p + r, 1e-9):.3f}"
        )
    print('("any": the estimate names exactly the played tensions)')


if __name__ == "__main__":
    main()
