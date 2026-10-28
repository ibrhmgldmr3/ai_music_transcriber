"""Playing techniques (ml.inference.techniques) against labeled and pickup-tracked guitar.

``--dataset idmt`` (IDMT-SMT-Guitar, subset 2): notes labeled bend, slide, vibrato,
harmonic, dead note or normal by the player. Bends and vibrato are scored note by note;
a slide is labeled on the note that slides (its landing pitch is no note of its own),
so a detected slide (marked on the note it lands on) is right when its previous note on
the string is a labeled slide (same pitch, onset within 50 ms) going the same way.
``--guitars`` picks guitars: thresholds were chosen on AR and FS, LP is the test.

``--dataset guitarset``: no technique labels, but every string's pitch is followed by
the hexaphonic pickup. The same rules applied to that reference curve give the labels,
so this measures how well the curve is followed in the microphone recording.

Notes: ``--notes reference`` uses the dataset's own notes (bends and vibrato only),
``--notes model`` a transcription (``--checkpoint``, ``--tab-checkpoint``; cached),
matched to the reference by pitch and onset (50 ms). ``--set name=value`` overrides a
``TechniqueSettings`` field.

``--tracker contour`` follows the pitch with the contour model (``--contour-checkpoint``,
``ml/configs/guitar_contour.yaml``) instead of the harmonic sum. On GuitarSet the report
also gives the curve's error against the pickup on the reference notes' frames.

    python scripts/benchmark_techniques.py --dataset idmt --guitars LP --notes model
    python scripts/benchmark_techniques.py --dataset guitarset --players 05
    python scripts/benchmark_techniques.py --dataset guitarset --players 05 --tracker contour
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ml.datasets.idmt import GUITARS, find_takes, load_annotation
from ml.inference.techniques import (
    ContourTracker,
    PitchTracker,
    TechniqueSettings,
    detect_techniques,
    find_bend,
    has_vibrato,
)
from ml.preprocessing.annotations import load_guitarset_contours, load_guitarset_jams
from ml.preprocessing.audio import load_audio
from music_core.notes import Note
from music_core.tab import STANDARD_TUNING

GUITARSET = Path("ml/data/raw/guitarset")
NOTES_CACHE = Path("ml/data/interim/techniques")
SAMPLE_RATE = 44100
EXPRESSION_OF = {"bend": "BE", "vibrato": "VI"}


def reference_curve(
    contour: tuple[np.ndarray, np.ndarray], note: Note, tracker: PitchTracker
) -> np.ndarray:
    """The pickup's pitch on the tracker's frames of ``note`` (NaN where it has none)."""
    s = tracker.s
    frames = np.arange(tracker.frame(note.start + s.attack), tracker.frame(note.end - s.release))
    times, midi = contour
    if len(frames) == 0 or len(times) == 0:
        return np.zeros(0)
    t = frames / tracker.frame_rate
    index = np.clip(np.searchsorted(times, t), 1, len(times) - 1)
    nearest = np.where(np.abs(times[index] - t) < np.abs(times[index - 1] - t), index, index - 1)
    curve = midi[nearest].copy()
    curve[np.abs(times[nearest] - t) > 0.004] = np.nan
    return curve


def make_tracker(y: np.ndarray, s: TechniqueSettings, args: argparse.Namespace) -> PitchTracker:
    if args.tracker == "harmonic":
        return PitchTracker(y, SAMPLE_RATE, s)
    if not hasattr(make_tracker, "predictor"):
        from ml.inference.predict import Predictor

        make_tracker.predictor = Predictor.from_checkpoint(args.contour_checkpoint)
    return ContourTracker.from_audio(y, make_tracker.predictor, s)


def curve_errors(
    notes: Sequence[Note], contours: dict, tracker: PitchTracker, counts: Counter
) -> list[np.ndarray]:
    """Cents between the followed and the pickup's pitch on each note's frames; counts
    the pickup's voiced frames and those the tracker found any pitch on."""
    errors = []
    for note in notes:
        others = [n for n in notes if n is not note and n.end > note.start and n.start < note.end]
        reference = reference_curve(contours.get(note.string, ([], [])), note, tracker)
        estimate = tracker.note_curve(note, others).midi[: len(reference)]
        reference = reference[: len(estimate)]
        voiced = ~np.isnan(reference)
        both = voiced & ~np.isnan(estimate)
        counts["curve_voiced"] += int(voiced.sum())
        counts["curve_found"] += int(both.sum())
        errors.append(100 * np.abs(estimate[both] - reference[both]))
    return errors


def curve_labels(curve: np.ndarray, pitch: int, rate: float, s: TechniqueSettings) -> dict:
    bend = find_bend(curve, pitch, rate, s)
    return {"bend": bend.amount if bend else None, "vibrato": has_vibrato(curve, rate, s)}


def own_note_labels(notes: Sequence[Note], tracker: PitchTracker) -> list[dict]:
    """Bend and vibrato of each note on the curve followed in the recording."""
    result = []
    for note in notes:
        others = [n for n in notes if n is not note and n.end > note.start and n.start < note.end]
        curve = tracker.note_curve(note, others).midi
        result.append(curve_labels(curve, note.pitch, tracker.frame_rate, tracker.s))
    return result


def transcribe(dataset: str, take: str, y: np.ndarray, args: argparse.Namespace) -> list[Note]:
    cache = NOTES_CACHE / dataset / f"{take}.json"
    if cache.exists():
        return [Note.from_dict(n) for n in json.loads(cache.read_text())]
    from ml.inference.predict import Predictor
    from ml.preprocessing.spectrogram import compute_features

    if not hasattr(transcribe, "predictor"):
        transcribe.predictor = Predictor.from_checkpoint(
            args.checkpoint, tab_checkpoint=args.tab_checkpoint
        )
    predictor = transcribe.predictor
    notes = predictor.decode(predictor.predict_features(compute_features(y, predictor.cfg)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([n.to_dict() for n in notes]))
    return notes


def match(reference: Sequence[Note], estimate: Sequence[Note], tolerance: float = 0.05) -> dict:
    """Reference index -> estimate index: same pitch, onsets within ``tolerance``."""
    pairs = sorted(
        (abs(r.start - e.start), i, j)
        for i, r in enumerate(reference)
        for j, e in enumerate(estimate)
        if r.pitch == e.pitch and abs(r.start - e.start) <= tolerance
    )
    used_r, used_e, result = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_r and j not in used_e:
            used_r.add(i)
            used_e.add(j)
            result[i] = j
    return result


def previous_on_string(notes: Sequence[Note]) -> dict[int, int]:
    """Note index -> index of the note before it on its string."""
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch))
    last: dict[int, int] = {}
    result = {}
    for i in order:
        string = notes[i].string
        if string is None:
            continue
        if string in last:
            result[i] = last[string]
        last[string] = i
    return result


def score(counts: Counter, name: str, ref_on: set, est_on: set, hits: set) -> None:
    counts[name, "ref"] += len(ref_on)
    counts[name, "est"] += len(est_on)
    counts[name, "hit"] += len(hits)


def run_idmt(args: argparse.Namespace, s: TechniqueSettings, counts: Counter) -> None:
    for take in tqdm(find_takes(guitars=tuple(args.guitars))):
        y = load_audio(take.audio_path, SAMPLE_RATE)
        labeled = load_annotation(take.annotation_path)
        reference = [n.note for n in labeled]
        if args.notes == "reference":
            estimate = reference
            est_labels = own_note_labels(reference, make_tracker(y, s, args))
            pairs = {i: i for i in range(len(reference))}
        else:
            estimate = detect_techniques(
                transcribe("idmt", take.name, y, args),
                y,
                SAMPLE_RATE,
                s,
                STANDARD_TUNING,
                tracker=make_tracker(y, s, args),
            )
            est_labels = [{"bend": n.bend, "vibrato": n.vibrato} for n in estimate]
            pairs = match(reference, estimate)
            counts["notes_ref"] += len(reference)
            counts["notes_matched"] += len(pairs)

        for name, code in EXPRESSION_OF.items():
            ref_on = {i for i, n in enumerate(labeled) if n.expression == code}
            est_on = {j for j, lab in enumerate(est_labels) if lab[name]}
            score(counts, name, ref_on, est_on, {i for i in ref_on if pairs.get(i) in est_on})
            for i, j in pairs.items():  # what the detector says about each kind of note
                counts["said", name, labeled[i].expression] += bool(est_labels[j][name])
        for i in pairs:
            counts["notes", labeled[i].expression] += 1

        if args.notes == "model":
            slides = {i for i, n in enumerate(labeled) if n.expression == "SL"}
            before = previous_on_string(estimate)
            found = {j for j, n in enumerate(estimate) if n.technique == "slide" and j in before}
            hits, right = set(), set()
            for j in found:
                first, second = estimate[before[j]], estimate[j]
                for i in slides:
                    ref = labeled[i]
                    same_way = np.sign(second.pitch - first.pitch) == np.sign(ref.slide_cents or 0)
                    if ref.note.pitch == first.pitch and abs(ref.note.start - first.start) <= 0.05:
                        if same_way:
                            hits.add(i)
                            right.add(j)
            score(counts, "slide", slides, found, hits)
            counts["slide", "right"] += len(right)
            wide = {i for i in slides if abs(labeled[i].slide_cents or 0) >= 200}
            counts["slide", "wide"] += len(wide)
            counts["slide", "wide_hit"] += len(wide & hits)


def run_guitarset(
    args: argparse.Namespace, s: TechniqueSettings, counts: Counter, errors: list | None = None
) -> None:
    pattern = "*.jams" if args.mode == "all" else f"*_{args.mode}.jams"
    for jams in tqdm(sorted((GUITARSET / "annotation").glob(pattern))):
        if args.players and jams.stem[:2] not in args.players:
            continue
        y = load_audio(GUITARSET / "audio_mono-mic" / f"{jams.stem}_mic.wav", SAMPLE_RATE)
        tracker = make_tracker(y, s, args)
        contours = load_guitarset_contours(jams)
        reference = [n for n in load_guitarset_jams(jams) if n.string is not None]
        if errors is not None:
            errors += curve_errors(reference, contours, tracker, counts)
        ref_labels = [
            curve_labels(
                reference_curve(contours.get(n.string, ([], [])), n, tracker),
                n.pitch,
                tracker.frame_rate,
                s,
            )
            for n in reference
        ]
        if args.notes == "reference":
            est_labels = own_note_labels(reference, tracker)
            pairs = {i: i for i in range(len(reference))}
        else:
            notes = transcribe("guitarset", jams.stem, y, args)
            estimate = detect_techniques(notes, y, SAMPLE_RATE, s, STANDARD_TUNING, tracker=tracker)
            est_labels = [{"bend": n.bend, "vibrato": n.vibrato} for n in estimate]
            pairs = match(reference, estimate)
            counts["notes_ref"] += len(reference)
            counts["notes_matched"] += len(pairs)
        for name in ("bend", "vibrato"):
            ref_on = {i for i, lab in enumerate(ref_labels) if lab[name]}
            est_on = {j for j, lab in enumerate(est_labels) if lab[name]}
            hits = {i for i in ref_on if pairs.get(i) in est_on}
            score(counts, name, ref_on, est_on, hits)
            if name == "bend":
                for i in hits:
                    counts["amount", ref_labels[i]["bend"] == est_labels[pairs[i]]["bend"]] += 1


def settings_from(overrides: Sequence[str]) -> TechniqueSettings:
    s = TechniqueSettings()
    types = {f.name: f.type for f in fields(s)}
    changes = {}
    for item in overrides:
        name, value = item.split("=", 1)
        if name not in types:
            raise SystemExit(f"Unknown setting {name!r}")
        changes[name] = type(getattr(s, name))(float(value))
    return replace(s, **changes)


def report(counts: Counter, dataset: str, notes: str, errors: Sequence | None = None) -> None:
    if errors:
        cents = np.concatenate(errors)
        found = counts["curve_found"] / max(counts["curve_voiced"], 1)
        print(
            f"curve vs pickup: median {np.median(cents):.1f} cents, "
            f"within 25 cents {np.mean(cents <= 25):.3f}, within 50 {np.mean(cents <= 50):.3f}, "
            f"pickup-voiced frames with a pitch {found:.3f}"
        )
    if counts["notes_ref"]:
        matched, total = counts["notes_matched"], counts["notes_ref"]
        print(f"reference notes matched by the transcription: {matched} / {total}")
    names = ["bend", "vibrato"] + (["slide"] if dataset == "idmt" and notes == "model" else [])
    for name in names:
        ref, est, hit = counts[name, "ref"], counts[name, "est"], counts[name, "hit"]
        right = counts[name, "right"] if name == "slide" else hit
        p = right / est if est else 0.0
        r = hit / ref if ref else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{name:8s} reference {ref:5d}  detected {est:5d}  P {p:.3f}  R {r:.3f}  F {f:.3f}")
    if counts["slide", "wide"]:
        wide, hit = counts["slide", "wide"], counts["slide", "wide_hit"]
        print(f"slides of two semitones or more found: {hit} / {wide}")
    amounts = counts["amount", True] + counts["amount", False]
    if amounts:
        right = counts["amount", True] / amounts
        print(f"bend amount (half semitones) right on detected bends: {right:.3f}")
    if dataset == "idmt":
        kinds = sorted({k[1] for k in counts if isinstance(k, tuple) and k[0] == "notes"})
        print("share of matched notes marked, by labeled expression:")
        print("          " + "".join(f"{k:>7s}" for k in kinds))
        for name in EXPRESSION_OF:
            row = "".join(
                f"{counts['said', name, k] / counts['notes', k]:7.3f}"
                if counts["notes", k]
                else "      -"
                for k in kinds
            )
            print(f"{name:10s}{row}")
        print("notes     " + "".join(f"{counts['notes', k]:7d}" for k in kinds))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", choices=("idmt", "guitarset"), default="idmt")
    parser.add_argument("--guitars", nargs="*", default=list(GUITARS), help="IDMT: AR FS LP")
    parser.add_argument("--players", nargs="*", help="GuitarSet: only these players, e.g. 05")
    parser.add_argument("--mode", choices=("solo", "comp", "all"), default="solo")
    parser.add_argument("--notes", choices=("reference", "model"), default="reference")
    parser.add_argument("--checkpoint", default="ml/checkpoints/guitar_v10/best.pt")
    parser.add_argument("--tab-checkpoint", default="ml/checkpoints/guitar_v7/best.pt")
    parser.add_argument("--tracker", choices=("harmonic", "contour"), default="harmonic")
    parser.add_argument("--contour-checkpoint", default="ml/checkpoints/guitar_contour5/best.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    args = parser.parse_args(argv)
    s = settings_from(args.set)

    counts: Counter = Counter()
    errors: list = []
    if args.dataset == "idmt":
        run_idmt(args, s, counts)
    else:
        run_guitarset(args, s, counts, errors)
    report(counts, args.dataset, args.notes, errors)


if __name__ == "__main__":
    main()
