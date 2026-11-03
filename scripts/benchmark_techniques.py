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
``--notes model`` a transcription (``--checkpoint``, ``--tab-checkpoint``; the model's
probabilities are cached, the decoding is not), matched to the reference by pitch and
onset (50 ms). ``--set name=value`` overrides a ``TechniqueSettings`` field,
``--inference name=json`` a decoding setting of the note model (e.g.
``onset_threshold=0.4``).

``--tracker contour`` follows the pitch with the contour model (``--contour-checkpoint``,
``ml/configs/guitar_contour.yaml``) instead of the harmonic sum. On GuitarSet the report
also gives the curve's error against the pickup on the reference notes' frames.

``--classifier`` decides bends and vibrato with a learned model
(``ml.inference.technique_classifier``) instead of the rules. ``--dump`` (IDMT) writes
every note's bend and vibrato features with its label, to train one with
``scripts/train_technique_classifier.py``.

    python scripts/benchmark_techniques.py --dataset idmt --guitars LP --notes model
    python scripts/benchmark_techniques.py --dataset idmt --guitars AR FS --notes model \
        --dump ml/data/interim/technique_features/arfs_model.npz
    python scripts/benchmark_techniques.py --dataset idmt --guitars LP --notes model \
        --classifier ml/checkpoints/techniques/classifier.json
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

from ml.datasets.idmt import GUITARS, IdmtNote, find_takes, load_annotation
from ml.inference.technique_classifier import (
    BEND_FEATURES,
    VIBRATO_FEATURES,
    TechniqueClassifier,
    bend_candidate,
    vibrato_features,
)
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


def curve_labels(
    curve: np.ndarray,
    pitch: int,
    rate: float,
    s: TechniqueSettings,
    classifier: TechniqueClassifier | None = None,
) -> dict:
    if classifier is not None:
        bend = classifier.bend(curve, pitch, rate, s)
        return {
            "bend": bend.amount if bend else None,
            "vibrato": classifier.vibrato(curve, rate, s),
        }
    bend = find_bend(curve, pitch, rate, s)
    return {"bend": bend.amount if bend else None, "vibrato": has_vibrato(curve, rate, s)}


def own_note_labels(
    notes: Sequence[Note],
    tracker: PitchTracker,
    classifier: TechniqueClassifier | None = None,
    trace: list | None = None,
) -> list[dict]:
    """Bend and vibrato of each note on the curve followed in the recording."""
    result = []
    for i, note in enumerate(notes):
        others = [n for n in notes if n is not note and n.end > note.start and n.start < note.end]
        curve = tracker.note_curve(note, others).midi
        result.append(curve_labels(curve, note.pitch, tracker.frame_rate, tracker.s, classifier))
        if trace is not None:
            trace.append(
                {
                    "index": i,
                    "bend": [(curve, note.pitch)],
                    "vibrato": curve,
                    "frame_rate": tracker.frame_rate,
                }
            )
    return result


class FeatureDump:
    """Bend and vibrato features of every note with its IDMT label, for
    ``scripts/train_technique_classifier.py``. Bend rows are the loose rule's candidates
    (a note cut short may give two: ``note`` tells them apart); notes without one can't
    be bent by a classifier but count in ``positives``."""

    def __init__(self) -> None:
        self.rows: dict[str, list] = {"vibrato": [], "bend": []}
        self.positives: Counter = Counter()

    def add(
        self,
        take: str,
        trace: Sequence[dict],
        labels: Sequence[str],
        reference: Sequence[IdmtNote],
        s: TechniqueSettings,
    ) -> None:
        """``labels``: the expression each traced note is taken to have."""
        for code in ("BE", "VI"):
            self.positives[code] += sum(n.expression == code for n in reference)
        for entry in trace:
            i, rate = entry["index"], entry["frame_rate"]
            label = labels[i]
            features = vibrato_features(entry["vibrato"], rate, s)
            if features is not None:
                self.rows["vibrato"].append((take, i, label == "VI", features))
            for curve, pitch in entry["bend"]:
                candidate = bend_candidate(curve, pitch, rate, s)
                if candidate is not None:
                    self.rows["bend"].append((take, i, label == "BE", candidate[1]))

    def save(self, path: Path, info: dict) -> None:
        arrays = {}
        for kind, names in (("vibrato", VIBRATO_FEATURES), ("bend", BEND_FEATURES)):
            rows = self.rows[kind]
            arrays[f"{kind}_take"] = np.array([r[0] for r in rows], dtype=str)
            arrays[f"{kind}_note"] = np.array([r[1] for r in rows], dtype=int)
            arrays[f"{kind}_label"] = np.array([r[2] for r in rows], dtype=bool)
            arrays[f"{kind}_X"] = (
                np.stack([r[3] for r in rows]) if rows else np.zeros((0, len(names)))
            )
            arrays[f"{kind}_features"] = np.array(names, dtype=str)
        arrays["positives_bend"] = np.array(self.positives["BE"])
        arrays["positives_vibrato"] = np.array(self.positives["VI"])
        arrays["info"] = np.array(json.dumps(info))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        print(
            f"wrote {path}: {len(self.rows['vibrato'])} vibrato rows, "
            f"{len(self.rows['bend'])} bend candidates "
            f"(labeled: {self.positives['VI']} vibrato, {self.positives['BE']} bends)"
        )


def estimate_labels(
    labeled: Sequence[IdmtNote], estimate: Sequence[Note], pairs: dict, tolerance: float = 0.05
) -> list[str]:
    """The expression each transcribed note is taken to have: its matched reference
    note's; else, a reference note starting within ``tolerance`` and up to three
    semitones away (a bent note named after its bent pitch); else normal (a note the
    transcription made up)."""
    labels = ["NO"] * len(estimate)
    matched = {j: i for i, j in pairs.items()}
    for j, note in enumerate(estimate):
        if j in matched:
            labels[j] = labeled[matched[j]].expression
            continue
        near = [
            (abs(r.note.start - note.start), r.expression)
            for r in labeled
            if abs(r.note.start - note.start) <= tolerance and abs(r.note.pitch - note.pitch) <= 3
        ]
        if near:
            labels[j] = min(near)[1]
    return labels


def transcribe(dataset: str, take: str, y: np.ndarray, args: argparse.Namespace) -> list[Note]:
    from ml.inference.predict import Predictor
    from ml.preprocessing.spectrogram import compute_features

    if not hasattr(transcribe, "predictor"):
        transcribe.predictor = Predictor.from_checkpoint(
            args.checkpoint, tab_checkpoint=args.tab_checkpoint
        )
        inference = transcribe.predictor.cfg["inference"]
        for item in args.inference:
            name, _, value = item.partition("=")
            inference[name] = json.loads(value)
    predictor = transcribe.predictor
    cache = NOTES_CACHE / dataset / f"{take}.npz"
    if cache.exists():
        with np.load(cache) as data:
            probs = dict(data)
    else:
        probs = predictor.predict_features(compute_features(y, predictor.cfg))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **probs)
    return predictor.decode(probs)


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


def run_idmt(
    args: argparse.Namespace,
    s: TechniqueSettings,
    counts: Counter,
    classifier: TechniqueClassifier | None = None,
    dump: FeatureDump | None = None,
) -> None:
    for take in tqdm(find_takes(guitars=tuple(args.guitars))):
        y = load_audio(take.audio_path, SAMPLE_RATE)
        labeled = load_annotation(take.annotation_path)
        reference = [n.note for n in labeled]
        trace: list | None = [] if dump is not None else None
        if args.notes == "reference":
            estimate = reference
            est_labels = own_note_labels(reference, make_tracker(y, s, args), classifier, trace)
            pairs = {i: i for i in range(len(reference))}
            note_labels = [n.expression for n in labeled]
        else:
            estimate = detect_techniques(
                transcribe("idmt", take.name, y, args),
                y,
                SAMPLE_RATE,
                s,
                STANDARD_TUNING,
                tracker=make_tracker(y, s, args),
                classifier=classifier,
                trace=trace,
            )
            est_labels = [{"bend": n.bend, "vibrato": n.vibrato} for n in estimate]
            pairs = match(reference, estimate)
            counts["notes_ref"] += len(reference)
            counts["notes_matched"] += len(pairs)
            note_labels = estimate_labels(labeled, estimate, pairs)
        if dump is not None:
            dump.add(take.name, trace, note_labels, labeled, s)

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
    args: argparse.Namespace,
    s: TechniqueSettings,
    counts: Counter,
    errors: list | None = None,
    classifier: TechniqueClassifier | None = None,
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
            est_labels = own_note_labels(reference, tracker, classifier)
            pairs = {i: i for i in range(len(reference))}
        else:
            notes = transcribe("guitarset", jams.stem, y, args)
            estimate = detect_techniques(
                notes, y, SAMPLE_RATE, s, STANDARD_TUNING, tracker=tracker, classifier=classifier
            )
            est_labels = [{"bend": n.bend, "vibrato": n.vibrato} for n in estimate]
            pairs = match(reference, estimate)
            counts["notes_ref"] += len(reference)
            counts["notes_matched"] += len(pairs)
        counts["marked_notes"] += len(est_labels)
        for name in ("bend", "vibrato"):
            ref_on = {i for i, lab in enumerate(ref_labels) if lab[name]}
            est_on = {j for j, lab in enumerate(est_labels) if lab[name]}
            hits = {i for i in ref_on if pairs.get(i) in est_on}
            score(counts, name, ref_on, est_on, hits)
            # Independent of the labels (the rules' verdict on the pickup): how many
            # notes are marked at all. On comps, few notes are bent or shaken.
            counts["marked", name] += len(est_on)
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
    if counts["marked_notes"]:
        total = counts["marked_notes"]
        marked = {name: counts["marked", name] for name in ("bend", "vibrato")}
        shares = ", ".join(f"{k} {n} ({1000 * n / total:.1f} per 1000)" for k, n in marked.items())
        print(f"notes marked: {shares}, of {total} notes")
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
    parser.add_argument("--inference", nargs="*", default=[], metavar="NAME=JSON")
    parser.add_argument("--classifier", help="learned bend/vibrato decisions (JSON)")
    parser.add_argument("--dump", type=Path, help="IDMT: write labeled features (.npz)")
    args = parser.parse_args(argv)
    s = settings_from(args.set)
    classifier = TechniqueClassifier.load(args.classifier) if args.classifier else None
    if args.dump is not None and args.dataset != "idmt":
        raise SystemExit("--dump needs --dataset idmt (its labels)")

    counts: Counter = Counter()
    errors: list = []
    if args.dataset == "idmt":
        dump = FeatureDump() if args.dump is not None else None
        run_idmt(args, s, counts, classifier, dump)
        if dump is not None:
            info = {
                "guitars": args.guitars,
                "notes": args.notes,
                "tracker": args.tracker,
                "settings": args.set,
                "inference": args.inference,
            }
            if args.notes == "model":
                info["checkpoint"] = args.checkpoint
            dump.save(args.dump, info)
    else:
        run_guitarset(args, s, counts, errors, classifier)
    report(counts, args.dataset, args.notes, errors)


if __name__ == "__main__":
    main()
