"""Transcription metrics.

* frame level: multi-pitch precision / recall / F1 on piano rolls
* note level: onset (and optionally offset) matching via ``mir_eval``
* tablature: string/fret correctness of the notes that were detected correctly
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from music_core.notes import Note, midi_to_hz


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    pred, target = pred.astype(bool), target.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    return tp, int(pred.sum()) - tp, int(target.sum()) - tp


def frame_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Precision / recall / F1 over active (frame, pitch) cells of boolean ``(T, P)`` rolls."""
    return _prf(*_counts(pred, target))


def tab_to_pianoroll(
    tab: np.ndarray, tuning: Sequence[int], min_midi: int, n_pitches: int
) -> np.ndarray:
    """Map ``(T, strings)`` tab classes (0 = silent, k = fret k - 1) to a boolean piano roll."""
    roll = np.zeros((tab.shape[0], n_pitches), dtype=bool)
    for string, open_pitch in enumerate(tuning):
        frames = np.nonzero(tab[:, string] > 0)[0]
        pitches = open_pitch + tab[frames, string] - 1 - min_midi
        valid = (pitches >= 0) & (pitches < n_pitches)
        roll[frames[valid], pitches[valid]] = True
    return roll


def tab_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    tuning: Sequence[int] | None = None,
    min_midi: int | None = None,
    n_pitches: int | None = None,
) -> dict[str, float]:
    """Frame-level tablature metrics (TabCNN, Wiggins & Kim 2019) for ``(T, strings)`` classes.

    ``precision/recall/f1`` count a string activation as correct only if the fret matches.
    With a tuning, also reports pitch metrics and the tab disambiguation rate (TDR): the
    share of correctly detected pitches that were also placed on the right string.
    """
    active_pred, active_true = pred > 0, target > 0
    tp = int((active_pred & active_true & (pred == target)).sum())
    metrics = _prf(tp, int(active_pred.sum()) - tp, int(active_true.sum()) - tp)
    if tuning is not None and min_midi is not None and n_pitches is not None:
        pitch_counts = _counts(
            tab_to_pianoroll(pred, tuning, min_midi, n_pitches),
            tab_to_pianoroll(target, tuning, min_midi, n_pitches),
        )
        metrics.update({f"pitch_{k}": v for k, v in _prf(*pitch_counts).items()})
        metrics["tdr"] = tp / pitch_counts[0] if pitch_counts[0] else 0.0
    return metrics


def _to_mir_eval(notes: Sequence[Note]) -> tuple[np.ndarray, np.ndarray]:
    valid = [n for n in notes if n.end > n.start]
    if not valid:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.start, n.end] for n in valid])
    pitches = np.array([midi_to_hz(n.pitch) for n in valid])
    return intervals, pitches


def note_metrics(
    reference: Sequence[Note],
    estimated: Sequence[Note],
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = None,
) -> dict[str, float]:
    """Note-level P/R/F1. ``offset_ratio=None`` ignores offsets; 0.2 is the MIREX setting."""
    import mir_eval

    ref_intervals, ref_pitches = _to_mir_eval(reference)
    est_intervals, est_pitches = _to_mir_eval(estimated)
    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def tab_note_accuracy(
    reference: Sequence[Note],
    estimated: Sequence[Note],
    onset_tolerance: float = 0.05,
) -> dict[str, float]:
    """How often correctly detected notes are placed on the annotated string and fret.

    Estimated notes are greedily matched to reference notes with the same pitch and an
    onset within ``onset_tolerance``; only matches whose reference has a string count.
    """
    unused = [n for n in reference if n.string is not None]
    matched = correct = 0
    for est in sorted(estimated, key=lambda n: n.start):
        best, best_dt = None, onset_tolerance
        for ref in unused:
            dt = abs(ref.start - est.start)
            if ref.pitch == est.pitch and dt <= best_dt:
                best, best_dt = ref, dt
        if best is None:
            continue
        unused.remove(best)
        matched += 1
        correct += int(best.string == est.string and best.fret == est.fret)
    return {"accuracy": correct / matched if matched else 0.0, "matched": float(matched)}


def mean_confidence(notes: Sequence[Note]) -> float | None:
    values = [n.confidence for n in notes if n.confidence is not None]
    return float(np.mean(values)) if values else None
