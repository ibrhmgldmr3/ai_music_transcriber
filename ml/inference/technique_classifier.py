"""Learned decisions for bends and vibrato on a note's pitch curve.

The rules in ``ml.inference.techniques`` decide from a few thresholds chosen by hand.
On IDMT-SMT-Guitar they mark few wrong notes but miss many: with IDMT's own notes,
half of the vibrato notes. Here the same pitch curve is described by a handful of
measurements (``vibrato_features``, ``bend_features``) and a small model fitted on
labeled notes (``scripts/train_technique_classifier.py``) decides:

* vibrato: every note heard for at least ``MIN_VIBRATO_SECONDS`` is a candidate.
* bend: a looser version of the bend rule (``LOOSE_BEND``) proposes a fretted note,
  an amount and where the rise starts; the model accepts or rejects it. So the model
  can drop the rule's false bends and add bends the strict thresholds missed, but no
  bend the loose rule doesn't see.

The model is a JSON file: logistic regression or gradient-boosted trees, read and run
with numpy alone (no pickle, no scikit-learn needed to transcribe). Either technique may
be missing from a file; the rule then decides it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from ml.inference.techniques import (
    Bend,
    TechniqueSettings,
    _smooth,
    bend_shape,
    find_bend,
    has_vibrato,
)

FORMAT = "technique-classifier@1"

# Shorter notes than the rule's vibrato_min (0.4 s) are candidates too: the note model's
# notes are 20-25 % shorter than IDMT's, which alone lost 6 of LP's 54 vibrato notes.
MIN_VIBRATO_SECONDS = 0.2

VIBRATO_FEATURES = (
    "heard",  # seconds of the note with a pitch
    "voiced",  # share of its frames with a pitch
    "depth",  # 2 sqrt(2) x std of the curve without its slow course (the rule's depth)
    "spread",  # 95th - 5th percentile of the same
    "regularity",  # autocorrelation at the best swing period in 4-9 Hz (the rule's)
    "rate",  # that period's swings per second (0: none measurable)
    "slow_regularity",  # the same for 2-4 Hz swings, with a slower trend removed
    "slow_depth",
    "crossings",  # zero crossings of the swing per second
    "drift",  # 95th - 5th percentile of the slow course (bends, glides)
    "offset",  # |median pitch - transcribed pitch|
    "rule",  # has_vibrato
)

# The bend rule with its thresholds loosened: candidates for the model to judge. Each
# field is the loosest value that still describes a rise from a fretted note.
LOOSE_BEND = {
    "bend_min": 0.4,
    "bend_rise": 0.0,
    "bend_step": 1.0,
    "base_tolerance": 0.5,
}

BEND_FEATURES = (
    "amount",  # semitones held above the fretted note
    "base_offset",  # |start's lowest part - fretted note|
    "below_pitch",  # transcribed pitch - fretted note
    "reaches",  # fretted + amount - transcribed pitch
    "rise_time",  # seconds from the fretted note to 85 % of the bend
    "step_share",  # largest step on the way / amount (slides climb in big steps)
    "start",  # seconds into the note where the rise starts
    "heard",
    "voiced",
    "depth",  # vibrato depth: a bend with vibrato, or vibrato mistaken for a bend
    "rule",  # find_bend with the settings in use
)


def vibrato_features(
    midi: np.ndarray, frame_rate: float, s: TechniqueSettings = TechniqueSettings()
) -> np.ndarray | None:
    """``VIBRATO_FEATURES`` of a note's pitch curve (fractional MIDI per frame, NaN where
    unheard); None for notes heard for less than ``MIN_VIBRATO_SECONDS``."""
    curve = _smooth(midi, 3)
    valid = ~np.isnan(curve)
    heard = np.count_nonzero(valid) / frame_rate
    if heard < MIN_VIBRATO_SECONDS:
        return None
    x = np.interp(np.arange(len(curve)), np.flatnonzero(valid), curve[valid])
    fast, fast_trend = _swing(x, frame_rate, s.vibrato_rates)
    slow, _ = _swing(x, frame_rate, (2.0, s.vibrato_rates[0]))
    detrended = x - fast_trend
    crossings = np.count_nonzero(np.diff(np.signbit(detrended - detrended.mean())))
    return np.array(
        [
            heard,
            float(valid.mean()),
            fast["depth"],
            fast["spread"],
            fast["regularity"],
            fast["rate"],
            slow["regularity"],
            slow["depth"],
            crossings / (len(x) / frame_rate),
            float(np.percentile(fast_trend, 95) - np.percentile(fast_trend, 5)),
            float(abs(np.median(x) - round(float(np.median(x))))),
            float(has_vibrato(midi, frame_rate, s)),
        ]
    )


def _swing(
    x: np.ndarray, frame_rate: float, rates: tuple[float, float]
) -> tuple[dict[str, float], np.ndarray]:
    """Depth and regularity of the swing at ``rates`` Hz, as in ``has_vibrato``: the slow
    course (a moving average over the slowest swing's period) removed, then the best
    autocorrelation over the periods that fit twice into the note."""
    trend = ndimage.uniform_filter1d(x, max(3, int(round(frame_rate / rates[0]))), mode="nearest")
    d = x - trend
    out = {
        "depth": float(2 * np.sqrt(2) * d.std()),
        "spread": float(np.percentile(d, 95) - np.percentile(d, 5)),
        "regularity": 0.0,
        "rate": 0.0,
    }
    lo = max(1, int(frame_rate / rates[1]))
    hi = min(int(np.ceil(frame_rate / rates[0])), len(d) // 2)
    d = d - d.mean()
    energy = float(np.dot(d, d))
    if hi >= lo and energy > 1e-12:
        corr = [float(np.dot(d[:-lag], d[lag:])) / energy for lag in range(lo, hi + 1)]
        best = int(np.argmax(corr))
        out["regularity"] = corr[best]
        out["rate"] = frame_rate / (lo + best)
    return out, trend


def bend_candidate(
    midi: np.ndarray, pitch: int, frame_rate: float, s: TechniqueSettings = TechniqueSettings()
) -> tuple[Bend, np.ndarray] | None:
    """The bend the loose rule finds on a note and its ``BEND_FEATURES``, or None."""
    loose = replace(s, **LOOSE_BEND)
    bend = find_bend(midi, pitch, frame_rate, loose)
    if bend is None:
        return None
    shape = bend_shape(midi, pitch, frame_rate, loose)
    valid = ~np.isnan(midi)
    vibrato = vibrato_features(midi, frame_rate, s)
    features = np.array(
        [
            shape.amount,
            abs(shape.base - shape.fretted),
            pitch - bend.fretted,
            bend.fretted + shape.amount - pitch,
            shape.rise_time,
            shape.max_step / shape.amount,
            shape.start / frame_rate,
            np.count_nonzero(valid) / frame_rate,
            float(valid.mean()),
            vibrato[2] if vibrato is not None else 0.0,
            float(find_bend(midi, pitch, frame_rate, s) is not None),
        ]
    )
    return bend, features


class Model:
    """A binary classifier read from JSON: ``probability(X)`` for rows of features."""

    def __init__(self, spec: dict[str, Any]):
        self.kind = spec["kind"]
        self.features = tuple(spec["features"])
        self.threshold = float(spec["threshold"])
        if self.kind == "logistic":
            self.mean = np.asarray(spec["mean"], dtype=float)
            self.scale = np.asarray(spec["scale"], dtype=float)
            self.coef = np.asarray(spec["coef"], dtype=float)
            self.intercept = float(spec["intercept"])
        elif self.kind == "trees":
            self.init = float(spec["init"])
            self.learning_rate = float(spec["learning_rate"])
            self.trees = [{k: np.asarray(v) for k, v in tree.items()} for tree in spec["trees"]]
        else:
            raise ValueError(f"Unknown classifier kind {self.kind!r}")

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.features):
            raise ValueError(f"Expected {len(self.features)} features, got {X.shape[1]}")
        if self.kind == "logistic":
            return ((X - self.mean) / self.scale) @ self.coef + self.intercept
        total = np.full(len(X), self.init)
        for tree in self.trees:
            total += self.learning_rate * _tree_predict(tree, X)
        return total

    def probability(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def accepts(self, x: np.ndarray) -> bool:
        return bool(self.probability(x)[0] >= self.threshold)


def _tree_predict(tree: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """A regression tree in scikit-learn's array layout (leaves: left == -1)."""
    X = X.astype(np.float32).astype(float)  # scikit-learn's trees compare in float32
    node = np.zeros(len(X), dtype=int)
    rows = np.arange(len(X))
    for _ in range(64):
        inner = tree["left"][node] >= 0
        if not inner.any():
            break
        n = node[inner]
        go_left = X[rows[inner], tree["feature"][n]] <= tree["threshold"][n]
        node[inner] = np.where(go_left, tree["left"][n], tree["right"][n])
    return tree["value"][node].astype(float)


@dataclass
class TechniqueClassifier:
    """Learned vibrato and bend decisions; a missing one falls back to the rule."""

    vibrato_model: Model | None = None
    bend_model: Model | None = None
    name: str = ""

    @classmethod
    def load(cls, path: str | Path) -> TechniqueClassifier:
        path = Path(path)
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        models = {}
        for kind, names in (("vibrato", VIBRATO_FEATURES), ("bend", BEND_FEATURES)):
            if spec.get(kind):
                model = Model(spec[kind])
                if model.features != names:
                    raise ValueError(f"{path}: {kind} features differ from this version's")
                models[kind] = model
        return cls(models.get("vibrato"), models.get("bend"), path.parent.name or path.stem)

    def vibrato(
        self, midi: np.ndarray, frame_rate: float, s: TechniqueSettings = TechniqueSettings()
    ) -> bool:
        if self.vibrato_model is None:
            return has_vibrato(midi, frame_rate, s)
        features = vibrato_features(midi, frame_rate, s)
        return features is not None and self.vibrato_model.accepts(features)

    def bend(
        self,
        midi: np.ndarray,
        pitch: int,
        frame_rate: float,
        s: TechniqueSettings = TechniqueSettings(),
    ) -> Bend | None:
        if self.bend_model is None:
            return find_bend(midi, pitch, frame_rate, s)
        candidate = bend_candidate(midi, pitch, frame_rate, s)
        if candidate is None:
            return None
        bend, features = candidate
        return bend if self.bend_model.accepts(features) else None


def save(path: str | Path, models: dict[str, dict[str, Any] | None], **info: Any) -> None:
    """Write a classifier file: ``models`` maps "vibrato"/"bend" to a model spec."""
    spec = {"format": FORMAT, **info, **models}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
