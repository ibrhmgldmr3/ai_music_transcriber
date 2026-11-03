"""Fit the learned bend and vibrato decisions (``ml.inference.technique_classifier``).

Input: feature dumps of labeled notes from ``scripts/benchmark_techniques.py --dump``
(IDMT-SMT-Guitar; pick the guitars there, e.g. AR and FS, and keep LP for the test).
Several dumps can be combined, e.g. one on IDMT's own notes and one on the model's.

Model choice and decision threshold come from cross-validation grouped by take (no
take's notes are both fitted and scored). The score is the F-measure over the labeled
notes: a bend the loose rule proposes no candidate for counts as missed. The rules'
score on the same rows is printed next to it. The chosen model is then fitted on all
rows and written as JSON (no pickle); ``--classifier`` in the benchmark, the API's
``TECHNIQUE_CLASSIFIER`` and ``python -m ml.inference.predict --technique-classifier``
use it.

    pip install -e ".[train]"   # scikit-learn
    python scripts/benchmark_techniques.py --dataset idmt --guitars AR FS --notes model \\
        --dump ml/data/interim/technique_features/arfs_model.npz
    python scripts/benchmark_techniques.py --dataset idmt --guitars AR FS --notes reference \\
        --dump ml/data/interim/technique_features/arfs_reference.npz
    python scripts/train_technique_classifier.py ml/data/interim/technique_features/arfs_*.npz \\
        --out ml/checkpoints/techniques/classifier.json
    python scripts/benchmark_techniques.py --dataset idmt --guitars LP --notes model \\
        --classifier ml/checkpoints/techniques/classifier.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.inference.technique_classifier import BEND_FEATURES, VIBRATO_FEATURES, save

FEATURES = {"vibrato": VIBRATO_FEATURES, "bend": BEND_FEATURES}
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)


@dataclass
class Rows:
    X: np.ndarray
    y: np.ndarray
    take: np.ndarray  # group for cross-validation
    note: np.ndarray  # (dump, take, note) id: a note's candidates count once
    positives: int  # labeled notes, with or without a row

    def __len__(self) -> int:
        return len(self.y)


def load_rows(paths: Sequence[Path], kind: str) -> Rows:
    Xs, ys, takes, notes, positives = [], [], [], [], 0
    for k, path in enumerate(paths):
        with np.load(path) as data:
            names = tuple(str(n) for n in data[f"{kind}_features"])
            if names != FEATURES[kind]:
                raise SystemExit(f"{path}: {kind} features differ from this version's; dump again")
            Xs.append(data[f"{kind}_X"])
            ys.append(data[f"{kind}_label"])
            take = data[f"{kind}_take"]
            takes.append(take)
            notes.append([f"{k}/{t}/{n}" for t, n in zip(take, data[f"{kind}_note"])])
            positives += int(data[f"positives_{kind}"])
    X = np.concatenate(Xs) if Xs else np.zeros((0, len(FEATURES[kind])))
    return Rows(
        np.nan_to_num(X.astype(float)),
        np.concatenate(ys).astype(bool),
        np.concatenate(takes),
        np.concatenate(notes),
        positives,
    )


def note_scores(rows: Rows, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per note: the highest probability among its rows, and its label."""
    order = np.argsort(rows.note, kind="stable")
    ids, first = np.unique(rows.note[order], return_index=True)
    best = np.maximum.reduceat(probability[order], first)
    label = np.logical_or.reduceat(rows.y[order], first)
    return best, label


def prf(predicted: np.ndarray, label: np.ndarray, positives: int) -> tuple[float, float, float]:
    hits = int(np.count_nonzero(predicted & label))
    p = hits / max(int(np.count_nonzero(predicted)), 1)
    r = hits / max(positives, 1)
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def candidates(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models: dict[str, Any] = {}
    for c in (0.1, 1.0, 10.0):
        models[f"logistic C={c}"] = lambda c=c: make_pipeline(
            StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        )
    for depth in (2, 3):
        for n in (100, 300):
            models[f"trees depth={depth} n={n}"] = lambda depth=depth, n=n: (
                GradientBoostingClassifier(
                    n_estimators=n,
                    max_depth=depth,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_leaf=5,
                    random_state=seed,
                )
            )
    return models


def cross_validated(make, rows: Rows, folds: int) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    probability = np.zeros(len(rows))
    splits = min(folds, len(np.unique(rows.take)))
    for train, test in GroupKFold(n_splits=splits).split(rows.X, rows.y, rows.take):
        if rows.y[train].all() or not rows.y[train].any():
            probability[test] = rows.y[train].mean()
            continue
        model = make().fit(rows.X[train], rows.y[train])
        probability[test] = model.predict_proba(rows.X[test])[:, 1]
    return probability


def best_threshold(rows: Rows, probability: np.ndarray) -> tuple[float, tuple]:
    best, label = note_scores(rows, probability)
    scores = [(prf(best >= t, label, rows.positives), float(t)) for t in THRESHOLDS]
    (p, r, f), t = max(scores, key=lambda item: (item[0][2], -abs(item[1] - 0.5)))
    return t, (p, r, f)


def export(model, names: Sequence[str], threshold: float) -> dict[str, Any]:
    """The fitted model as a ``technique_classifier.Model`` spec."""
    spec: dict[str, Any] = {"features": list(names), "threshold": threshold}
    if hasattr(model, "estimators_"):  # GradientBoostingClassifier
        trees = []
        for (estimator,) in model.estimators_:
            t = estimator.tree_
            trees.append(
                {
                    "left": t.children_left.tolist(),
                    "right": t.children_right.tolist(),
                    "feature": [max(int(f), 0) for f in t.feature],
                    "threshold": t.threshold.tolist(),
                    "value": t.value[:, 0, 0].tolist(),
                }
            )
        spec.update(kind="trees", learning_rate=model.learning_rate, trees=trees)
        # The trees' sum starts from the prior's log-odds.
        x0 = np.zeros((1, len(names)))
        tree_sum = sum(e.predict(x0)[0] for (e,) in model.estimators_)
        spec["init"] = float(model.decision_function(x0)[0] - model.learning_rate * tree_sum)
    else:  # StandardScaler + LogisticRegression
        scaler, logistic = model.steps[0][1], model.steps[-1][1]
        spec.update(
            kind="logistic",
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
            coef=logistic.coef_[0].tolist(),
            intercept=float(logistic.intercept_[0]),
        )
    return spec


def fit_kind(kind: str, rows: Rows, args: argparse.Namespace) -> dict[str, Any] | None:
    names = FEATURES[kind]
    n_pos = int(np.count_nonzero(rows.y))
    print(
        f"\n{kind}: {len(rows)} rows from {len(np.unique(rows.take))} takes, "
        f"{n_pos} labeled {kind} rows; {rows.positives} labeled notes in all"
    )
    if min(n_pos, len(rows) - n_pos) < 5:
        print(
            f"  fitting needs at least 5 rows labeled {kind} and 5 not; the rule keeps deciding it"
        )
        return None
    rule = (rows.X[:, names.index("rule")] > 0.5).astype(float)
    rule_best, label = note_scores(rows, rule)
    rule_score = prf(rule_best > 0.5, label, rows.positives)
    _, has_row = note_scores(rows, np.ones(len(rows)))
    print("  {:26s}  P {:.3f}  R {:.3f}  F {:.3f}".format("rule", *rule_score))
    print(
        f"  (labeled notes with a row: {np.count_nonzero(has_row) / max(rows.positives, 1):.3f}, "
        "the highest recall possible)"
    )

    models = candidates(args.seed)
    if args.model != "auto":
        models = {k: v for k, v in models.items() if k.startswith(args.model)}
    results = []
    for name, make in models.items():
        threshold, score = best_threshold(rows, cross_validated(make, rows, args.folds))
        results.append((score[2], name, threshold, score))
        print("  {:26s}  P {:.3f}  R {:.3f}  F {:.3f}".format(name, *score), end="")
        print(f"  (threshold {threshold:.3f})")
    # Ties go to the simpler model (logistic regressions come first).
    _, name, threshold, score = max(results, key=lambda result: result[0])
    print(f"  chosen: {name}, threshold {threshold:.3f}")
    model = models[name]().fit(rows.X, rows.y)
    spec = export(model, names, threshold)
    _verify_export(model.predict_proba(rows.X)[:, 1], spec, rows.X)
    keys = ("precision", "recall", "f")
    spec["cross_validation"] = {
        "model": name,
        **dict(zip(keys, score)),
        "rule": dict(zip(keys, rule_score)),
        "rows": len(rows),
        "labeled_notes": rows.positives,
    }
    return spec


def _verify_export(expected: np.ndarray, spec: dict[str, Any], X: np.ndarray) -> None:
    """The exported JSON must give scikit-learn's probabilities."""
    from ml.inference.technique_classifier import Model

    got = Model(spec).probability(X)
    if not np.allclose(got, expected, atol=1e-6):
        raise RuntimeError("Exported classifier disagrees with the fitted model")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dumps", nargs="+", type=Path, help="benchmark_techniques.py --dump files")
    parser.add_argument(
        "--out", type=Path, default=Path("ml/checkpoints/techniques/classifier.json")
    )
    parser.add_argument(
        "--kinds", nargs="+", choices=("vibrato", "bend"), default=["vibrato", "bend"]
    )
    parser.add_argument("--model", choices=("auto", "logistic", "trees"), default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    models = {}
    for kind in args.kinds:
        models[kind] = fit_kind(kind, load_rows(args.dumps, kind), args)
    if not any(models.values()):
        raise SystemExit("Nothing fitted.")
    info = []
    for path in args.dumps:
        with np.load(path) as data:
            info.append({"file": path.name, **json.loads(str(data["info"]))})
    save(args.out, models, trained_on=info)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
