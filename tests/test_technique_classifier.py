import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ml.inference.technique_classifier import (
    BEND_FEATURES,
    FORMAT,
    VIBRATO_FEATURES,
    Model,
    TechniqueClassifier,
    bend_candidate,
    save,
    vibrato_features,
)
from ml.inference.techniques import Bend, PitchTracker, detect_techniques, find_bend
from music_core.notes import Note

SR = 44100
RATE = SR / 256  # PitchTracker frames per second


def tone(midi: np.ndarray, decay: float = 1.5) -> np.ndarray:
    hz = 440.0 * 2.0 ** ((midi - 69) / 12)
    phase = 2 * np.pi * np.cumsum(hz) / SR
    t = np.arange(len(midi)) / SR
    return (sum(np.sin(k * phase) / k for k in range(1, 9)) * np.exp(-decay * t)).astype(np.float32)


def frames(*segments: tuple[float, float, float]) -> np.ndarray:
    """A pitch curve at the tracker's frame rate: (seconds, from, to) segments."""
    return np.concatenate([np.linspace(a, b, int(seconds * RATE)) for seconds, a, b in segments])


def vibrato_curve(seconds: float, depth: float, rate: float = 6.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return 64 + depth / 2 * np.sin(2 * np.pi * rate * t)


def feature(row: np.ndarray, names: tuple[str, ...], name: str) -> float:
    return float(row[names.index(name)])


def test_vibrato_features_separate_a_swing_from_a_steady_note():
    swing = vibrato_features(vibrato_curve(0.8, 0.4), RATE)
    steady = vibrato_features(frames((0.8, 64, 64)), RATE)
    assert len(swing) == len(VIBRATO_FEATURES)
    assert feature(swing, VIBRATO_FEATURES, "rule") == 1.0
    assert feature(steady, VIBRATO_FEATURES, "rule") == 0.0
    assert feature(swing, VIBRATO_FEATURES, "depth") > 0.3
    assert feature(steady, VIBRATO_FEATURES, "depth") < 0.01
    assert feature(swing, VIBRATO_FEATURES, "regularity") > 0.5
    assert feature(swing, VIBRATO_FEATURES, "rate") == pytest.approx(6.0, rel=0.1)
    assert feature(swing, VIBRATO_FEATURES, "heard") == pytest.approx(0.8, abs=0.01)


def test_short_notes_are_vibrato_candidates_below_the_rules_minimum():
    # The rule wants 0.4 s of a note; the classifier sees notes from 0.2 s on.
    short = vibrato_features(vibrato_curve(0.3, 0.4), RATE)
    assert short is not None and feature(short, VIBRATO_FEATURES, "rule") == 0.0
    assert vibrato_features(vibrato_curve(0.1, 0.4), RATE) is None


def test_bend_candidates():
    whole = frames((0.15, 64, 64), (0.08, 64, 66), (0.3, 66, 66))
    bend, row = bend_candidate(whole, 64, RATE)
    assert bend == Bend(64, 2.0)
    assert feature(row, BEND_FEATURES, "rule") == 1.0
    assert feature(row, BEND_FEATURES, "amount") == pytest.approx(2.0, abs=0.05)
    # 15 % -> 85 % of an 80 ms rise.
    assert feature(row, BEND_FEATURES, "rise_time") == pytest.approx(0.7 * 0.08, abs=0.01)

    # A shallow bend: below the rule's 0.75 semitones, a candidate for the classifier.
    shallow = frames((0.15, 64, 64), (0.08, 64, 64.55), (0.3, 64.55, 64.55))
    assert find_bend(shallow, 64, RATE) is None
    bend, row = bend_candidate(shallow, 64, RATE)
    assert bend.fretted == 64 and bend.amount == 0.5
    assert feature(row, BEND_FEATURES, "rule") == 0.0

    assert bend_candidate(frames((0.5, 64, 64)), 64, RATE) is None


def logistic(names, intercept: float, threshold: float = 0.5, coef=None) -> dict:
    n = len(names)
    return {
        "kind": "logistic",
        "features": list(names),
        "threshold": threshold,
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "coef": coef if coef is not None else [0.0] * n,
        "intercept": intercept,
    }


def test_logistic_model():
    names = ("a", "b")
    model = Model(logistic(names, -1.0, coef=[2.0, -1.0]))
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 3.0]])
    assert model.probability(X) == pytest.approx(1 / (1 + np.exp(-(X @ [2.0, -1.0] - 1.0))))
    assert model.accepts(X[1]) and not model.accepts(X[2])
    with pytest.raises(ValueError):
        model.probability(np.zeros((1, 3)))


def test_tree_model():
    # x0 <= 0.5 ? (x1 <= 2 ? -1 : 3) : 2
    tree = {
        "left": [1, 3, -1, -1, -1],
        "right": [2, 4, -1, -1, -1],
        "feature": [0, 1, 0, 0, 0],
        "threshold": [0.5, 2.0, -2.0, -2.0, -2.0],
        "value": [0.0, 0.0, 2.0, -1.0, 3.0],
    }
    spec = {"kind": "trees", "features": ["x0", "x1"], "threshold": 0.5, "init": 0.25}
    model = Model({**spec, "learning_rate": 0.5, "trees": [tree, tree]})
    X = np.array([[0.0, 1.0], [0.0, 5.0], [1.0, 0.0]])
    assert model.decision(X) == pytest.approx(0.25 + 0.5 * 2 * np.array([-1.0, 3.0, 2.0]))


def write(path: Path, vibrato=None, bend=None) -> Path:
    save(path, {"vibrato": vibrato, "bend": bend}, trained_on=[])
    return path


def test_load_checks_the_file(tmp_path):
    path = write(tmp_path / "techniques" / "classifier.json", logistic(VIBRATO_FEATURES, 0.0))
    classifier = TechniqueClassifier.load(path)
    assert classifier.vibrato_model is not None and classifier.bend_model is None
    assert classifier.name == "techniques"

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError):
        TechniqueClassifier.load(wrong)
    stale = write(tmp_path / "stale.json", logistic(("depth", "rate"), 0.0))
    with pytest.raises(ValueError, match="features differ"):
        TechniqueClassifier.load(stale)
    assert json.loads(path.read_text())["format"] == FORMAT


def test_detect_techniques_follows_the_classifier(tmp_path):
    # A steady E4 and a whole-step bend; the classifier says vibrato to every note and
    # no to every bend candidate.
    y = np.concatenate(
        [tone(np.full(int(0.6 * SR), 64.0)), tone(np.concatenate([
            np.full(int(0.15 * SR), 64.0), np.linspace(64, 66, int(0.08 * SR)),
            np.full(int(0.3 * SR), 66.0),
        ]))]
    )  # fmt: skip
    notes = [Note(64, 0.0, 0.6, string=4, fret=5), Note(64, 0.6, 1.13, string=4, fret=5)]
    rules = detect_techniques(notes, y, SR)
    assert [(n.bend, n.vibrato) for n in rules] == [(None, False), (2.0, False)]

    path = write(
        tmp_path / "c.json", logistic(VIBRATO_FEATURES, 5.0), logistic(BEND_FEATURES, -5.0)
    )
    trace: list = []
    learned = detect_techniques(
        notes, y, SR, classifier=TechniqueClassifier.load(path), trace=trace
    )
    assert [(n.bend, n.vibrato) for n in learned] == [(None, True), (None, True)]
    assert sorted(entry["index"] for entry in trace) == [0, 1]
    entry = next(e for e in trace if e["index"] == 1)
    assert entry["frame_rate"] == pytest.approx(RATE)
    curve, pitch = entry["bend"][0]
    assert pitch == 64 and bend_candidate(curve, pitch, RATE) is not None

    # Only a vibrato model: bends stay with the rule.
    only_vibrato = TechniqueClassifier.load(
        write(tmp_path / "v.json", logistic(VIBRATO_FEATURES, -5.0))
    )
    assert [n.bend for n in detect_techniques(notes, y, SR, classifier=only_vibrato)] == [None, 2.0]


def test_curves_match_the_trackers():
    # The candidate features are measured on what the tracker gives, not on ideal curves.
    t = np.arange(int(0.8 * SR)) / SR
    y = tone(64 + 0.2 * np.sin(2 * np.pi * 6 * t))
    midi = PitchTracker(y, SR).note_curve(Note(64, 0.0, 0.8)).midi
    row = vibrato_features(midi, RATE)
    assert feature(row, VIBRATO_FEATURES, "rule") == 1.0
    assert feature(row, VIBRATO_FEATURES, "rate") == pytest.approx(6.0, rel=0.15)


@pytest.fixture
def trainer():
    pytest.importorskip("sklearn")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import train_technique_classifier

        yield train_technique_classifier
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("kind", ["logistic", "trees"])
def test_exported_models_give_scikit_learns_probabilities(trainer, kind):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(VIBRATO_FEATURES)))
    y = X[:, 2] + 0.5 * X[:, 4] + 0.3 * rng.normal(size=300) > 0.5
    make = next(m for name, m in trainer.candidates(0).items() if name.startswith(kind))
    model = make().fit(X, y)
    spec = trainer.export(model, VIBRATO_FEATURES, 0.4)
    assert spec["kind"] == kind
    expected = model.predict_proba(X)[:, 1]
    assert Model(json.loads(json.dumps(spec))).probability(X) == pytest.approx(expected, abs=1e-6)


def test_training_from_dumps(trainer, tmp_path):
    """End to end on a synthetic dump: rows, grouped cross-validation, a JSON file the
    classifier loads."""
    rng = np.random.default_rng(1)
    arrays = {}
    for kind, names in (("vibrato", VIBRATO_FEATURES), ("bend", BEND_FEATURES)):
        n = 400
        X = rng.normal(size=(n, len(names)))
        label = X[:, 0] + 0.2 * rng.normal(size=n) > 0.8
        X[:, names.index("rule")] = (X[:, 0] > 1.4).astype(float)  # a strict rule
        arrays[f"{kind}_X"] = X
        arrays[f"{kind}_label"] = label
        arrays[f"{kind}_take"] = np.array([f"take{i % 20}" for i in range(n)])
        arrays[f"{kind}_note"] = np.arange(n)
        arrays[f"{kind}_features"] = np.array(names)
        arrays[f"positives_{kind}"] = np.array(int(label.sum()) + 5)  # 5 without a row
    arrays["info"] = np.array(json.dumps({"guitars": ["AR", "FS"], "notes": "model"}))
    dump = tmp_path / "dump.npz"
    np.savez(dump, **arrays)

    rows = trainer.load_rows([dump], "bend")
    rule = rows.X[:, BEND_FEATURES.index("rule")]
    best, label = trainer.note_scores(rows, rule)
    p, r, _ = trainer.prf(best > 0.5, label, rows.positives)
    assert p > 0.9 and r < 0.6  # the strict rule: precise, misses many

    out = tmp_path / "techniques" / "classifier.json"
    trainer.main([str(dump), "--out", str(out), "--model", "logistic", "--folds", "4"])
    spec = json.loads(out.read_text())
    for kind in ("vibrato", "bend"):
        cv = spec[kind]["cross_validation"]
        assert cv["f"] > cv["rule"]["f"]
        assert cv["recall"] <= label.sum() / rows.positives + 1e-9
    assert spec["trained_on"][0]["file"] == "dump.npz"
    classifier = TechniqueClassifier.load(out)
    assert classifier.vibrato_model.kind == "logistic"


def test_transcribed_notes_take_their_reference_notes_label():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        from benchmark_techniques import estimate_labels, match
    finally:
        sys.path.pop(0)
    from ml.datasets.idmt import IdmtNote

    labeled = [
        IdmtNote(Note(64, 0.0, 0.5), "BE", "PK"),
        IdmtNote(Note(67, 0.6, 1.0), "VI", "PK"),
        IdmtNote(Note(60, 1.1, 1.5), "NO", "PK"),
    ]
    estimate = [
        Note(66, 0.01, 0.5),  # the bend, named after its bent pitch: no exact match
        Note(67, 0.62, 1.0),
        Note(60, 1.1, 1.5),
        Note(72, 2.0, 2.2),  # made up
    ]
    pairs = match([n.note for n in labeled], estimate)
    assert pairs == {1: 1, 2: 2}
    assert estimate_labels(labeled, estimate, pairs) == ["BE", "VI", "NO", "NO"]
