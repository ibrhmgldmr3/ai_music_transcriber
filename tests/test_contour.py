import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ml.preprocessing.annotations import contour_bins, contours_to_targets
from ml.preprocessing.augmentation import variant_contours

CONFIG = Path(__file__).resolve().parents[1] / "ml" / "configs" / "guitar_contour.yaml"
SR, HOP = 44100, 512


def targets(contours, n_frames=20):
    return contours_to_targets(contours, n_frames, SR, HOP, min_midi=40, max_midi=52)


def test_a_contour_point_peaks_at_its_bin_with_gaussian_neighbours():
    fps = SR / HOP
    times = np.arange(5, 10) / fps
    target = targets({0: (times, np.full(5, 45.0))}).astype(float)
    assert target.shape == (20, contour_bins(40, 52))
    row = target[7]
    assert row.argmax() == 15  # (45 - 40) * 3
    assert row[15] == pytest.approx(1.0)
    assert row[14] == pytest.approx(np.exp(-0.5), abs=1e-3)
    assert row[11] == 0  # beyond 3 sigma
    assert target[:5].max() == 0 and target[10:].max() == 0


def test_a_bend_between_bins_lands_between_them_and_unvoiced_points_stay_empty():
    fps = SR / HOP
    times = np.array([3, 4, 5]) / fps
    target = targets({2: (times, np.array([45.0 + 1 / 6, np.nan, 47.0]))}).astype(float)
    assert target[3, 15] == pytest.approx(target[3, 16], abs=1e-3)  # halfway: equal heights
    assert target[4].max() == 0
    assert target[5].argmax() == 21


def test_strings_share_a_frame_and_out_of_range_points_are_dropped():
    fps = SR / HOP
    times = np.array([2]) / fps
    target = targets(
        {0: (times, np.array([41.0])), 1: (times, np.array([50.0])), 2: (times, np.array([80.0]))}
    ).astype(float)
    assert target[2, 3] == pytest.approx(1.0) and target[2, 30] == pytest.approx(1.0)
    assert target[2].sum() == pytest.approx(2 * target[2, :10].sum(), rel=1e-3)


def test_contours_follow_pitch_shift_and_time_stretch():
    contours = {0: (np.array([1.0, 2.0]), np.array([45.0, np.nan]))}
    shifted = variant_contours(contours, {"name": "ps-2", "pitch_shift": -2})
    np.testing.assert_allclose(shifted[0][0], [1.0, 2.0])
    np.testing.assert_allclose(shifted[0][1], [43.0, np.nan])
    stretched = variant_contours(contours, {"name": "tempo0.9", "time_stretch": 0.9})
    np.testing.assert_allclose(stretched[0][0], [1 / 0.9, 2 / 0.9])
    assert variant_contours(contours, {"name": "room", "room": True})[0][1][0] == 45.0


torch = pytest.importorskip("torch")

from ml.config import contour_bins as config_contour_bins  # noqa: E402
from ml.config import feature_bins, load_config, num_pitches  # noqa: E402
from ml.models import build_model  # noqa: E402
from ml.training.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint  # noqa: E402
from ml.training.losses import TranscriptionLoss  # noqa: E402

SMALL = {"model": {"conv_channels": [8, 8, 8], "embed_dim": 16, "rnn_hidden": 8, "rnn_layers": 1}}


def test_contour_model_outputs_and_loss():
    cfg = load_config(CONFIG, SMALL)
    assert config_contour_bins(cfg) == 157
    model = build_model(cfg)
    outputs = model(torch.randn(2, feature_bins(cfg), 24))
    assert outputs["contour"].shape == (2, 24, 157)
    assert outputs["frame"].shape == (2, 24, num_pitches(cfg))

    batch = {k: torch.zeros(2, 24, num_pitches(cfg)) for k in ("onset", "frame", "offset")}
    batch["contour"] = torch.zeros(2, 24, 157)
    loss, parts = TranscriptionLoss()(outputs, batch)
    loss.backward()
    assert set(parts) == {"contour", "onset", "frame", "offset"}


def test_freq_pools_keep_resolution_and_note_models_are_unchanged():
    cfg = load_config(CONFIG, SMALL)
    pooled = build_model(
        load_config(CONFIG, {**SMALL, "model": {**SMALL["model"], "freq_pools": None}})
    )
    assert build_model(cfg).encoder.proj[0].in_features == 8 * (252 // 4)
    assert pooled.encoder.proj[0].in_features == 8 * (252 // 8)


def test_a_contour_head_needs_contour_labels():
    cfg = load_config(CONFIG, SMALL)
    del cfg["labels"]["contour"]
    with pytest.raises(ValueError, match="labels.contour"):
        build_model(cfg)


def write_history(path, scores):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_contour_f1"])
        writer.writeheader()
        writer.writerows({"epoch": i + 1, "val_contour_f1": s} for i, s in enumerate(scores))


def test_a_resumed_run_keeps_the_best_score_patience_and_history(tmp_path):
    history = tmp_path / "history.csv"
    write_history(history, [0.5, 0.7, 0.6, 0.65, 0.66])  # epoch 5 then crashed mid-epoch 6
    torch.save({"metrics": {"val_contour_f1": 0.7}}, tmp_path / "best.pt")
    trainer = SimpleNamespace(start_epoch=5, should_stop=False)  # last.pt was epoch 4

    checkpoint = ModelCheckpoint(tmp_path, "val_contour_f1")
    checkpoint.on_train_begin(trainer)
    assert checkpoint.best == 0.7

    stopping = EarlyStopping("val_contour_f1", patience=3, history=history)
    stopping.on_train_begin(trainer)
    assert (stopping.best, stopping.wait) == (0.7, 2)  # epochs 3 and 4 didn't improve
    stopping.on_epoch_end(trainer, 5, {"val_contour_f1": 0.69})
    assert trainer.should_stop

    CSVLogger(history).on_train_begin(trainer)
    with history.open(newline="", encoding="utf-8") as f:
        assert [row["epoch"] for row in csv.DictReader(f)] == ["1", "2", "3", "4"]


def test_a_fresh_run_ignores_old_state(tmp_path):
    history = tmp_path / "history.csv"
    write_history(history, [0.9])
    torch.save({"metrics": {"val_contour_f1": 0.9}}, tmp_path / "best.pt")
    trainer = SimpleNamespace(start_epoch=1, should_stop=False)
    checkpoint = ModelCheckpoint(tmp_path, "val_contour_f1")
    checkpoint.on_train_begin(trainer)
    stopping = EarlyStopping("val_contour_f1", history=history)
    stopping.on_train_begin(trainer)
    assert checkpoint.best is None and stopping.best is None
