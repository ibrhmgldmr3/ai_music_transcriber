from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.config import feature_bins, load_config, num_pitches, tab_shape  # noqa: E402
from ml.models import build_model  # noqa: E402
from ml.training.losses import TranscriptionLoss  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "ml" / "configs" / "guitar.yaml"
TAB_CONFIG = CONFIG.with_name("guitar_tab.yaml")
HEADS_TAB = ["onset", "frame", "offset", "tab"]


def test_tab_config_adds_the_tab_head():
    assert load_config(TAB_CONFIG)["model"]["heads"] == HEADS_TAB


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"model": {"rnn_type": "gru"}},
        {"model": {"name": "cnn"}},
        {"features": {"type": "cqt"}, "model": {"heads": ["onset", "frame", "offset", "tab"]}},
    ],
)
def test_forward_shapes_and_loss(overrides):
    cfg = load_config(CONFIG, overrides)
    model = build_model(cfg)
    x = torch.randn(2, feature_bins(cfg), 40)
    outputs = model(x)

    pitches = num_pitches(cfg)
    for head in ("onset", "frame", "offset"):
        assert outputs[head].shape == (2, 40, pitches)
    batch = {head: torch.zeros(2, 40, pitches) for head in ("onset", "frame", "offset")}
    if "tab" in outputs:
        assert outputs["tab"].shape == (2, 40, *tab_shape(cfg))
        batch["tab"] = torch.zeros(2, 40, tab_shape(cfg)[0], dtype=torch.long)

    loss, parts = TranscriptionLoss()(outputs, batch)
    loss.backward()
    assert set(parts) == set(outputs)


def test_predictor_takes_positions_from_a_separate_tab_model():
    from ml.inference.predict import Predictor

    small = {"model": {"conv_channels": [8], "embed_dim": 16, "rnn_hidden": 8, "rnn_layers": 1}}
    note_cfg = load_config(CONFIG, small)
    tab_cfg = load_config(CONFIG, {**small, "model": {**small["model"], "heads": HEADS_TAB}})
    predictor = Predictor(build_model(note_cfg), note_cfg, "cpu", build_model(tab_cfg), tab_cfg)
    probs = predictor.predict_features(np.random.randn(feature_bins(note_cfg), 30))
    assert probs["frame"].shape == (30, num_pitches(note_cfg))
    assert probs["tab"].shape == (30, *tab_shape(tab_cfg))

    mismatched = load_config(CONFIG, {**small, "features": {"type": "cqt"}})
    with pytest.raises(ValueError, match="features"):
        Predictor(build_model(note_cfg), note_cfg, "cpu", build_model(mismatched), mismatched)


def test_crnn_can_memorize_a_batch():
    """Guards against architectures that stall at the 'predict nothing' baseline."""
    torch.manual_seed(0)
    small = {
        "model": {
            "conv_channels": [8, 16],
            "embed_dim": 64,
            "rnn_hidden": 32,
            "rnn_layers": 1,
            "dropout": 0.0,
        }
    }
    cfg = load_config(CONFIG, small)
    model = build_model(cfg)
    pitches = num_pitches(cfg)
    x = torch.randn(2, feature_bins(cfg), 32)
    frame = (x[:, :pitches, :] > 1.0).float().transpose(1, 2)  # learnable from the input
    batch = {"frame": frame, "onset": torch.zeros_like(frame), "offset": torch.zeros_like(frame)}

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = TranscriptionLoss()
    for _ in range(150):
        optimizer.zero_grad()
        loss, parts = loss_fn(model(x), batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x)["frame"])
    assert probs[frame > 0].mean() > 0.8
    assert probs[frame == 0].mean() < 0.2
