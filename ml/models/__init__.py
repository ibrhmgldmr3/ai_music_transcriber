"""Model definitions and a config-driven factory."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ml.config import contour_bins, feature_bins, num_pitches, tab_shape
from ml.models.cnn import CNN, ConvStack
from ml.models.crnn import CRNN
from ml.models.heads import PITCH_HEADS, PitchHead, TabHead, build_heads

__all__ = [
    "CNN",
    "CRNN",
    "PITCH_HEADS",
    "ConvStack",
    "PitchHead",
    "TabHead",
    "build_heads",
    "build_model",
    "resolve_device",
]


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    num_strings, num_classes = tab_shape(cfg)
    common = dict(
        n_bins=feature_bins(cfg),
        n_pitches=num_pitches(cfg),
        heads=tuple(model_cfg.get("heads", ("onset", "frame", "offset"))),
        num_strings=num_strings,
        num_classes=num_classes,
        conv_channels=tuple(model_cfg.get("conv_channels", (32, 64, 128))),
        embed_dim=model_cfg.get("embed_dim", 512),
        dropout=model_cfg.get("dropout", 0.25),
        freq_pools=model_cfg.get("freq_pools"),
        n_contour_bins=contour_bins(cfg),
    )
    name = model_cfg.get("name", "crnn")
    contour_conv = _contour_conv(cfg)
    if name == "cnn":
        if contour_conv is not None:
            raise ValueError("model.contour_conv needs the crnn model")
        return CNN(**common)
    if name == "crnn":
        return CRNN(
            **common,
            rnn_type=model_cfg.get("rnn_type", "lstm"),
            rnn_hidden=model_cfg.get("rnn_hidden", 256),
            rnn_layers=model_cfg.get("rnn_layers", 2),
            bidirectional=model_cfg.get("bidirectional", True),
            contour_conv=contour_conv,
        )
    raise ValueError(f"Unknown model {name!r} (expected 'cnn' or 'crnn')")


def _contour_conv(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """``ConvContourHead`` arguments from ``model.contour_conv``, laid on the CQT's bins."""
    options = cfg["model"].get("contour_conv")
    if not options:
        return None
    features, labels = cfg["features"], cfg["labels"]
    per_semitone = int(labels["contour"].get("bins_per_semitone", 3))
    if features["type"] != "cqt" or features["bins_per_octave"] != 12 * per_semitone:
        raise ValueError(
            "model.contour_conv needs a CQT with the contour's resolution "
            f"({12 * per_semitone} bins per octave)"
        )
    return {
        "n_bins": features["n_bins"],
        "bins_per_octave": features["bins_per_octave"],
        "n_out": contour_bins(cfg),
        "offset": (labels["min_midi"] - features["fmin_midi"]) * per_semitone,
        **options,
    }


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
