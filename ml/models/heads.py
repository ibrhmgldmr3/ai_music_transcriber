"""Output heads operating on per-frame embeddings ``(B, T, D)``."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

# Heads producing per-frame, per-pitch logits (B, T, pitches).
PITCH_HEADS = ("onset", "frame", "offset")


class PitchHead(nn.Module):
    """Per-frame, per-pitch logits: note onsets, frame activity or offsets."""

    def __init__(self, in_dim: int, n_pitches: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_pitches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabHead(nn.Module):
    """Optional string/fret classifier: logits ``(B, T, strings, frets + 2)``.

    Class 0 means the string is silent; class k > 0 means fret k - 1 is played.
    """

    def __init__(self, in_dim: int, num_strings: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.num_strings = num_strings
        self.num_classes = num_classes
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_strings * num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return self.net(x).view(b, t, self.num_strings, self.num_classes)


def build_heads(
    in_dim: int,
    names: Iterable[str],
    n_pitches: int,
    num_strings: int,
    num_classes: int,
    dropout: float = 0.0,
) -> nn.ModuleDict:
    heads: dict[str, nn.Module] = {}
    for name in names:
        if name in PITCH_HEADS:
            heads[name] = PitchHead(in_dim, n_pitches, dropout)
        elif name == "tab":
            heads[name] = TabHead(in_dim, num_strings, num_classes, dropout)
        else:
            raise ValueError(f"Unknown head {name!r}; choose from {PITCH_HEADS + ('tab',)}")
    if "frame" not in heads:
        raise ValueError("The 'frame' head is required for note decoding")
    return nn.ModuleDict(heads)
