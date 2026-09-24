"""Convolutional front-end and a frame-wise CNN baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ml.models.heads import build_heads


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool_freq: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(pool_freq, 1)) if pool_freq > 1 else nn.Identity(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvStack(nn.Module):
    """Spectrogram ``(B, bins, T)`` -> frame embeddings ``(B, T, embed_dim)``.

    Pooling is applied along frequency only, so the output keeps the label frame rate.
    """

    def __init__(
        self,
        n_bins: int,
        channels: Sequence[int] = (32, 64, 128),
        dropout: float = 0.25,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(n_bins)  # per-bin normalization of log features
        blocks, in_ch, freq = [], 1, n_bins
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch, pool_freq=2, dropout=dropout))
            in_ch, freq = out_ch, freq // 2
        if freq < 1:
            raise ValueError(f"Too many pooling stages for {n_bins} input bins")
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Sequential(
            nn.Linear(in_ch * freq, embed_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x).unsqueeze(1)  # (B, 1, F, T)
        x = self.blocks(x)  # (B, C, F', T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T, C * F')
        return self.proj(x)


class CNN(nn.Module):
    """Frame-wise baseline without recurrence (receptive field of a few frames)."""

    def __init__(
        self,
        n_bins: int,
        n_pitches: int,
        heads: Sequence[str] = ("onset", "frame", "offset"),
        num_strings: int = 6,
        num_classes: int = 22,
        conv_channels: Sequence[int] = (32, 64, 128),
        embed_dim: int = 512,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = ConvStack(n_bins, conv_channels, dropout, embed_dim)
        self.heads = build_heads(embed_dim, heads, n_pitches, num_strings, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {name: head(h) for name, head in self.heads.items()}
