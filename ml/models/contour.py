"""Frequency-convolutional pitch-contour head (as in Basic Pitch, Bittner et al., 2022).

The linear contour head reads the pitch from the conv stack's frame embedding, after
the frequency axis has been pooled and projected away; its output stepped from bin to
bin and lost vibrato (README, "Perde eğrisi modeli"). This head works on the
spectrogram itself and keeps the frequency axis to the end: each output bin is computed
at its own pitch by convolutions that slide along frequency, so a pitch moving between
bins moves the activation with it.

A harmonic stack lets those local convolutions see a note's overtones: channel h holds
the spectrum shifted down by harmonic h's interval, so at the bin of a fundamental all
channels line up the energy of its harmonics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

DEFAULT_HARMONICS = (0.5, 1, 2, 3, 4, 5)


class HarmonicStack(nn.Module):
    """``(B, F, T)`` log-frequency spectrogram -> ``(B, H, F, T)``.

    Channel h at bin f holds bin ``f + bins_per_octave * log2(harmonics[h])``; bins
    shifted in from beyond the spectrogram are zeros (the mean after normalization).
    """

    def __init__(self, bins_per_octave: int, harmonics: Sequence[float] = DEFAULT_HARMONICS):
        super().__init__()
        self.shifts = [int(round(bins_per_octave * math.log2(h))) for h in harmonics]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, f, t = x.shape
        out = x.new_zeros(b, len(self.shifts), f, t)
        for i, shift in enumerate(self.shifts):
            if shift >= 0:
                out[:, i, : f - shift] = x[:, shift:]
            else:
                out[:, i, -shift:] = x[:, : f + shift]
        return out


class ConvContourHead(nn.Module):
    """Contour logits ``(B, T, n_out)`` straight from the spectrogram ``(B, n_bins, T)``.

    Output bin k is input bin ``offset + k``, so the input must have the contour's
    resolution (bins per octave = 12 x contour bins per semitone). Two 5x5 convolutions
    over the harmonic stack, one wide along frequency (``freq_kernel`` bins: the
    neighbouring pitches and partials a note's bin is told apart from) and a 5x5 output
    convolution; the receptive field spans 15 frames (~170 ms at hop 512).
    """

    def __init__(
        self,
        n_bins: int,
        bins_per_octave: int,
        n_out: int,
        offset: int = 0,
        harmonics: Sequence[float] = DEFAULT_HARMONICS,
        channels: int = 16,
        freq_kernel: int = 39,
        dropout: float = 0.0,
    ):
        super().__init__()
        if offset < 0 or offset + n_out > n_bins:
            raise ValueError(f"Contour bins {offset}..{offset + n_out} outside {n_bins} input bins")
        self.offset, self.n_out = offset, n_out
        self.norm = nn.BatchNorm1d(n_bins)
        self.stack = HarmonicStack(bins_per_octave, harmonics)
        narrow = max(1, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(len(harmonics), channels, 5, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, narrow, (freq_kernel, 3), padding=(freq_kernel // 2, 1)),
            nn.BatchNorm2d(narrow),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(narrow, 1, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(self.stack(self.norm(x)))[:, 0]  # (B, F, T)
        return logits[:, self.offset : self.offset + self.n_out].transpose(1, 2)
