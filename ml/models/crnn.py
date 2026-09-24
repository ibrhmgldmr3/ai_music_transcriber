"""CNN + bidirectional LSTM/GRU transcription model.

spectrogram -> ConvStack -> BiLSTM -> [conv ; rnn] -> onset / frame / offset (/ tab) heads
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ml.models.cnn import ConvStack
from ml.models.heads import build_heads

RNN_TYPES = {"lstm": nn.LSTM, "gru": nn.GRU}


class CRNN(nn.Module):
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
        rnn_type: str = "lstm",
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        if rnn_type not in RNN_TYPES:
            raise ValueError(f"Unknown rnn_type {rnn_type!r}; choose from {sorted(RNN_TYPES)}")
        self.encoder = ConvStack(n_bins, conv_channels, dropout, embed_dim)
        self.rnn = RNN_TYPES[rnn_type](
            embed_dim,
            rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        rnn_out = rnn_hidden * (2 if bidirectional else 1)
        # Heads see the conv embedding (spectral evidence) next to the RNN output (temporal
        # context). Without this skip, gradients reach the CNN only through the saturating
        # RNN and training stalls at the "predict nothing" baseline.
        self.heads = build_heads(
            embed_dim + rnn_out, heads, n_pitches, num_strings, num_classes, dropout
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        conv = self.encoder(x)
        context, _ = self.rnn(conv)
        h = torch.cat([conv, context], dim=-1)
        return {name: head(h) for name, head in self.heads.items()}
