"""BTC chord recognizer (Park et al., "A Bi-Directional Transformer for Musical Chord
Recognition", ISMIR 2019), inference only.

Adapted from https://github.com/jayg996/BTC-ISMIR19 (btc_model.py and
utils/transformer_modules.py); module and parameter names are kept so the published
checkpoints load unchanged. Training-only parts (losses, dropout) are left out.

MIT License, Copyright (c) 2019 Jonggwon Park. Permission is hereby granted, free of
charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software. THE
SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

# The published large-vocabulary model: 144 CQT bins in, 170 chord classes out, windows
# of 108 frames (10 s).
CONFIG = {
    "feature_size": 144,
    "hidden_size": 128,
    "num_layers": 8,
    "num_heads": 4,
    "total_key_depth": 128,
    "total_value_depth": 128,
    "filter_size": 128,
    "timestep": 108,
    "num_chords": 170,
}


def _bias_mask(length: int) -> torch.Tensor:
    """-inf above the diagonal: attention only to earlier (or, transposed, later) frames."""
    mask = np.triu(np.full([length, length], -np.inf), 1)
    return torch.from_numpy(mask).float()[None, None]


def _timing_signal(length: int, channels: int) -> torch.Tensor:
    position = np.arange(length)
    timescales = channels // 2
    increment = math.log(1.0e4) / (timescales - 1)
    inv = np.exp(np.arange(timescales).astype(float) * -increment)
    scaled = position[:, None] * inv[None, :]
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]])
    return torch.from_numpy(signal.reshape(1, length, channels)).float()


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(
        self, depth: int, key_depth: int, value_depth: int, heads: int, mask: torch.Tensor
    ):
        super().__init__()
        self.heads = heads
        self.scale = (key_depth // heads) ** -0.5
        self.register_buffer("bias_mask", mask, persistent=False)
        self.query_linear = nn.Linear(depth, key_depth, bias=False)
        self.key_linear = nn.Linear(depth, key_depth, bias=False)
        self.value_linear = nn.Linear(depth, value_depth, bias=False)
        self.output_linear = nn.Linear(value_depth, depth, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.heads, d // self.heads).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.query_linear(x)) * self.scale
        k = self._split(self.key_linear(x))
        v = self._split(self.value_linear(x))
        logits = q @ k.transpose(-1, -2)
        logits = logits + self.bias_mask[:, :, : logits.shape[-2], : logits.shape[-1]]
        context = torch.softmax(logits, dim=-1) @ v
        b, h, t, d = context.shape
        return self.output_linear(context.permute(0, 2, 1, 3).reshape(b, t, h * d))


class Conv(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.pad = nn.ConstantPad1d((1, 1), 0)
        self.conv = nn.Conv1d(in_size, out_size, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x.permute(0, 2, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, depth: int, filter_size: int):
        super().__init__()
        self.layers = nn.ModuleList([Conv(depth, filter_size), Conv(filter_size, depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))  # the original applies ReLU after both layers
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, c: dict, mask: torch.Tensor):
        super().__init__()
        h = c["hidden_size"]
        self.multi_head_attention = MultiHeadAttention(
            h, c["total_key_depth"], c["total_value_depth"], c["num_heads"], mask
        )
        self.positionwise_convolution = PositionwiseFeedForward(h, c["filter_size"])
        self.layer_norm_mha = LayerNorm(h)
        self.layer_norm_ffn = LayerNorm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_head_attention(self.layer_norm_mha(x))
        return x + self.positionwise_convolution(self.layer_norm_ffn(x))


class BiDirectionalSelfAttention(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        mask = _bias_mask(c["timestep"])
        self.attn_block = SelfAttentionBlock(c, mask)
        self.backward_attn_block = SelfAttentionBlock(c, mask.transpose(2, 3))
        self.linear = nn.Linear(c["hidden_size"] * 2, c["hidden_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([self.attn_block(x), self.backward_attn_block(x)], dim=2))


class BiDirectionalSelfAttentionLayers(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.register_buffer(
            "timing_signal", _timing_signal(c["timestep"], c["hidden_size"]), persistent=False
        )
        self.embedding_proj = nn.Linear(c["feature_size"], c["hidden_size"], bias=False)
        self.self_attn_layers = nn.Sequential(
            *[BiDirectionalSelfAttention(c) for _ in range(c["num_layers"])]
        )
        self.layer_norm = LayerNorm(c["hidden_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_proj(x) + self.timing_signal[:, : x.shape[1]]
        return self.layer_norm(self.self_attn_layers(x))


class SoftmaxOutputLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.output_projection = nn.Linear(hidden_size, output_size)
        # Unused at inference, but part of the published checkpoints.
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output_projection(hidden)


class BTC(nn.Module):
    """Normalized log-CQT windows ``(batch, 108, 144)`` -> chord logits ``(batch, 108, 170)``."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        c = config or CONFIG
        self.timestep = c["timestep"]
        self.self_attn_layers = BiDirectionalSelfAttentionLayers(c)
        self.output_layer = SoftmaxOutputLayer(c["hidden_size"], c["num_chords"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.self_attn_layers(x))
