"""Multi-task transcription loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ml.models.heads import PITCH_HEADS


class TranscriptionLoss(nn.Module):
    """Weighted sum of per-head losses.

    * onset / frame / offset: binary cross-entropy per (frame, pitch); onsets and offsets
      are sparse, so a ``pos_weight`` > 1 counteracts the class imbalance.
    * tab: cross-entropy over the string classes (0 = silent, k = fret k - 1).
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        pos_weights: dict[str, float | None] | None = None,
    ):
        super().__init__()
        self.weights = {"onset": 1.0, "frame": 1.0, "offset": 0.5, "tab": 1.0} | (weights or {})
        self.pos_weights = {k: v for k, v in (pos_weights or {}).items() if v}

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses: dict[str, torch.Tensor] = {}
        for name in PITCH_HEADS:
            if name in outputs and name in batch:
                logits = outputs[name]
                pos_weight = self.pos_weights.get(name)
                losses[name] = F.binary_cross_entropy_with_logits(
                    logits,
                    batch[name],
                    pos_weight=(
                        torch.tensor(pos_weight, device=logits.device) if pos_weight else None
                    ),
                )
        if "tab" in outputs and "tab" in batch:
            logits = outputs["tab"]
            losses["tab"] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), batch["tab"].reshape(-1)
            )
        if not losses:
            raise ValueError("No matching outputs/targets to compute a loss")

        total = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        return total, {k: float(v.detach()) for k, v in losses.items()}
