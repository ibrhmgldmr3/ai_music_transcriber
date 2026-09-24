"""Song mode: isolate the guitar from a full mix before transcription.

Uses Demucs ``htdemucs_6s`` (Rouard et al., 2023), which separates drums, bass, vocals,
piano, guitar and "other". Its weights are downloaded on first use into the torch hub
cache. The transcription model was trained on solo guitar, so on band recordings the
guitar stem is transcribed instead of the whole mix.
"""

from __future__ import annotations

import numpy as np
import torch

from ml.models import resolve_device
from ml.preprocessing.audio import normalize_peak

MODEL_NAME = "htdemucs_6s"


class GuitarSeparator:
    def __init__(self, device: str | None = None, model_name: str = MODEL_NAME):
        from demucs.pretrained import get_model

        self.device = resolve_device(device)
        self.model = get_model(model_name).to(self.device).eval()
        if "guitar" not in self.model.sources:
            raise ValueError(f"Demucs model {model_name} has no guitar stem")
        self.guitar = self.model.sources.index("guitar")
        self.sample_rate = int(self.model.samplerate)

    @torch.no_grad()
    def __call__(self, y: np.ndarray, sample_rate: int) -> np.ndarray:
        """Mono mix -> peak-normalized mono guitar stem at the same sample rate."""
        import librosa
        from demucs.apply import apply_model

        x = y
        if sample_rate != self.sample_rate:
            x = librosa.resample(y, orig_sr=sample_rate, target_sr=self.sample_rate)
        mix = torch.from_numpy(np.stack([x, x])).float()  # the model expects stereo
        center, scale = mix.mean(), mix.std() + 1e-8  # Demucs' own input normalization
        sources = apply_model(
            self.model,
            ((mix - center) / scale)[None].to(self.device),
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
        guitar = (sources[self.guitar] * scale + center).mean(0).cpu().numpy()
        if sample_rate != self.sample_rate:
            guitar = librosa.resample(guitar, orig_sr=self.sample_rate, target_sr=sample_rate)
        guitar = np.pad(guitar, (0, max(0, len(y) - len(guitar))))[: len(y)]
        return normalize_peak(guitar).astype(np.float32)
