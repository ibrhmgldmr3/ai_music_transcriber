"""Song mode: isolate the guitar (or, in voice mode, the vocals) from a full mix.

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
        self.sample_rate = int(self.model.samplerate)

    def __call__(self, y: np.ndarray, sample_rate: int, stem: str = "guitar") -> np.ndarray:
        """Mono mix -> peak-normalized mono ``stem`` (e.g. "vocals") at the same rate."""
        return normalize_peak(self.stems(y, sample_rate, [stem])[stem]).astype(np.float32)

    @torch.no_grad()
    def stems(
        self, y: np.ndarray, sample_rate: int, names: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Mono mix -> mono stems at the same rate and level (all, or ``names``)."""
        import librosa
        from demucs.apply import apply_model

        wanted = names or list(self.model.sources)
        x = y
        if sample_rate != self.sample_rate:
            x = librosa.resample(y, orig_sr=sample_rate, target_sr=self.sample_rate)
        mix = torch.from_numpy(np.stack([x, x])).float()  # the model expects stereo
        center, scale = mix.mean(), mix.std() + 1e-8  # Demucs' own input normalization
        sources = apply_model(
            self.model,
            ((mix - center) / scale)[None].to(self.device),
            shifts=0,  # the default 1 is one pass at a random offset: same quality, not repeatable
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
        out = {}
        for name in wanted:
            stem = (sources[self.model.sources.index(name)] * scale + center).mean(0).cpu().numpy()
            if sample_rate != self.sample_rate:
                stem = librosa.resample(stem, orig_sr=self.sample_rate, target_sr=sample_rate)
            out[name] = np.pad(stem, (0, max(0, len(y) - len(stem))))[: len(y)].astype(np.float32)
        return out
