"""Transcribe audio files with a trained checkpoint.

python -m ml.inference.predict guitar.wav --midi out.mid --tab out.txt
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ml.inference.postprocess import assign_positions_from_tab, decode_notes
from ml.models import build_model, resolve_device
from ml.preprocessing.audio import load_audio
from ml.preprocessing.spectrogram import compute_features
from music_core.analysis import refine_tempo
from music_core.notes import Note
from music_core.tab import assign_tab

DEFAULT_CHECKPOINT = Path("ml/checkpoints/guitar/best.pt")

# progress(fraction, stage) with stage one of "loading", "separating", "transcribing",
# "finishing"; fraction is the whole job's, in [0, 1].
ProgressCallback = Callable[[float, str], None]


@dataclass
class TranscriptionResult:
    notes: list[Note]
    duration: float
    tempo: float | None = None
    tuning: list[int] = field(default_factory=list)
    # Semitones the notes were moved from the recording's pitch (voice mode, to fit the
    # guitar's range).
    transpose: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": [n.to_dict() for n in self.notes],
            "duration": self.duration,
            "tempo": self.tempo,
            "tuning": self.tuning,
            "transpose": self.transpose,
        }


class Predictor:
    """Wraps a trained model: audio -> features -> probabilities -> note events -> tab.

    An optional second ``tab_model`` supplies only the string/fret probabilities. Training
    a tab head costs the shared layers some pitch accuracy, so pairing the best note
    model with a tab-head model gives both: on the GuitarSet test split, notes from the
    note model plus positions from the tab model got 81% of reference notes fully right
    (pitch, onset and string/fret) versus 77% for the tab-head model alone.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict[str, Any],
        device: str | None = None,
        tab_model: nn.Module | None = None,
        tab_cfg: dict[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tab_model = tab_model.to(self.device).eval() if tab_model is not None else None
        if tab_cfg is not None:
            for section in ("audio", "features"):
                if tab_cfg[section] != cfg[section]:
                    raise ValueError(f"Tab model uses different '{section}' settings")
        # The tuning the tab head learned positions in.
        self.head_tuning = list(((tab_cfg or cfg).get("tab") or {}).get("tuning", []))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | None = None,
        tab_checkpoint: str | Path | None = None,
    ) -> Predictor:
        model, cfg = _load(path)
        tab_model, tab_cfg = _load(tab_checkpoint) if tab_checkpoint else (None, None)
        if tab_cfg is not None and "tab" not in tab_cfg["model"]["heads"]:
            raise ValueError(f"{tab_checkpoint} has no tab head")
        return cls(model, cfg, device, tab_model, tab_cfg)

    @property
    def frame_rate(self) -> float:
        audio = self.cfg["audio"]
        return audio["sample_rate"] / audio["hop_length"]

    @property
    def tuning(self) -> list[int]:
        return list((self.cfg.get("tab") or {}).get("tuning", []))

    def predict_features(
        self,
        features: np.ndarray,
        progress: Callable[[float], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the model(s) on ``(bins, T)`` features; ``progress`` gets the done fraction.

        Returns sigmoid probabilities ``(T, pitches)`` per pitch head and softmax
        probabilities ``(T, strings, classes)`` for the tab head (from the tab model
        when one is set).
        """
        x = torch.from_numpy(np.asarray(features, dtype=np.float32))
        if x.shape[1] == 0:
            raise ValueError("Audio is too short to transcribe")
        models = 2 if self.tab_model is not None else 1
        report = progress or (lambda _: None)
        probs = self._run(self.model, x, lambda f: report(f / models))
        if self.tab_model is not None:
            probs["tab"] = self._run(self.tab_model, x, lambda f: report((1 + f) / models))["tab"]
        return probs

    @torch.no_grad()
    def _run(
        self,
        model: nn.Module,
        x: torch.Tensor,
        progress: Callable[[float], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Chunked inference with context frames on both sides of every chunk."""
        inference = self.cfg["inference"]
        step = int(inference.get("chunk_frames", 1024))
        context = int(inference.get("context_frames", 64))
        n_frames = x.shape[1]

        chunks: dict[str, list[np.ndarray]] = {}
        for start in range(0, n_frames, step):
            lo, hi = max(0, start - context), min(n_frames, start + step + context)
            outputs = model(x[:, lo:hi].unsqueeze(0).to(self.device))
            keep = slice(start - lo, start - lo + min(step, n_frames - start))
            for name, logits in outputs.items():
                probs = torch.softmax(logits, dim=-1) if name == "tab" else torch.sigmoid(logits)
                chunks.setdefault(name, []).append(probs[0, keep].float().cpu().numpy())
            if progress is not None:
                progress(min(1.0, (start + step) / n_frames))
        return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}

    def decode(
        self, probs: dict[str, np.ndarray], tuning: Sequence[int] | None = None
    ) -> list[Note]:
        """Probabilities -> note events, with string/fret positions when a tuning is set.

        ``tuning`` (open strings, capo included) replaces the model's standard tuning.
        """
        inference, labels = self.cfg["inference"], self.cfg["labels"]
        notes = decode_notes(
            probs["frame"],
            probs.get("onset"),
            probs.get("offset"),
            frame_rate=self.frame_rate,
            min_midi=labels["min_midi"],
            onset_threshold=inference["onset_threshold"],
            frame_threshold=inference["frame_threshold"],
            offset_threshold=inference.get("offset_threshold", 0.5),
            min_duration=inference["min_note_duration"],
        )
        tab_cfg = self.cfg.get("tab")
        if not tab_cfg:
            return notes
        tuning, num_frets = list(tuning or tab_cfg["tuning"]), tab_cfg["num_frets"]
        if "tab" in probs:
            return assign_positions_from_tab(
                notes,
                probs["tab"],
                frame_rate=self.frame_rate,
                tuning=tuning,
                num_frets=num_frets,
                model_tuning=self.head_tuning,
            )
        return assign_tab(notes, tuning, num_frets)

    def transcribe(
        self,
        audio_path: str | Path,
        estimate_tempo: bool = True,
        separator: Callable[[np.ndarray, int], np.ndarray] | None = None,
        tuning: Sequence[int] | None = None,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """Transcribe a file; ``separator`` (e.g. ``GuitarSeparator``) first isolates the
        guitar from a band mix. ``tuning``: open strings, capo included (default: the
        model's standard tuning). ``progress`` follows the job's stages."""
        report = progress or (lambda fraction, stage: None)
        # Share of the job before the model runs: separation takes about as long as the rest.
        model_start = 0.5 if separator is not None else 0.05
        audio = self.cfg["audio"]
        sr = audio["sample_rate"]
        report(0.0, "loading")
        y = load_audio(audio_path, sr, mono=True, normalize=audio.get("normalize", True))
        if separator is not None:
            report(0.05, "separating")
            y = separator(y, sr)
        report(model_start, "transcribing")
        probs = self.predict_features(
            compute_features(y, self.cfg),
            lambda f: report(model_start + (0.95 - model_start) * f, "transcribing"),
        )
        report(0.95, "finishing")
        notes = self.decode(probs, tuning)
        tempo = _estimate_tempo(probs, y, sr, audio["hop_length"]) if estimate_tempo else None
        if tempo is not None:
            # The bar grid (MusicXML, editor) needs a much finer tempo than the estimate.
            tempo = round(refine_tempo(notes, tempo), 2)
        return TranscriptionResult(
            notes=notes, duration=len(y) / sr, tempo=tempo, tuning=list(tuning or self.tuning)
        )


def _load(path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. Train one with `python scripts/train.py`."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model, ckpt["config"]


def _estimate_tempo(
    probs: dict[str, np.ndarray], y: np.ndarray, sample_rate: int, hop_length: int
) -> float | None:
    """BPM from the model's onsets when available, else from the audio.

    Onset probabilities give one clean peak per note, a better periodicity signal than
    the audio's spectral flux: on the GuitarSet test split, 82% of estimates were within
    4% (allowing double/half-tempo errors) versus 70% with ``librosa.beat.beat_track``.
    """
    import librosa

    if "onset" in probs:
        envelope = probs["onset"].max(axis=1)
        tempo = librosa.feature.tempo(
            onset_envelope=envelope, sr=sample_rate, hop_length=hop_length
        )
    else:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sample_rate)
    value = float(np.atleast_1d(tempo)[0])
    return round(value, 1) if value > 0 else None


def main(argv: list[str] | None = None) -> None:
    from music_core.midi import write_midi
    from music_core.tab import tab_to_ascii

    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tab-checkpoint", type=Path, help="separate model whose tab head places the notes"
    )
    parser.add_argument(
        "--separate", action="store_true", help="song mode: isolate the guitar first (Demucs)"
    )
    parser.add_argument("--midi", type=Path, help="write a MIDI file")
    parser.add_argument("--tab", type=Path, help="write ASCII tablature")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    predictor = Predictor.from_checkpoint(
        args.checkpoint, device=args.device, tab_checkpoint=args.tab_checkpoint
    )
    separator = None
    if args.separate:
        from ml.inference.separation import GuitarSeparator

        separator = GuitarSeparator(device=args.device)
    result = predictor.transcribe(args.audio, separator=separator)
    tempo = f"{result.tempo:.1f} BPM" if result.tempo else "unknown tempo"
    print(f"{len(result.notes)} notes, {result.duration:.1f} s, {tempo}")
    if args.midi:
        write_midi(result.notes, args.midi, tempo=result.tempo or 120.0)
        print(f"MIDI written to {args.midi}")
    if args.tab:
        args.tab.write_text(tab_to_ascii(result.notes, result.tuning), encoding="utf-8")
        print(f"Tab written to {args.tab}")


if __name__ == "__main__":
    main()
