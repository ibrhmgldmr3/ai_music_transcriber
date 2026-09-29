"""Training entry point.

Usage (from the repository root)::

    python scripts/train.py --config ml/configs/guitar.yaml
    python scripts/train.py --config ml/configs/guitar.yaml --set training.epochs=20 model.name=cnn
    python scripts/train.py --config ml/configs/guitar.yaml --resume ml/checkpoints/guitar/last.pt

    # fine-tune: start from a trained model's weights with a fresh optimizer and schedule
    python scripts/train.py --init ml/checkpoints/guitar/best.pt --set training.lr=2e-4
"""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.config import load_config, num_pitches, parse_overrides, tab_shape
from ml.datasets.dataset import TranscriptionDataset
from ml.evaluation.metrics import frame_metrics, tab_metrics
from ml.models import build_model, resolve_device
from ml.preprocessing.augmentation import FeatureAugmenter
from ml.training.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint
from ml.training.losses import TranscriptionLoss

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    """Training loop with AMP, gradient clipping, per-epoch validation and callbacks."""

    def __init__(
        self,
        cfg: dict[str, Any],
        model: nn.Module,
        device: torch.device,
        callbacks: Iterable[Callback] = (),
    ):
        t = cfg["training"]
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t["epochs"]
        )
        self.criterion = TranscriptionLoss(
            t.get("loss_weights"),
            {"onset": t.get("onset_pos_weight"), "offset": t.get("offset_pos_weight")},
        )
        self.use_amp = bool(t.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = t.get("grad_clip")
        self.frame_threshold = cfg["inference"]["frame_threshold"]
        self.callbacks = list(callbacks)
        self.epoch = 0
        self.start_epoch = 1
        self.should_stop = False

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        steps = 0
        for batch in tqdm(loader, desc=f"epoch {self.epoch}", leave=False):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(batch["features"])
                loss, parts = self.criterion(outputs, batch)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            steps += 1
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for name, value in parts.items():
                totals[f"loss_{name}"] = totals.get(f"loss_{name}", 0.0) + value
        return {f"train_{k}": v / max(steps, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        loss_sum, steps = 0.0, 0
        frame_pred, frame_true, tab_pred, tab_true = [], [], [], []
        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.model(batch["features"])
            loss, _ = self.criterion(outputs, batch)
            loss_sum += float(loss)
            steps += 1

            n_pitches = outputs["frame"].shape[-1]
            probs = torch.sigmoid(outputs["frame"])
            frame_pred.append((probs >= self.frame_threshold).cpu().numpy().reshape(-1, n_pitches))
            frame_true.append(batch["frame"].cpu().numpy().reshape(-1, n_pitches) > 0.5)
            if "tab" in outputs and "tab" in batch:
                n_strings = outputs["tab"].shape[2]
                tab_pred.append(outputs["tab"].argmax(-1).cpu().numpy().reshape(-1, n_strings))
                tab_true.append(batch["tab"].cpu().numpy().reshape(-1, n_strings))

        logs = {"val_loss": loss_sum / max(steps, 1)}
        metrics = frame_metrics(np.concatenate(frame_pred), np.concatenate(frame_true))
        logs.update({f"val_frame_{k}": v for k, v in metrics.items()})
        if tab_pred:
            tuning = self.cfg["tab"]["tuning"]
            metrics = tab_metrics(
                np.concatenate(tab_pred),
                np.concatenate(tab_true),
                tuning=tuning,
                min_midi=self.cfg["labels"]["min_midi"],
                n_pitches=num_pitches(self.cfg),
            )
            logs.update({f"val_tab_{k}": v for k, v in metrics.items()})
        return logs

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.cfg["training"]["epochs"]
        for cb in self.callbacks:
            cb.on_train_begin(self)
        for epoch in range(self.start_epoch, epochs + 1):
            self.epoch = epoch
            logs: dict[str, float] = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
            logs.update(self.train_epoch(train_loader))
            logs.update(self.validate(val_loader))
            self.scheduler.step()
            summary = " ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "epoch")
            logger.info("epoch %d | %s", epoch, summary)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)
            if self.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break
        for cb in self.callbacks:
            cb.on_train_end(self)

    def save_checkpoint(self, path: Path, logs: dict[str, float] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "config": self.cfg,
                "metrics": dict(logs or {}),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info("Resumed from %s (next epoch %d)", path, self.start_epoch)

    def load_weights(self, path: Path) -> None:
        """Model weights only (fine-tuning); optimizer, schedule and epoch start fresh.

        When this model covers more low pitches than the checkpoint, the pitch heads'
        rows are matched by pitch and the new bottom rows start as copies of the old
        lowest one.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        shift = ckpt["config"]["labels"]["min_midi"] - self.cfg["labels"]["min_midi"]
        state = self.model.state_dict()
        for name, value in ckpt["model"].items():
            target = state[name]
            if value.shape == target.shape:
                state[name] = value
            elif (
                shift > 0
                and value.shape[1:] == target.shape[1:]
                and (target.shape[0] == value.shape[0] + shift)
            ):
                grown = torch.cat([value[:1].expand(shift, *value.shape[1:]), value])
                state[name] = grown.clone()
            else:
                raise ValueError(
                    f"{name}: checkpoint {tuple(value.shape)}, model {tuple(target.shape)}"
                )
        self.model.load_state_dict(state)
        logger.info("Initialized weights from %s (%d new low pitches)", path, max(shift, 0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a transcription model.")
    parser.add_argument("--config", default="ml/configs/guitar.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. training.lr=1e-4 model.rnn_type=gru",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--resume", type=Path, help="checkpoint to resume from")
    parser.add_argument(
        "--init", type=Path, help="start from this checkpoint's weights (fine-tuning)"
    )
    args = parser.parse_args(argv)
    if args.resume and args.init:
        parser.error("--resume and --init are mutually exclusive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, parse_overrides(args.overrides))
    set_seed(cfg.get("seed", 42))
    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    paths, t = cfg["paths"], cfg["training"]
    processed_dir, splits_dir = Path(paths["processed_dir"]), Path(paths["splits_dir"])
    pitch_range = (
        num_pitches(cfg),
        cfg["labels"]["min_midi"],
        cfg["audio"]["sample_rate"],
        cfg["audio"]["hop_length"],
    )
    train_ds = TranscriptionDataset.from_split(
        processed_dir,
        splits_dir / "train.txt",
        segment_frames=t["segment_frames"],
        augmenter=FeatureAugmenter.from_config(cfg.get("augmentation")),
        repeats=t.get("repeats", 1),
        preload=t.get("preload", True),
        pitch_range=pitch_range,
    )
    val_ds = TranscriptionDataset.from_split(
        processed_dir, splits_dir / "val.txt", pitch_range=pitch_range
    )
    logger.info("Train samples/epoch: %d, validation tracks: %d", len(train_ds), len(val_ds))

    num_workers = t.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) >= t["batch_size"],
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=1)  # whole tracks of different lengths

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model %s: %.2fM parameters, tab shape %s",
        cfg["model"]["name"],
        n_params / 1e6,
        tab_shape(cfg),
    )

    ckpt_dir = Path(paths["checkpoint_dir"])
    monitor, mode = t.get("monitor", "val_frame_f1"), t.get("monitor_mode", "max")
    trainer = Trainer(
        cfg,
        model,
        device,
        callbacks=[
            ModelCheckpoint(ckpt_dir, monitor, mode),
            EarlyStopping(monitor, mode, patience=t.get("early_stopping_patience", 15)),
            CSVLogger(ckpt_dir / "history.csv"),
        ],
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.init:
        trainer.load_weights(args.init)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
