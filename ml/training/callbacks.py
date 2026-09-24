"""Training callbacks: checkpointing, early stopping and CSV logging."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.training.train import Trainer

logger = logging.getLogger(__name__)


def _improved(current: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    return current > best + min_delta if mode == "max" else current < best - min_delta


class Callback:
    """Base class; override the hooks you need."""

    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass


class ModelCheckpoint(Callback):
    """Saves ``last.pt`` every epoch and ``best.pt`` whenever the monitored metric improves."""

    def __init__(self, directory: str | Path, monitor: str = "val_frame_f1", mode: str = "max"):
        self.directory = Path(directory)
        self.monitor = monitor
        self.mode = mode
        self.best: float | None = None

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        trainer.save_checkpoint(self.directory / "last.pt", logs)
        current = logs.get(self.monitor)
        if current is None:
            logger.warning("ModelCheckpoint: metric %r not found in logs", self.monitor)
            return
        if _improved(current, self.best, self.mode):
            self.best = current
            trainer.save_checkpoint(self.directory / "best.pt", logs)
            logger.info("New best %s=%.4f saved to %s", self.monitor, current, self.directory)


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "val_frame_f1",
        mode: str = "max",
        patience: int = 15,
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.wait = 0

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
        if _improved(current, self.best, self.mode, self.min_delta):
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True


class CSVLogger(Callback):
    """Appends one row of metrics per epoch (history.csv)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fields: list[str] | None = None

    def on_train_begin(self, trainer: Trainer) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if trainer.start_epoch == 1 and self.path.exists():
            self.path.unlink()  # fresh run: start a new history

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        if self._fields is None:
            self._fields = list(logs)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(logs)
