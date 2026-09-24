"""Plotting helpers for piano rolls, tablature and training curves (matplotlib)."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from music_core.notes import midi_to_name


def plot_pianoroll(
    pred: np.ndarray,
    target: np.ndarray | None = None,
    min_midi: int = 40,
    frame_rate: float | None = None,
    title: str | None = None,
) -> Figure:
    """Boolean ``(T, P)`` roll. With a target: green = correct, red = false alarm, blue = missed."""
    pred = pred.astype(bool)
    n_frames, n_pitches = pred.shape
    image = np.zeros((n_pitches, n_frames, 3))
    if target is None:
        image[pred.T] = (0.95, 0.95, 0.95)
    else:
        target = target.astype(bool)
        image[(pred & target).T] = (0.2, 0.75, 0.3)
        image[(pred & ~target).T] = (0.9, 0.25, 0.25)
        image[(~pred & target).T] = (0.25, 0.45, 0.95)

    fig, ax = plt.subplots(figsize=(14, 5))
    extent_x = n_frames / frame_rate if frame_rate else n_frames
    ax.imshow(image, origin="lower", aspect="auto", extent=(0, extent_x, 0, n_pitches))
    ticks = [p for p in range(n_pitches) if (p + min_midi) % 12 == 0]
    ax.set_yticks([t + 0.5 for t in ticks], [midi_to_name(t + min_midi) for t in ticks])
    ax.set_xlabel("time (s)" if frame_rate else "frame")
    ax.set_ylabel("pitch")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_tab(
    tab: np.ndarray,
    string_names: Sequence[str] = ("E", "A", "D", "G", "B", "e"),
    frame_rate: float | None = None,
    title: str | None = None,
) -> Figure:
    """``(T, strings)`` tab classes as fret numbers per string (highest string on top)."""
    n_frames, n_strings = tab.shape
    frets = np.ma.masked_less_equal(tab.T.astype(float), 0) - 1
    fig, ax = plt.subplots(figsize=(14, 3))
    extent_x = n_frames / frame_rate if frame_rate else n_frames
    im = ax.imshow(
        frets,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=(0, extent_x, -0.5, n_strings - 0.5),
    )
    ax.set_yticks(range(n_strings), list(string_names)[:n_strings])
    ax.set_xlabel("time (s)" if frame_rate else "frame")
    fig.colorbar(im, ax=ax, label="fret")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_training_history(
    csv_path: str | Path,
    metrics: Sequence[str] = ("train_loss", "val_loss", "val_frame_f1"),
) -> Figure:
    with Path(csv_path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(float(r["epoch"])) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    for metric in metrics:
        if rows and metric in rows[0]:
            ax.plot(epochs, [float(r[metric]) for r in rows], label=metric)
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 120) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
