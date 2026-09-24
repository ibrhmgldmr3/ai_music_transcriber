"""YAML config loading with inheritance (``extends:``) and CLI overrides."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TUNING = [40, 45, 50, 55, 59, 64]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config. ``extends: other.yaml`` is resolved relative to the file."""
    cfg = _load_with_parents(Path(path), chain=())
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _load_with_parents(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in chain:
        cycle = " -> ".join(p.name for p in (*chain, resolved))
        raise ValueError(f"Config inheritance cycle: {cycle}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping")
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(_load_with_parents(path.parent / parent, (*chain, resolved)), cfg)
    return cfg


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Turn ``["training.lr=1e-4", "model.name=cnn"]`` into a nested dict."""
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid override {item!r}, expected KEY=VALUE")
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # PyYAML reads "1e-4" as a string
            try:
                value = float(value)
            except ValueError:
                pass
        *parents, leaf = key.strip().split(".")
        node = result
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return result


def feature_bins(cfg: dict[str, Any]) -> int:
    features = cfg["features"]
    return features["n_bins"] if features["type"] == "cqt" else features["n_mels"]


def num_pitches(cfg: dict[str, Any]) -> int:
    labels = cfg["labels"]
    return labels["max_midi"] - labels["min_midi"] + 1


def tab_shape(cfg: dict[str, Any]) -> tuple[int, int]:
    """(number of strings, classes per string). Class 0 = silent, class k = fret k - 1."""
    tab = cfg.get("tab") or {}
    tuning = tab.get("tuning") or DEFAULT_TUNING
    return len(tuning), int(tab.get("num_frets", 20)) + 2
