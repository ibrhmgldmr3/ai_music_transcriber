"""Annotation parsing (GuitarSet JAMS) and frame-level training targets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from music_core.notes import Note, sort_notes
from music_core.tab import STANDARD_TUNING

NOTE_ARRAY_COLUMNS = ("pitch", "start", "end", "velocity", "string", "fret")


def load_guitarset_jams(path: str | Path, tuning: Sequence[int] = STANDARD_TUNING) -> list[Note]:
    """Read the per-string ``note_midi`` annotations of a GuitarSet JAMS file.

    JAMS files are plain JSON, so no ``jams`` dependency is needed. Each of the six
    ``note_midi`` annotations belongs to one string (``annotation_metadata.data_source``).
    """
    with open(path, encoding="utf-8") as f:
        jam = json.load(f)

    note_annotations = [a for a in jam["annotations"] if a.get("namespace") == "note_midi"]
    notes: list[Note] = []
    for index, annotation in enumerate(note_annotations):
        source = annotation.get("annotation_metadata", {}).get("data_source")
        try:
            string = int(source)
        except (TypeError, ValueError):
            string = index
        for obs in _observations(annotation["data"]):
            start = float(obs["time"])
            end = start + float(obs["duration"])
            if end <= start:
                continue
            pitch = int(round(float(obs["value"])))
            fret = pitch - tuning[string] if 0 <= string < len(tuning) else -1
            notes.append(
                Note(
                    pitch=pitch,
                    start=start,
                    end=end,
                    string=string if fret >= 0 else None,
                    fret=fret if fret >= 0 else None,
                )
            )
    return sort_notes(notes)


def _observations(data: Any) -> list[dict[str, Any]]:
    # JAMS stores observations either as a list of records or as column arrays.
    if isinstance(data, dict):
        return [
            {"time": t, "duration": d, "value": v}
            for t, d, v in zip(data["time"], data["duration"], data["value"])
        ]
    return data


def notes_to_targets(
    notes: Sequence[Note],
    n_frames: int,
    sample_rate: int,
    hop_length: int,
    min_midi: int,
    max_midi: int,
    num_strings: int = 6,
    num_frets: int = 20,
    onset_frames: int = 1,
    offset_frames: int = 1,
) -> dict[str, np.ndarray]:
    """Rasterize notes into frame-level targets.

    Returns ``onset``, ``frame`` and ``offset`` piano rolls of shape ``(n_frames, n_pitches)``
    and ``tab`` of shape ``(n_frames, num_strings)`` where 0 = silent and k = fret k - 1.
    The offset target sits on the first frame after the note (its end time).
    """
    n_pitches = max_midi - min_midi + 1
    frame = np.zeros((n_frames, n_pitches), dtype=np.uint8)
    onset = np.zeros_like(frame)
    offset = np.zeros_like(frame)
    tab = np.zeros((n_frames, num_strings), dtype=np.int8)
    fps = sample_rate / hop_length

    for note in notes:
        p = note.pitch - min_midi
        start = int(round(note.start * fps))
        if not 0 <= p < n_pitches or start >= n_frames:
            continue
        end = min(n_frames, max(start + 1, int(round(note.end * fps))))
        frame[start:end, p] = 1
        onset[start : min(start + onset_frames, n_frames), p] = 1
        if end < n_frames:
            offset[end : min(end + offset_frames, n_frames), p] = 1
        if (
            note.string is not None
            and note.fret is not None
            and 0 <= note.string < num_strings
            and 0 <= note.fret <= num_frets
        ):
            tab[start:end, note.string] = note.fret + 1
    return {"onset": onset, "frame": frame, "offset": offset, "tab": tab}


def notes_to_array(notes: Sequence[Note]) -> np.ndarray:
    """Pack notes into a float array ``(N, 6)`` (see ``NOTE_ARRAY_COLUMNS``); None -> -1."""
    rows = [
        [
            n.pitch,
            n.start,
            n.end,
            n.velocity,
            -1 if n.string is None else n.string,
            -1 if n.fret is None else n.fret,
        ]
        for n in notes
    ]
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(NOTE_ARRAY_COLUMNS))


def array_to_notes(array: np.ndarray) -> list[Note]:
    return [
        Note(
            pitch=int(pitch),
            start=float(start),
            end=float(end),
            velocity=int(velocity),
            string=None if string < 0 else int(string),
            fret=None if fret < 0 else int(fret),
        )
        for pitch, start, end, velocity, string, fret in np.asarray(array).reshape(-1, 6)
    ]
