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


def load_string_midi(
    path: str | Path,
    tuning: Sequence[int] = STANDARD_TUNING,
    string_from: str = "channel",
) -> list[Note]:
    """Per-string MIDI from a hexaphonic / MIDI pickup (EGDB, Guitar-TECHS).

    Strings are numbered as guitarists do, 1 = highest: by MIDI channel (0-5) or by
    track name ("1" ... "6", or "e", "B", "G", "D", "A", "E"), ``string_from`` =
    "channel" or "track". A note whose pitch can't be played on its string keeps no
    position rather than a wrong one.
    """
    import mido

    midi = mido.MidiFile(str(path))
    seconds = _tick_to_seconds(midi)
    notes: list[Note] = []
    for track in midi.tracks:
        track_string = _string_number(track.name)
        tick = 0
        active: dict[tuple[int, int], tuple[int, int]] = {}  # (channel, pitch) -> (tick, vel)
        for message in track:
            tick += message.time
            if message.type not in ("note_on", "note_off"):
                continue
            key = (message.channel, message.note)
            started = active.pop(key, None)
            if started is not None and tick > started[0]:
                number = track_string if string_from == "track" else message.channel + 1
                notes.append(
                    _string_note(
                        message.note, seconds(started[0]), seconds(tick), started[1], number, tuning
                    )
                )
            if message.type == "note_on" and message.velocity > 0:
                active[key] = (tick, message.velocity)
    return sort_notes(notes)


_STRING_NAMES = ("e", "B", "G", "D", "A", "E")  # standard tuning, string 1 first


def _string_number(track_name: str) -> int | None:
    name = track_name.strip()
    if name.isdigit():
        return int(name)
    return _STRING_NAMES.index(name) + 1 if name in _STRING_NAMES else None


def _string_note(
    pitch: int, start: float, end: float, velocity: int, number: int | None, tuning: Sequence[int]
) -> Note:
    string = len(tuning) - number if number is not None else -1
    fret = pitch - tuning[string] if 0 <= string < len(tuning) else -1
    placed = fret >= 0
    return Note(
        pitch=pitch,
        start=start,
        end=end,
        velocity=velocity,
        string=string if placed else None,
        fret=fret if placed else None,
    )


def _tick_to_seconds(midi: Any):
    """Absolute tick -> seconds under the file's tempo map (tempo events may be on any track)."""
    import mido

    changes: list[tuple[int, int]] = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                changes.append((tick, message.tempo))
    changes.sort()
    # Seconds elapsed at each tempo change, starting from 120 BPM (MIDI default).
    points: list[tuple[int, float, int]] = [(0, 0.0, 500000)]
    for tick, tempo in changes:
        last_tick, last_seconds, last_tempo = points[-1]
        elapsed = mido.tick2second(tick - last_tick, midi.ticks_per_beat, last_tempo)
        points.append((tick, last_seconds + elapsed, tempo))

    def seconds(tick: int) -> float:
        base = next(p for p in reversed(points) if p[0] <= tick)
        return base[1] + mido.tick2second(tick - base[0], midi.ticks_per_beat, base[2])

    return seconds


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
