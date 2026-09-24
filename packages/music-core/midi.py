"""MIDI import / export (via ``mido``)."""

from __future__ import annotations

import io
import math
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import mido

from music_core.notes import Note, sort_notes

TICKS_PER_BEAT = 480
GUITAR_PROGRAM = 25  # General MIDI "Acoustic Guitar (steel)", 0-based
DRUM_CHANNEL = 9


def _seconds_to_ticks(seconds: float, tempo: float) -> int:
    return int(round(seconds * tempo / 60.0 * TICKS_PER_BEAT))


def notes_to_midi(
    notes: Iterable[Note],
    tempo: float = 120.0,
    program: int = GUITAR_PROGRAM,
    channel: int = 0,
) -> mido.MidiFile:
    """Build a single-track MIDI file from notes (times in seconds)."""
    if not (math.isfinite(tempo) and tempo > 0):
        tempo = 120.0  # an unusable tempo estimate must not break the export
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=int(mido.bpm2tempo(tempo)), time=0))
    track.append(mido.Message("program_change", program=program, channel=channel, time=0))

    # (tick, order, message): note_off (0) sorts before note_on (1) on the same tick.
    events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        if note.end <= note.start:
            continue
        pitch = min(max(int(note.pitch), 0), 127)
        velocity = min(max(int(note.velocity), 1), 127)
        on = _seconds_to_ticks(note.start, tempo)
        off = max(on + 1, _seconds_to_ticks(note.end, tempo))
        events.append(
            (on, 1, mido.Message("note_on", note=pitch, velocity=velocity, channel=channel))
        )
        events.append((off, 0, mido.Message("note_off", note=pitch, velocity=0, channel=channel)))
    events.sort(key=lambda e: (e[0], e[1]))

    last_tick = 0
    for tick, _, message in events:
        track.append(message.copy(time=tick - last_tick))
        last_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def write_midi(
    notes: Iterable[Note],
    path: str | Path,
    tempo: float = 120.0,
    program: int = GUITAR_PROGRAM,
) -> None:
    notes_to_midi(notes, tempo=tempo, program=program).save(str(path))


def notes_to_midi_bytes(
    notes: Iterable[Note],
    tempo: float = 120.0,
    program: int = GUITAR_PROGRAM,
) -> bytes:
    buffer = io.BytesIO()
    notes_to_midi(notes, tempo=tempo, program=program).save(file=buffer)
    return buffer.getvalue()


def midi_to_notes(source: str | Path | BinaryIO) -> list[Note]:
    """Read all non-drum notes from a MIDI file; times are returned in seconds."""
    if isinstance(source, (str, Path)):
        mid = mido.MidiFile(filename=str(source))
    else:
        mid = mido.MidiFile(file=source)

    notes: list[Note] = []
    active: dict[tuple[int, int], tuple[float, int]] = {}  # (channel, pitch) -> (start, velocity)
    now = 0.0
    for message in mid:  # merged tracks; message.time is a delta in seconds
        now += message.time
        if message.type not in ("note_on", "note_off") or message.channel == DRUM_CHANNEL:
            continue
        key = (message.channel, message.note)
        started = active.pop(key, None)
        if started is not None and now > started[0]:
            notes.append(Note(message.note, started[0], now, started[1]))
        if message.type == "note_on" and message.velocity > 0:
            active[key] = (now, message.velocity)
    return sort_notes(notes)
