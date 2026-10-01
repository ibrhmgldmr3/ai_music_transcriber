"""MIDI import / export (via ``mido``)."""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import BinaryIO

import mido

from music_core.analysis import Key
from music_core.notes import Note, sort_notes
from music_core.timing import BeatGrid

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
    key: Key | None = None,
    beats_per_measure: int = 4,
    grid: BeatGrid | None = None,
) -> mido.MidiFile:
    """Build a single-track MIDI file from notes (times in seconds).

    Note times are kept as they are, so the file lines up with the recording; the key
    and time signature are informational. With a ``grid`` (e.g. tracked beats), a tempo
    map makes the file's beats and bars fall on the grid's: whatever comes before the
    first bar line is a pickup measure of whole beats.
    """
    if not (math.isfinite(tempo) and tempo > 0):
        tempo = 120.0  # an unusable tempo estimate must not break the export
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    if grid is None:
        to_ticks = lambda seconds: _seconds_to_ticks(seconds, tempo)  # noqa: E731
        meta = [
            (0, _set_tempo(tempo)),
            (0, mido.MetaMessage("time_signature", numerator=beats_per_measure, denominator=4)),
        ]
    else:
        to_ticks, meta = _tempo_map(grid)
    if key is not None:
        name = key.tonic_name + ("m" if key.mode == "minor" else "")
        meta.append((0, mido.MetaMessage("key_signature", key=name)))
    meta.append((0, mido.Message("program_change", program=program, channel=channel)))

    # (tick, order, message): meta (0) and note_off (1) sort before note_on (2); the
    # sort is stable, so events of one kind keep their order.
    events: list[tuple[int, int, mido.Message]] = [(tick, 0, message) for tick, message in meta]
    for note in notes:
        if note.end <= note.start:
            continue
        pitch = min(max(int(note.pitch), 0), 127)
        velocity = min(max(int(note.velocity), 1), 127)
        on = to_ticks(note.start)
        off = max(on + 1, to_ticks(note.end))
        events.append(
            (on, 2, mido.Message("note_on", note=pitch, velocity=velocity, channel=channel))
        )
        events.append((off, 1, mido.Message("note_off", note=pitch, velocity=0, channel=channel)))
    events.sort(key=lambda e: (e[0], e[1]))

    last_tick = 0
    for tick, _, message in events:
        track.append(message.copy(time=tick - last_tick))
        last_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def _tempo_map(grid: BeatGrid) -> tuple[Callable[[float], int], list[tuple[int, mido.Message]]]:
    """Seconds -> ticks along ``grid``, and its tempo and time signature events.

    Tick 0 stays 0 s. The time before the first bar line at or after 0 s becomes a pickup
    of whole beats (at least one) at its own tempo; from that bar line on, every beat of
    the grid is one MIDI beat at the beat's tempo.
    """
    n = grid.beats_per_measure
    start = float(grid.position(0.0))
    first_bar = next(
        i for i in range(math.ceil(start - 1e-6), math.ceil(start) + n + 1) if grid.is_bar_line(i)
    )
    bar_time = float(grid.time(first_bar))
    span = first_bar - start  # beats before the first bar line
    pickup = 0 if span < 1e-3 else max(1, round(span))
    events: list[tuple[int, mido.Message]] = []
    if pickup:
        events.append((0, _set_tempo(60.0 * pickup / bar_time)))
        if pickup != n:
            events.append((0, mido.MetaMessage("time_signature", numerator=pickup, denominator=4)))
    offset = pickup * TICKS_PER_BEAT
    events.append((offset, mido.MetaMessage("time_signature", numerator=n, denominator=4)))
    last_bpm = None
    for index in range(first_bar, len(grid.beats)):
        bpm = grid.tempo_at(index)
        if last_bpm is None or abs(bpm - last_bpm) > 0.05:
            events.append((offset + (index - first_bar) * TICKS_PER_BEAT, _set_tempo(bpm)))
            last_bpm = bpm

    def to_ticks(seconds: float) -> int:
        if seconds < bar_time and pickup:
            return int(round(seconds / bar_time * offset))
        return offset + int(round((float(grid.position(seconds)) - first_bar) * TICKS_PER_BEAT))

    return to_ticks, events


def _set_tempo(bpm: float) -> mido.MetaMessage:
    return mido.MetaMessage("set_tempo", tempo=int(round(mido.bpm2tempo(bpm))))


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
    key: Key | None = None,
    beats_per_measure: int = 4,
    grid: BeatGrid | None = None,
) -> bytes:
    buffer = io.BytesIO()
    midi = notes_to_midi(
        notes,
        tempo=tempo,
        program=program,
        key=key,
        beats_per_measure=beats_per_measure,
        grid=grid,
    )
    midi.save(file=buffer)
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
