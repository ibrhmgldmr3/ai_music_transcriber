"""MIDI import / export (via ``mido``)."""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import BinaryIO

import mido

from music_core.analysis import Key
from music_core.notes import Note, sort_notes
from music_core.timing import BeatGrid

TICKS_PER_BEAT = 480
GUITAR_PROGRAM = 25  # General MIDI "Acoustic Guitar (steel)", 0-based
DRUM_CHANNEL = 9

# Playing techniques. A pitch bend moves a whole channel, so a file with techniques
# puts every string on a channel of its own (the "MIDI guitar" layout Guitar Pro also
# writes) with a bend range of an octave, wide enough for slides. The shapes are the
# editor's playback's: keep them in sync with apps/web/lib/techniques.ts.
BEND_RANGE = 12  # semitones a full pitch-bend reaches
BEND_DELAY = 0.04  # seconds after the pick before a bend starts (at most 1/5 of the note)
BEND_RISE = 0.1  # seconds to reach the bent pitch (at most 2/5 of the note)
VIBRATO_RATE = 5.5  # Hz
VIBRATO_DEPTH = 0.2  # semitones each way
VIBRATO_DELAY = 0.1  # seconds after the pick (or the bend) before the vibrato starts
SLIDE_GLIDE = 0.06  # seconds a slide takes, ending where the next note starts
LEGATO_VELOCITY = 0.7  # a hammer-on or pull-off against a picked note
PITCH_STEP = 0.01  # seconds between the pitch-bend messages of a curve


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

    Notes with playing techniques sound them: bends, vibrato and slides as pitch-bend
    curves, each string on a channel of its own; hammer-ons and pull-offs a little softer.
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

    notes = [n for n in notes if n.end > n.start]
    channels = _string_channels(notes) if any(_has_technique(n) for n in notes) else None
    if channels is None:
        meta.append((0, mido.Message("program_change", program=program, channel=channel)))
    else:
        for ch in sorted(set(channels.values())):
            meta.append((0, mido.Message("program_change", program=program, channel=ch)))
            meta.extend((0, message) for message in _bend_range(ch))

    # (tick, order, message): meta (0) and note_off (1) sort before a note's pitch reset
    # (2), note_on (3) and its pitch curve (4); the sort is stable, so events of one kind
    # keep their order.
    events: list[tuple[int, int, mido.Message]] = [(tick, 0, message) for tick, message in meta]
    played = _played(notes, channels, channel)
    for note, ch, end in played:
        pitch = min(max(int(note.pitch), 0), 127)
        velocity = min(max(int(note.velocity), 1), 127)
        if note.technique in ("hammer_on", "pull_off"):
            velocity = max(1, round(velocity * LEGATO_VELOCITY))  # no pick attack
        on = to_ticks(note.start)
        off = max(on + 1, to_ticks(end))
        events.append((on, 3, mido.Message("note_on", note=pitch, velocity=velocity, channel=ch)))
        events.append((off, 1, mido.Message("note_off", note=pitch, velocity=0, channel=ch)))
    if channels is not None:
        events.extend(_pitch_events(played, to_ticks))
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


def _has_technique(note: Note) -> bool:
    return bool(note.bend) or note.vibrato or note.technique is not None


def _string_channels(notes: Sequence[Note]) -> dict[int | None, int]:
    """Channel per string, lowest first; unplaced notes (key None) get the next one."""
    free = [c for c in range(16) if c != DRUM_CHANNEL]
    strings = sorted({n.string for n in notes if n.string is not None})[: len(free) - 1]
    mapping: dict[int | None, int] = {s: free[i] for i, s in enumerate(strings)}
    mapping[None] = free[len(strings)]
    return mapping


def _bend_range(channel: int) -> list[mido.Message]:
    """RPN 0 (pitch-bend sensitivity) set to BEND_RANGE semitones, then RPN null."""
    return [
        mido.Message("control_change", channel=channel, control=control, value=value)
        for control, value in ((101, 0), (100, 0), (6, BEND_RANGE), (38, 0), (101, 127), (100, 127))
    ]


def _played(
    notes: Sequence[Note], channels: dict[int | None, int] | None, channel: int = 0
) -> list[tuple[Note, int, float]]:
    """(note, channel, end) in onset order. With a channel per string a string sounds
    one note at a time: a note ends where the next on its string starts, and one that
    slides into the next lasts until it."""
    ordered = sort_notes(notes)
    if channels is None:
        return [(n, channel, n.end) for n in ordered]
    result = []
    last: dict[int, int] = {}  # channel -> index in result of its latest note
    for note in ordered:
        ch = channels.get(note.string, channels[None])
        if ch in last and note.string is not None:
            previous, prev_ch, end = result[last[ch]]
            if note.technique == "slide" or end > note.start:
                result[last[ch]] = (previous, prev_ch, max(previous.start, note.start))
        last[ch] = len(result)
        result.append((note, ch, note.end))
    return result


def _pitch_curve(note: Note, end: float, slide_to: Note | None) -> Callable[[float], float]:
    """Semitones off the note's pitch at time t: its bend, vibrato and slide out."""
    length = end - note.start
    bend_start = note.start + min(BEND_DELAY, length / 5)
    bend_end = bend_start + min(BEND_RISE, 2 * length / 5)
    vibrato_start = (bend_end if note.bend else note.start) + min(VIBRATO_DELAY, length / 5)
    glide = min(SLIDE_GLIDE, length / 2)

    def curve(t: float) -> float:
        value = 0.0
        if note.bend:
            value += note.bend * min(1.0, max(0.0, (t - bend_start) / (bend_end - bend_start)))
        if note.vibrato and t >= vibrato_start:
            value += VIBRATO_DEPTH * math.sin(2 * math.pi * VIBRATO_RATE * (t - vibrato_start))
        # The glide lands a step before the next note, so its last message is the target.
        landing = end - PITCH_STEP
        if slide_to is not None and t > landing - glide:
            target = slide_to.pitch - note.pitch - value
            value += target * min(1.0, (t - (landing - glide)) / glide)
        return value

    return curve


def _pitch_events(
    played: Sequence[tuple[Note, int, float]], to_ticks: Callable[[float], int]
) -> list[tuple[int, int, mido.Message]]:
    """Pitch-bend curves of the notes with techniques, reset to 0 as each note starts
    and ends."""
    events: list[tuple[int, int, mido.Message]] = []
    following: dict[int, Note] = {}  # index -> next note on the same channel
    last: dict[int, int] = {}
    for index, (note, ch, _) in enumerate(played):
        if ch in last:
            following[last[ch]] = note
        last[ch] = index
    for index, (note, ch, end) in enumerate(played):
        nxt = following.get(index)
        slide_to = (
            nxt
            if nxt is not None and nxt.technique == "slide" and nxt.start <= end + 1e-6
            else None
        )
        events.append((to_ticks(note.start), 2, _bend(ch, 0.0)))
        if not (note.bend or note.vibrato or slide_to):
            continue
        curve = _pitch_curve(note, end, slide_to)
        sent = 0
        steps = max(1, int((end - note.start) / PITCH_STEP))
        for k in range(1, steps):  # the end belongs to the next note on the channel
            t = note.start + k * (end - note.start) / steps
            value = _wheel(curve(t))
            if value != sent:
                events.append((to_ticks(t), 4, _bend(ch, curve(t))))
                sent = value
        events.append((max(to_ticks(note.start) + 1, to_ticks(end)), 2, _bend(ch, 0.0)))
    return events


def _wheel(semitones: float) -> int:
    return max(-8192, min(8191, int(round(semitones / BEND_RANGE * 8191))))


def _bend(channel: int, semitones: float) -> mido.Message:
    return mido.Message("pitchwheel", channel=channel, pitch=_wheel(semitones))


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
