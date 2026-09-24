"""MusicXML export: standard notation plus guitar tablature, quantized to a beat grid.

The score has one guitar part with two staves: treble clef (sounding an octave lower,
as guitar music is written) and a six-line TAB staff carrying each note's string and
fret. MuseScore, Guitar Pro (File > Import) and most notation programs open it.

Rhythm: onsets are snapped to a 16th-note grid at the given tempo in 4/4. Each event
(a note or chord) lasts until its notes end or the next event starts, whichever comes
first; gaps become rests and notes crossing a barline are tied.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from music_core.notes import Note, sort_notes
from music_core.tab import STANDARD_TUNING

DIVISIONS = 4  # grid units per quarter note -> 16th-note resolution
BEATS_PER_MEASURE = 4
MEASURE_UNITS = DIVISIONS * BEATS_PER_MEASURE
GUITAR_MIDI_PROGRAM = 26  # 1-based General MIDI "Acoustic Guitar (steel)"

# Grid units -> (MusicXML note type, dots); largest first for greedy decomposition.
_NOTE_VALUES = [
    (16, "whole", 0),
    (12, "half", 1),
    (8, "half", 0),
    (6, "quarter", 1),
    (4, "quarter", 0),
    (3, "eighth", 1),
    (2, "eighth", 0),
    (1, "16th", 0),
]
_SPELLING = [
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
]  # fmt: skip

DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


@dataclass
class _Event:
    start: int  # grid units from the beginning
    length: int  # grid units
    notes: list[Note]  # empty for a rest


def notes_to_musicxml(
    notes: Sequence[Note],
    tempo: float = 120.0,
    tuning: Sequence[int] = STANDARD_TUNING,
    title: str = "Transcription",
) -> bytes:
    """Build a MusicXML 4.0 document (UTF-8 bytes) from positioned notes."""
    if not (math.isfinite(tempo) and tempo > 0):
        tempo = 120.0
    events = _quantize(notes, tempo)
    total_units = events[-1].start + events[-1].length if events else 0
    n_measures = max(1, math.ceil(total_units / MEASURE_UNITS))
    measures: list[list[tuple[int, list[Note], bool, bool]]] = [[] for _ in range(n_measures)]
    for event in events:
        pieces = list(_pieces(event.start, event.length))
        for k, (measure, units) in enumerate(pieces):
            tie_stop, tie_start = k > 0, k < len(pieces) - 1
            measures[measure].append((units, event.notes, tie_start, tie_stop))
    for content in measures:  # pad the last measure with rests
        filled = sum(units for units, *_ in content)
        for units in _decompose(MEASURE_UNITS - filled):
            content.append((units, [], False, False))

    root = ET.Element("score-partwise", version="4.0")
    ET.SubElement(ET.SubElement(root, "work"), "work-title").text = title
    encoding = ET.SubElement(ET.SubElement(root, "identification"), "encoding")
    ET.SubElement(encoding, "software").text = "music-transcriber"
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Guitar"
    instrument = ET.SubElement(score_part, "score-instrument", id="P1-I1")
    ET.SubElement(instrument, "instrument-name").text = "Guitar"
    midi = ET.SubElement(score_part, "midi-instrument", id="P1-I1")
    ET.SubElement(midi, "midi-program").text = str(GUITAR_MIDI_PROGRAM)

    part = ET.SubElement(root, "part", id="P1")
    for index, content in enumerate(measures):
        measure = ET.SubElement(part, "measure", number=str(index + 1))
        if index == 0:
            _write_attributes(measure, tuning)
            _write_tempo(measure, tempo)
        for staff, voice in ((1, "1"), (2, "5")):
            if staff == 2:
                ET.SubElement(ET.SubElement(measure, "backup"), "duration").text = str(
                    MEASURE_UNITS
                )
            for units, event_notes, tie_start, tie_stop in content:
                _write_event(measure, units, event_notes, tie_start, tie_stop, staff, voice, tuning)

    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{DOCTYPE}\n{body}\n'.encode()


def _quantize(notes: Sequence[Note], tempo: float) -> list[_Event]:
    """Snap onsets to the grid and build a gap-free sequence of note/chord/rest events."""
    unit = 60.0 / tempo / DIVISIONS  # seconds per grid unit
    cells: dict[int, list[Note]] = {}
    for note in sort_notes(n for n in notes if n.end > n.start):
        cells.setdefault(round(note.start / unit), []).append(note)

    starts = sorted(cells)
    events: list[_Event] = []
    cursor = 0
    for i, start in enumerate(starts):
        members = _one_per_string(cells[start])
        end = max(start + 1, round(max(n.end for n in members) / unit))
        if i + 1 < len(starts):
            end = min(end, starts[i + 1])
        if start > cursor:
            events.append(_Event(cursor, start - cursor, []))
        events.append(_Event(start, end - start, sorted(members, key=lambda n: n.pitch)))
        cursor = end
    return events


def _one_per_string(notes: list[Note]) -> list[Note]:
    """Notes merged into one grid cell may collide; keep one per string / unplaced pitch."""
    seen: dict[tuple[str, int], Note] = {}
    for note in notes:
        key = ("string", note.string) if note.string is not None else ("pitch", note.pitch)
        seen.setdefault(key, note)
    return list(seen.values())


def _pieces(start: int, length: int) -> Iterator[tuple[int, int]]:
    """Split a span at barlines and into writable note values: (measure, units)."""
    position, remaining = start, length
    while remaining > 0:
        chunk = min(remaining, MEASURE_UNITS - position % MEASURE_UNITS)
        for units in _decompose(chunk):
            yield position // MEASURE_UNITS, units
            position += units
        remaining -= chunk


def _decompose(units: int) -> list[int]:
    parts = []
    for value, _, _ in _NOTE_VALUES:
        while units >= value:
            parts.append(value)
            units -= value
    return parts


def _spell(midi: int) -> tuple[str, int, int]:
    step, alter = _SPELLING[midi % 12]
    return step, alter, midi // 12 - 1


def _write_attributes(measure: ET.Element, tuning: Sequence[int]) -> None:
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = str(DIVISIONS)
    ET.SubElement(ET.SubElement(attributes, "key"), "fifths").text = "0"
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = str(BEATS_PER_MEASURE)
    ET.SubElement(time, "beat-type").text = "4"
    ET.SubElement(attributes, "staves").text = "2"
    treble = ET.SubElement(attributes, "clef", number="1")
    ET.SubElement(treble, "sign").text = "G"
    ET.SubElement(treble, "line").text = "2"
    ET.SubElement(treble, "clef-octave-change").text = "-1"  # guitar sounds an octave lower
    tab = ET.SubElement(attributes, "clef", number="2")
    ET.SubElement(tab, "sign").text = "TAB"
    ET.SubElement(tab, "line").text = "5"
    details = ET.SubElement(attributes, "staff-details", number="2")
    ET.SubElement(details, "staff-lines").text = str(len(tuning))
    for line, open_pitch in enumerate(tuning, start=1):  # line 1 = lowest string
        step, alter, octave = _spell(open_pitch)
        staff_tuning = ET.SubElement(details, "staff-tuning", line=str(line))
        ET.SubElement(staff_tuning, "tuning-step").text = step
        if alter:
            ET.SubElement(staff_tuning, "tuning-alter").text = str(alter)
        ET.SubElement(staff_tuning, "tuning-octave").text = str(octave)


def _write_tempo(measure: ET.Element, tempo: float) -> None:
    direction = ET.SubElement(measure, "direction", placement="above")
    metronome = ET.SubElement(ET.SubElement(direction, "direction-type"), "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(round(tempo))
    ET.SubElement(direction, "staff").text = "1"
    ET.SubElement(direction, "sound", tempo=f"{tempo:.2f}")


def _write_event(
    measure: ET.Element,
    units: int,
    notes: list[Note],
    tie_start: bool,
    tie_stop: bool,
    staff: int,
    voice: str,
    tuning: Sequence[int],
) -> None:
    value_type, dots = next((t, d) for u, t, d in _NOTE_VALUES if u == units)
    if not notes:
        note = ET.SubElement(measure, "note")
        rest = ET.SubElement(note, "rest")
        if units == MEASURE_UNITS:
            rest.set("measure", "yes")
        _write_timing(note, units, voice, value_type, dots, staff)
        return

    for index, source in enumerate(notes):
        note = ET.SubElement(measure, "note")
        if index > 0:
            ET.SubElement(note, "chord")
        step, alter, octave = _spell(source.pitch)
        pitch = ET.SubElement(note, "pitch")
        ET.SubElement(pitch, "step").text = step
        if alter:
            ET.SubElement(pitch, "alter").text = str(alter)
        ET.SubElement(pitch, "octave").text = str(octave)
        ET.SubElement(note, "duration").text = str(units)
        ties = [kind for kind, on in (("stop", tie_stop), ("start", tie_start)) if on]
        for kind in ties:
            ET.SubElement(note, "tie", type=kind)
        ET.SubElement(note, "voice").text = voice
        ET.SubElement(note, "type").text = value_type
        for _ in range(dots):
            ET.SubElement(note, "dot")
        ET.SubElement(note, "staff").text = str(staff)

        positioned = staff == 2 and source.string is not None and source.fret is not None
        if ties or positioned:
            notations = ET.SubElement(note, "notations")
            for kind in ties:
                ET.SubElement(notations, "tied", type=kind)
            if positioned:
                technical = ET.SubElement(notations, "technical")
                # MusicXML numbers strings from the highest (1) down.
                ET.SubElement(technical, "string").text = str(len(tuning) - source.string)
                ET.SubElement(technical, "fret").text = str(source.fret)


def _write_timing(
    note: ET.Element, units: int, voice: str, value_type: str, dots: int, staff: int
) -> None:
    ET.SubElement(note, "duration").text = str(units)
    ET.SubElement(note, "voice").text = voice
    ET.SubElement(note, "type").text = value_type
    for _ in range(dots):
        ET.SubElement(note, "dot")
    ET.SubElement(note, "staff").text = str(staff)
