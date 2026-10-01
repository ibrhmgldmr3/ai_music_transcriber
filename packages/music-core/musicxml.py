"""MusicXML export: standard notation plus guitar tablature, quantized to a beat grid.

The score has one guitar part with two staves: treble clef (sounding an octave lower,
as guitar music is written) and a six-line TAB staff carrying each note's string and
fret. MuseScore, Guitar Pro (File > Import) and most notation programs open it.

Rhythm: onsets are snapped to 16th notes of the beat grid (``music_core.analysis``:
one tempo, or beats tracked in the recording), with bar lines where the analysis puts
them (or at the given downbeat). Each event (a note or chord) lasts until its notes end
or the next event starts, whichever comes first; gaps become rests and notes crossing a
barline are tied. When the tracked beats speed up or slow down, measures carry the new
tempo for playback (and a visible mark when it moved more than 8 %). The key signature,
note spelling and chord symbols come from the same analysis, or from the song's keys.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from music_core.analysis import Chord, Key, KeySpan, analyze, spell
from music_core.guitar_chords import QUALITIES
from music_core.notes import Note, sort_notes
from music_core.tab import STANDARD_TUNING
from music_core.timing import BeatGrid

DIVISIONS = 4  # grid units per quarter note -> 16th-note resolution
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
_HARMONY_KINDS = {
    "maj": "major",
    "min": "minor",
    "7": "dominant",
    "maj7": "major-seventh",
    "min7": "minor-seventh",
    "5": "power",
    "sus2": "suspended-second",
    "sus4": "suspended-fourth",
    "dim": "diminished",
    "maj6": "major-sixth",
    "min6": "minor-sixth",
    "dim7": "diminished-seventh",
    "hdim7": "half-diminished",
    "aug": "augmented",
    "minmaj7": "major-minor",
}

DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


@dataclass
class _Event:
    start: int  # grid units from the first bar line
    length: int  # grid units
    notes: list[Note]  # empty for a rest


def notes_to_musicxml(
    notes: Sequence[Note],
    tempo: float = 120.0,
    tuning: Sequence[int] = STANDARD_TUNING,
    title: str = "Transcription",
    *,
    beats_per_measure: int = 4,
    key: Key | None = None,
    downbeat: float | None = None,
    capo: int = 0,
    chords: Sequence[Chord] | None = None,
    beats: Sequence[float] | None = None,
    downbeats: Sequence[float] | None = None,
    keys: Sequence[KeySpan] | None = None,
    chord_scores: Sequence[dict] | None = None,
) -> bytes:
    """Build a MusicXML 4.0 document (UTF-8 bytes) from positioned notes.

    ``key`` and ``downbeat`` (the time of any bar line, seconds) override the estimates,
    ``chords`` the chord symbols found in the notes (e.g. chords recognized in a song).
    ``beats`` / ``downbeats`` (tracked in the recording) replace the single ``tempo``.
    ``keys`` (a song's keys over time) change the key signature where they change,
    unless ``key`` is given; ``chord_scores`` are fused with the notes' chords as in
    ``analyze``. With a capo, ``tuning`` includes it, so the TAB staff's frets count
    from the capo (the way it is played) in any program, and "Capo N" is written above
    the first bar.
    """
    if not (math.isfinite(tempo) and tempo > 0):
        tempo = 120.0
    notes = [n for n in notes if n.end > n.start]
    symbols_end = max((c.end for c in chords or []), default=None)
    analysis = analyze(
        notes, tempo, beats_per_measure, key=key, downbeat=downbeat,
        beats=beats, downbeats=downbeats, end=symbols_end, keys=keys,
        chord_scores=chord_scores if chords is None else None,
    )  # fmt: skip
    grid = analysis.grid
    measure_units = DIVISIONS * beats_per_measure
    origin = first_bar_line(notes, grid)  # beat index of measure 1's bar line

    def to_units(time: float) -> int:
        """Grid units (16ths) from measure 1's bar line."""
        return round(float(grid.position(time)) * DIVISIONS) - origin * DIVISIONS

    events = _quantize(notes, to_units)
    total_units = events[-1].start + events[-1].length if events else 0
    symbols = analysis.chords if chords is None else chords
    if symbols:  # chords may go on after the last note (a song's outro)
        total_units = max(total_units, to_units(max(c.end for c in symbols)))
    n_measures = max(1, math.ceil(total_units / measure_units))
    measures: list[list[tuple[int, list[Note], bool, bool]]] = [[] for _ in range(n_measures)]
    for event in events:
        pieces = list(_pieces(event.start, event.length, measure_units))
        for k, (measure, units) in enumerate(pieces):
            tie_stop, tie_start = k > 0, k < len(pieces) - 1
            measures[measure].append((units, event.notes, tie_start, tie_stop))
    for content in measures:  # pad the last measure with rests
        filled = sum(units for units, *_ in content)
        for units in _decompose(measure_units - filled):
            content.append((units, [], False, False))

    harmonies: list[list[tuple[int, Chord]]] = [[] for _ in range(n_measures)]
    for chord in symbols:
        position = max(0, to_units(chord.start))
        if position < n_measures * measure_units:
            harmonies[position // measure_units].append((position % measure_units, chord))

    def bar_time(index: int) -> float:
        return float(grid.time(origin + index * beats_per_measure))

    def measure_key(index: int) -> Key | None:
        if key is not None or not keys:
            return analysis.key
        start = bar_time(index) + 1e-3
        return next((s.key for s in keys if s.start <= start < s.end), keys[-1].key)

    def measure_tempo(index: int) -> float:
        return 60.0 * beats_per_measure / (bar_time(index + 1) - bar_time(index))

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
    written_key, sounding_tempo, shown_tempo = None, 0.0, 0.0
    for index, content in enumerate(measures):
        measure = ET.SubElement(part, "measure", number=str(index + 1))
        this_key = measure_key(index)
        local = measure_tempo(index) if analysis.tracked else tempo
        if index == 0:
            _write_attributes(measure, tuning, this_key, beats_per_measure)
            _write_tempo(measure, local)
            sounding_tempo = shown_tempo = local
            if capo:
                _write_words(measure, f"Capo {capo}")
        else:
            if this_key != written_key:
                _write_key_change(measure, this_key)
            if abs(local - sounding_tempo) > 0.02 * sounding_tempo:
                visible = abs(local - shown_tempo) > 0.08 * shown_tempo
                _write_tempo(measure, local, visible)
                sounding_tempo = local
                shown_tempo = local if visible else shown_tempo
        written_key = this_key
        for staff, voice in ((1, "1"), (2, "5")):
            if staff == 2:
                ET.SubElement(ET.SubElement(measure, "backup"), "duration").text = str(
                    measure_units
                )
            pending = sorted(harmonies[index], key=lambda h: h[0]) if staff == 1 else []
            position = 0
            for length, event_notes, tie_start, tie_stop in content:
                while pending and pending[0][0] < position + length:
                    at, chord = pending.pop(0)
                    _write_harmony(measure, chord, this_key, max(0, at - position))
                _write_event(
                    measure, length, event_notes, tie_start, tie_stop, staff, voice, tuning,
                    this_key, measure_units,
                )  # fmt: skip
                position += length

    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{DOCTYPE}\n{body}\n'.encode()


def first_bar_line(notes: Sequence[Note], grid: BeatGrid) -> int:
    """Beat index of the last bar line at or before the first (quantized) onset: measure 1
    starts there (the first bar line at or after 0 s without notes)."""
    if not notes:
        beat = next((i for i, t in enumerate(grid.beats) if t >= -1e-9), 0)
    else:
        sixteenth = round(float(grid.position(min(n.start for n in notes))) * DIVISIONS)
        beat = sixteenth // DIVISIONS
    while not grid.is_bar_line(beat):
        beat -= 1
    return beat


def _quantize(notes: Sequence[Note], to_units: Callable[[float], int]) -> list[_Event]:
    """Snap onsets to the grid and build a gap-free sequence of note/chord/rest events."""
    cells: dict[int, list[Note]] = {}
    for note in sort_notes(notes):
        cells.setdefault(max(0, to_units(note.start)), []).append(note)

    starts = sorted(cells)
    events: list[_Event] = []
    cursor = 0
    for i, start in enumerate(starts):
        members = _one_per_string(cells[start])
        end = max(start + 1, to_units(max(n.end for n in members)))
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


def _pieces(start: int, length: int, measure_units: int) -> Iterator[tuple[int, int]]:
    """Split a span at barlines and into writable note values: (measure, units)."""
    position, remaining = start, length
    while remaining > 0:
        chunk = min(remaining, measure_units - position % measure_units)
        for units in _decompose(chunk):
            yield position // measure_units, units
            position += units
        remaining -= chunk


def _decompose(units: int) -> list[int]:
    parts = []
    for value, _, _ in _NOTE_VALUES:
        while units >= value:
            parts.append(value)
            units -= value
    return parts


def _spell(midi: int, key: Key | None) -> tuple[str, int, int]:
    """(step, alter, octave); the octave follows the letter (B#3 is MIDI 60)."""
    step, alter = spell(midi, key)
    return step, alter, (midi - alter) // 12 - 1


def _write_attributes(
    measure: ET.Element, tuning: Sequence[int], key: Key | None, beats_per_measure: int
) -> None:
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = str(DIVISIONS)
    key_element = ET.SubElement(attributes, "key")
    ET.SubElement(key_element, "fifths").text = str(key.fifths if key else 0)
    if key:
        ET.SubElement(key_element, "mode").text = key.mode
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = str(beats_per_measure)
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
        step, alter, octave = _spell(open_pitch, None)
        staff_tuning = ET.SubElement(details, "staff-tuning", line=str(line))
        ET.SubElement(staff_tuning, "tuning-step").text = step
        if alter:
            ET.SubElement(staff_tuning, "tuning-alter").text = str(alter)
        ET.SubElement(staff_tuning, "tuning-octave").text = str(octave)


def _write_words(measure: ET.Element, text: str) -> None:
    direction = ET.SubElement(measure, "direction", placement="above")
    ET.SubElement(ET.SubElement(direction, "direction-type"), "words").text = text
    ET.SubElement(direction, "staff").text = "1"


def _write_tempo(measure: ET.Element, tempo: float, visible: bool = True) -> None:
    """A tempo for playback, shown as a metronome mark when ``visible``."""
    if not visible:
        ET.SubElement(measure, "sound", tempo=f"{tempo:.2f}")
        return
    direction = ET.SubElement(measure, "direction", placement="above")
    metronome = ET.SubElement(ET.SubElement(direction, "direction-type"), "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(round(tempo))
    ET.SubElement(direction, "staff").text = "1"
    ET.SubElement(direction, "sound", tempo=f"{tempo:.2f}")


def _write_key_change(measure: ET.Element, key: Key | None) -> None:
    attributes = ET.SubElement(measure, "attributes")
    key_element = ET.SubElement(attributes, "key")
    ET.SubElement(key_element, "fifths").text = str(key.fifths if key else 0)
    if key:
        ET.SubElement(key_element, "mode").text = key.mode


def _write_harmony(measure: ET.Element, chord: Chord, key: Key | None, offset: int) -> None:
    """A chord symbol above the staff, ``offset`` grid units after the next note/rest."""
    harmony = ET.SubElement(measure, "harmony")
    root = ET.SubElement(harmony, "root")
    step, alter = spell(chord.root, key)
    ET.SubElement(root, "root-step").text = step
    if alter:
        ET.SubElement(root, "root-alter").text = str(alter)
    suffix = QUALITIES[chord.quality][1]
    ET.SubElement(harmony, "kind", text=suffix).text = _HARMONY_KINDS[chord.quality]
    if chord.bass is not None:
        bass = ET.SubElement(harmony, "bass")
        step, alter = spell(chord.bass, key)
        ET.SubElement(bass, "bass-step").text = step
        if alter:
            ET.SubElement(bass, "bass-alter").text = str(alter)
    if offset:
        ET.SubElement(harmony, "offset").text = str(offset)
    ET.SubElement(harmony, "staff").text = "1"


def _write_event(
    measure: ET.Element,
    units: int,
    notes: list[Note],
    tie_start: bool,
    tie_stop: bool,
    staff: int,
    voice: str,
    tuning: Sequence[int],
    key: Key | None,
    measure_units: int,
) -> None:
    value_type, dots = next((t, d) for u, t, d in _NOTE_VALUES if u == units)
    if not notes:
        note = ET.SubElement(measure, "note")
        rest = ET.SubElement(note, "rest")
        if units == measure_units:
            rest.set("measure", "yes")
        _write_timing(note, units, voice, value_type, dots, staff)
        return

    for index, source in enumerate(notes):
        note = ET.SubElement(measure, "note")
        if index > 0:
            ET.SubElement(note, "chord")
        step, alter, octave = _spell(source.pitch, key)
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
