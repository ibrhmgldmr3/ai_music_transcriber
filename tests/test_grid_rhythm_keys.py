"""The beat grid (tracked beats), key changes, strumming patterns and fused chords."""

import io

import mido
import numpy as np
import pytest

from music_core.analysis import Key, analyze, segment_keys
from music_core.midi import notes_to_midi
from music_core.musicxml import notes_to_musicxml
from music_core.notes import Note
from music_core.rhythm import strum_rhythm, strums_from_notes
from music_core.timing import BeatGrid, downbeat_phase, median_tempo, tracked_meter

C = [48, 52, 55, 60]
G = [43, 47, 50, 55]
AM = [45, 52, 57, 60]
F = [41, 48, 53, 57]


def slowing_beats(count=32, start=0.2, first=0.5, last=0.65):
    """A player drifting from 120 to about 92 BPM."""
    gaps = np.linspace(first, last, count - 1)
    return list(np.round(start + np.concatenate([[0], np.cumsum(gaps)]), 4))


def strummed(chords, beats, per_chord=4):
    """One chord per bar of tracked beats, struck on every beat."""
    notes = []
    for i, pitches in enumerate(chords):
        for k in range(per_chord):
            b = i * per_chord + k
            if b + 1 < len(beats):
                end = beats[b] + 0.9 * (beats[b + 1] - beats[b])
                notes += [Note(p, beats[b], end) for p in pitches]
    return notes


# --- the grid ------------------------------------------------------------------------


def test_fixed_grid_puts_bar_lines_through_the_downbeat():
    grid = BeatGrid.fixed(120, 4, 0.3, 0.0, 5.0)
    assert grid.tempo == pytest.approx(120)
    lines = [t for t in grid.bar_lines if 0 <= t <= 5]
    assert lines == pytest.approx([0.3, 2.3, 4.3])
    assert grid.time(grid.position(1.234)) == pytest.approx(1.234)


def test_tracked_grid_extends_the_beats_and_converts_both_ways():
    beats = slowing_beats()
    grid = BeatGrid.tracked(beats, 3, 1, 0.0, beats[-1] + 3)
    assert grid.beats[0] < 0 and grid.beats[-1] > beats[-1] + 3  # covered, a bar to spare
    assert beats[1] in grid.bar_lines and beats[4] in grid.bar_lines  # every 3rd from #1
    for t in (0.05, 3.3, beats[-1] + 1.0):
        assert grid.time(grid.position(t)) == pytest.approx(t)
    moved = grid.with_bar_line_at(beats[2] + 0.01)  # "this note is beat 1"
    assert beats[2] in moved.bar_lines and beats[1] not in moved.bar_lines


def test_meter_phase_and_tempo_from_tracked_beats():
    beats = [0.5 * i for i in range(30)]
    downbeats = beats[2::3]
    downbeats.pop(4)  # a missed bar line
    assert tracked_meter(beats, downbeats) == 3
    assert downbeat_phase(beats, downbeats, 3) == 2
    assert median_tempo(beats) == pytest.approx(120)


def test_analysis_follows_tracked_beats():
    beats = slowing_beats()
    notes = strummed([C, G, AM, F, C, G, AM], beats)
    analysis = analyze(notes, 120, 4, beats=beats, downbeats=beats[::4])
    assert analysis.tracked and analysis.tempo == pytest.approx(median_tempo(beats))
    assert [round(t, 4) for t in analysis.grid.bar_lines if 0 <= t <= beats[-1]] == beats[::4]
    labels = [c.label(analysis.key) for c in analysis.chords]
    assert labels[:7] == ["C", "G", "Am", "F", "C", "G", "Am"]
    # One tempo can't follow the drift: its bar lines wander off the chord changes.
    fixed = analyze(notes, 120, 4, downbeat=beats[0])
    late = [t for t in fixed.grid.bar_lines if 10 < t < beats[-1]]
    assert min(abs(t - b) for t in late for b in beats[::4]) > 0.1


# --- exports -----------------------------------------------------------------------


def test_musicxml_writes_the_tempo_where_tracked_beats_slow_down():
    beats = slowing_beats()
    notes = strummed([C, G, AM, F, C, G, AM], beats)
    xml = notes_to_musicxml(notes, 120, beats=beats, downbeats=beats[::4]).decode()
    tempos = [float(x.split('"')[1]) for x in xml.split("<sound tempo=")[1:]]
    assert len(tempos) > 2 and tempos[0] > 110 and tempos[-1] < 100
    assert xml.count("<measure ") == 7  # one chord per measure, none split by drift


def test_midi_tempo_map_keeps_seconds_and_aligns_bars():
    beats = slowing_beats(start=0.9)
    grid = analyze([], 120, 4, beats=beats, downbeats=beats[1::4]).grid
    notes = [Note(60, b, b + 0.2) for b in beats[1:20]]
    midi = notes_to_midi(notes, grid=grid)
    buffer = io.BytesIO()
    midi.save(file=buffer)
    buffer.seek(0)
    starts = [
        round(m.time, 3) for m in _absolute(mido.MidiFile(file=buffer)) if m.type == "note_on"
    ]
    assert starts == pytest.approx(beats[1:20], abs=2e-3)
    ticks = 0
    bar_ticks = []
    for message in midi.tracks[0]:
        ticks += message.time
        if message.type == "note_on" and len(bar_ticks) < 2:
            bar_ticks.append(ticks)
    # The 1.4 s before the first bar line (beats[1]) are about 2.8 beats: a 3/4 pickup, so
    # the bar line falls on the file's second bar, then 4/4.
    assert bar_ticks[0] == 3 * midi.ticks_per_beat
    signatures = [m.numerator for m in midi.tracks[0] if m.type == "time_signature"]
    assert signatures == [3, 4]


def _absolute(midi):
    now = 0.0
    for message in midi:
        now += message.time
        yield message.copy(time=now)


# --- keys ------------------------------------------------------------------------------


def test_segment_keys_finds_a_modulation_and_keeps_one_key_otherwise():
    bar = 2.0
    in_c = strummed([C, F, G, C] * 4, [i * 0.5 for i in range(80)])
    d, g, a = [50, 54, 57, 62], [43, 47, 50, 55], [45, 49, 52, 57]
    in_d = [
        Note(n.pitch, n.start + 32.0, n.end + 32.0)
        for n in strummed([d, g, a, d] * 4, [i * 0.5 for i in range(80)])
    ]
    bars = [i * bar for i in range(33)]
    spans = segment_keys(in_c + in_d, bars)
    assert [s.key for s in spans] == [Key.parse("C major"), Key.parse("D major")]
    assert spans[1].start == pytest.approx(32.0, abs=bar)
    assert len(segment_keys(in_c, bars[:17])) == 1
    assert segment_keys([], bars) == []


# --- strumming pattern -----------------------------------------------------------------


def test_strums_and_their_pattern():
    beats = [0.5 * i for i in range(17)]
    eighths = [0, 2, 3, 5, 6, 7]  # D - D U - U D U
    notes = []
    for bar in range(4):
        for slot in eighths:
            t = bar * 2.0 + slot * 0.25
            notes += [Note(p, t + 0.005 * k, t + 0.2) for k, p in enumerate(C[:2])]
    strums = strums_from_notes(notes)
    assert len(strums) == 24 and all(s == 2.0 for _, s in strums)
    grid = BeatGrid.tracked(beats, 4, 0, 0.0, 8.0)
    rhythm = strum_rhythm(strums, grid)
    assert rhythm.per_beat == 2 and rhythm.text() == "D-DU-UDU"
    assert len(rhythm.bars) == 4
    # A melody line is not strummed.
    assert strums_from_notes([Note(60 + i, i * 0.25, i * 0.25 + 0.2) for i in range(8)]) == []
    assert analyze([Note(60 + i, i * 0.25, i * 0.25 + 0.2) for i in range(8)], 120).rhythm is None


def test_sixteenth_patterns_are_written_in_sixteenths():
    beats = [0.5 * i for i in range(9)]
    times = [b + o for b in beats[:8] for o in (0.0, 0.125, 0.375)]  # 1 e . a
    rhythm = strum_rhythm([(t, 3.0) for t in times], BeatGrid.tracked(beats, 4, 0, 0, 4))
    assert rhythm.per_beat == 4 and rhythm.text() == "DU-UDU-UDU-UDU-U"


# --- chords from the recording and the notes ------------------------------------------


def test_fused_chords_follow_the_recognizer_where_the_notes_are_unclear():
    beats = [0.5 * i for i in range(17)]
    # A bare fifth (C-G) is C major or minor; the recording says minor.
    notes = strummed([[48, 55, 60], [48, 55, 60]], beats)
    scores = [{"start": b, "top": [["C:min", -0.2], ["C:maj", -1.8]]} for b in beats]
    fused = analyze(notes, 120, beats=beats, chord_scores=scores)
    assert [c.label(fused.key) for c in fused.chords] == ["Cm"]
    # Clear notes win over a recognizer that is unsure.
    e_minor = strummed([[40, 47, 52, 55], [40, 47, 52, 55]], beats)
    unsure = [{"start": b, "top": [["C:maj", -0.9], ["E:min", -1.0]]} for b in beats]
    assert [c.label() for c in analyze(e_minor, 120, beats=beats, chord_scores=unsure).chords] == [
        "Em"
    ]
    # A melody line gets no chords, whatever the recognizer says.
    melody = [Note(60 + i % 5, i * 0.5, i * 0.5 + 0.4) for i in range(16)]
    assert analyze(melody, 120, beats=beats, chord_scores=scores).chords == []
