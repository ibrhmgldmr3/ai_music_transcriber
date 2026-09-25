import pytest

from music_core.analysis import (
    KEY_NAMES,
    Key,
    analyze,
    beat_phase,
    estimate_key,
    pitch_class_name,
    refine_tempo,
    spell,
)
from music_core.notes import Note

# Chord tones (MIDI) voiced as a guitarist would: bass note first.
C = [48, 52, 55, 60]
G = [43, 47, 50, 55]
AM = [45, 52, 57, 60]
F = [41, 48, 53, 57]
E5 = [40, 47, 52]


def strum(pitches, start, length=1.0):
    return [Note(p, start, start + length) for p in pitches]


def progression(chords, beat=0.5, beats_per_chord=4, offset=0.0, strums_per_chord=4):
    """One chord per bar, strummed on every beat."""
    notes = []
    for i, pitches in enumerate(chords):
        for s in range(strums_per_chord):
            start = offset + (i * beats_per_chord + s * beats_per_chord / strums_per_chord) * beat
            notes += strum(pitches, start, beat * 0.9)
    return notes


# --- keys and spelling -------------------------------------------------------------


def test_key_names_round_trip():
    assert len(KEY_NAMES) == len(set(KEY_NAMES)) == 24
    for name in KEY_NAMES:
        assert Key.parse(name).name == name
    with pytest.raises(ValueError):
        Key.parse("H major")


@pytest.mark.parametrize(
    ("name", "fifths"),
    [("C major", 0), ("A minor", 0), ("Bb major", -2), ("F# minor", 3), ("Eb minor", -6)],
)
def test_key_signature(name, fifths):
    assert Key.parse(name).fifths == fifths


def test_estimate_key_from_progressions():
    assert estimate_key(progression([C, F, G, C])) == Key.parse("C major")
    assert estimate_key(progression([AM, [40, 47, 52, 56, 59], AM, AM])) == Key.parse("A minor")
    # Transposition moves the key with it.
    up = [Note(n.pitch + 2, n.start, n.end) for n in progression([C, F, G, C])]
    assert estimate_key(up) == Key.parse("D major")
    assert estimate_key([]) is None


def test_spelling_follows_the_key():
    assert pitch_class_name(10, Key.parse("F major")) == "Bb"
    assert pitch_class_name(6, Key.parse("G major")) == "F#"
    assert pitch_class_name(8, Key.parse("A minor")) == "G#"  # raised leading tone
    assert pitch_class_name(3, Key.parse("E major")) == "D#"
    assert pitch_class_name(3, Key.parse("Bb major")) == "Eb"
    assert spell(5, Key.parse("F# major")) == ("E", 1)  # E#, not F
    # Outside a key: sharps in sharp keys, flats in flat keys.
    assert pitch_class_name(1, Key.parse("D major")) == "C#"
    assert pitch_class_name(1, Key.parse("Eb major")) == "Db"
    assert pitch_class_name(2, Key.parse("Ab major")) == "D"


def test_every_pitch_class_is_spelled_in_every_key():
    for name in KEY_NAMES:
        key = Key.parse(name)
        for pc in range(12):
            letter, alter = spell(pc, key)
            assert letter in "CDEFGAB" and -2 <= alter <= 2
            natural = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
            assert (natural + alter) % 12 == pc, (name, pc)


# --- bar grid ----------------------------------------------------------------------


def test_beat_phase_and_downbeat_with_a_pickup():
    # 120 BPM (0.5 s beats), bars start 0.3 s in; one pickup beat before the first bar.
    notes = strum([64], 0.3 - 0.5, 0.4) + progression([C, G, AM, F, C], offset=0.3)
    notes = [n for n in notes if n.start >= 0] + strum([67], 0.05, 0.2)
    assert beat_phase(notes, 120) == pytest.approx(0.3, abs=0.01)
    analysis = analyze(notes, 120)
    assert analysis.downbeat == pytest.approx(0.3, abs=0.01)


def test_downbeat_follows_chord_changes():
    # Chords change every 3 beats in 3/4; the grid starts on beat 2 of a bar.
    notes = progression([C, G, AM, F, C, G], beats_per_chord=3, strums_per_chord=3, offset=-0.5)
    notes = [n for n in notes if n.start >= 0]
    analysis = analyze(notes, 120, beats_per_measure=3)
    bar = 1.5
    assert (analysis.downbeat - 1.0) % bar == pytest.approx(0, abs=0.01)


def test_refine_tempo_recovers_the_exact_tempo():
    notes = progression([C, G, AM, F] * 4, beat=60 / 97.0)
    assert refine_tempo(notes, 99.0) == pytest.approx(97.0, rel=0.002)
    assert refine_tempo([], 99.0) == 99.0


# --- chords ------------------------------------------------------------------------


def labels(notes, tempo=120, **kwargs):
    analysis = analyze(notes, tempo, **kwargs)
    return [chord.label(analysis.key) for chord in analysis.chords]


def test_chord_progression():
    assert labels(progression([C, G, AM, F])) == ["C", "G", "Am", "F"]


def test_arpeggios_are_chords_but_a_melody_is_not():
    arpeggio = [Note(p, i * 0.25, 2.0) for i, p in enumerate(C)]  # let ring
    assert labels(arpeggio) == ["C"]
    melody = [Note(60 + step, i * 0.5, i * 0.5 + 0.45) for i, step in enumerate([0, 2, 4, 5])]
    assert labels(melody) == []
    # A legato solo line outlining a chord still isn't labeled: one note at a time.
    solo = [Note(p, i * 0.25, i * 0.25 + 0.3) for i, p in enumerate([60, 64, 67, 72] * 4)]
    assert labels(solo) == []


def test_power_seventh_and_slash_chords():
    assert labels(progression([E5, E5])) == ["E5"]
    g7 = [43, 47, 50, 53]
    assert labels(progression([g7, C]))[0] == "G7"
    d_over_f_sharp = [42, 50, 57, 62]  # F#2 in the bass
    assert labels(progression([d_over_f_sharp, G])) == ["D/F#", "G"]


def test_chord_names_are_spelled_in_the_key():
    b_flat = [46, 53, 58, 62]
    notes = progression([F, b_flat, C, F])
    assert labels(notes, key=Key.parse("F major")) == ["F", "Bb", "C", "F"]


def test_empty_and_degenerate_input():
    analysis = analyze([], None)
    assert (analysis.tempo, analysis.downbeat, analysis.key, analysis.chords) == (
        120.0,
        0,
        None,
        [],
    )
    assert analyze([Note(60, 1.0, 1.0)], 100).chords == []  # zero-length notes are ignored
    assert analyze([Note(60, 0.0, 1.0)], float("nan")).tempo == 120.0
