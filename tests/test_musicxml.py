import xml.etree.ElementTree as ET

import pytest

from music_core.analysis import Key
from music_core.musicxml import notes_to_musicxml
from music_core.notes import Note

# At 60 BPM a quarter note is 1 s and one grid unit (a 16th) is 0.25 s.
TEMPO = 60.0
MEASURE_UNITS = 16  # 4/4


def parse(data: bytes) -> ET.Element:
    assert data.startswith(b'<?xml version="1.0" encoding="UTF-8"?>')
    return ET.fromstring(data.split(b"\n", 2)[2])  # skip declaration + DOCTYPE


def staff_notes(root: ET.Element, staff: str) -> list[ET.Element]:
    return [n for n in root.iter("note") if n.findtext("staff") == staff]


def test_every_measure_is_full_on_both_staves():
    notes = [Note(60 + i % 5, i * 0.3, i * 0.3 + 0.2, string=3, fret=5) for i in range(30)]
    root = parse(notes_to_musicxml(notes, tempo=TEMPO))
    assert_full_measures(root, MEASURE_UNITS)


def assert_full_measures(root: ET.Element, measure_units: int) -> None:
    for measure in root.iter("measure"):
        for staff in ("1", "2"):
            total = sum(
                int(n.findtext("duration"))
                for n in measure.findall("note")
                if n.findtext("staff") == staff and n.find("chord") is None
            )
            assert total == measure_units, (measure.get("number"), staff, total)


def test_tab_staff_uses_musicxml_string_numbering():
    # Internally string 5 is the high e; MusicXML calls it string 1.
    root = parse(notes_to_musicxml([Note(64, 0.0, 1.0, string=5, fret=0)], tempo=TEMPO))
    (tab_note,) = [n for n in staff_notes(root, "2") if n.find("pitch") is not None]
    assert tab_note.findtext("notations/technical/string") == "1"
    assert tab_note.findtext("notations/technical/fret") == "0"
    assert root.findtext(".//clef[@number='2']/sign") == "TAB"
    lowest = root.find(".//staff-tuning[@line='1']")
    assert (lowest.findtext("tuning-step"), lowest.findtext("tuning-octave")) == ("E", "2")


def test_chords_rests_and_ties():
    notes = [
        Note(52, 0.0, 1.0, string=2, fret=2),  # E3 + G#3 chord, one beat
        Note(56, 0.0, 1.0, string=3, fret=1),
        Note(60, 3.0, 6.0, string=3, fret=5),  # starts on beat 4, crosses the barline
    ]
    root = parse(notes_to_musicxml(notes, tempo=TEMPO, downbeat=0.0))
    treble = staff_notes(root, "1")
    assert treble[1].find("chord") is not None  # second chord tone
    assert any(n.find("rest") is not None for n in treble)  # beats 2-3 are silent
    tied = [n for n in treble if n.findtext("pitch/step") == "C"]
    assert [t.get("type") for n in tied for t in n.findall("tie")] == ["start", "stop"]
    assert sum(int(n.findtext("duration")) for n in tied) == 12  # 3 beats


def test_spelling_key_signature_and_tempo():
    notes = [Note(61, 0.0, 1.0)]  # C# / Db
    root = parse(notes_to_musicxml(notes, tempo=97.4, key=Key.parse("A major")))
    pitch = next(iter(root.iter("pitch")))
    assert (pitch.findtext("step"), pitch.findtext("alter"), pitch.findtext("octave")) == (
        "C",
        "1",
        "4",
    )
    assert (root.findtext(".//key/fifths"), root.findtext(".//key/mode")) == ("3", "major")
    assert root.findtext(".//metronome/per-minute") == "97"

    root = parse(notes_to_musicxml(notes, key=Key.parse("Bb minor")))
    pitch = next(iter(root.iter("pitch")))
    assert (pitch.findtext("step"), pitch.findtext("alter")) == ("D", "-1")
    assert root.findtext(".//key/fifths") == "-5"


def test_enharmonic_octave_follows_the_letter():
    root = parse(notes_to_musicxml([Note(59, 0.0, 1.0)], key=Key.parse("Eb minor")))  # B3 = Cb4
    pitch = next(iter(root.iter("pitch")))
    assert (pitch.findtext("step"), pitch.findtext("alter"), pitch.findtext("octave")) == (
        "C",
        "-1",
        "4",
    )


def test_three_four_time():
    notes = [Note(60 + i % 3, i * 0.5, i * 0.5 + 0.4) for i in range(20)]
    root = parse(notes_to_musicxml(notes, tempo=TEMPO, beats_per_measure=3, downbeat=0.0))
    assert (root.findtext(".//time/beats"), root.findtext(".//time/beat-type")) == ("3", "4")
    assert_full_measures(root, 12)


def test_bar_lines_start_at_the_downbeat():
    # A pickup note on beat 4, then the first full bar at 1 s.
    notes = [Note(64, 0.0, 1.0), Note(60, 1.0, 2.0), Note(62, 2.0, 3.0)]
    root = parse(notes_to_musicxml(notes, tempo=TEMPO, downbeat=1.0))
    first, second = root.findall(".//measure")[:2]
    treble = [n for n in first.findall("note") if n.findtext("staff") == "1"]
    assert treble[0].find("rest") is not None and treble[0].findtext("duration") == "12"
    assert treble[1].findtext("pitch/step") == "E"
    assert next(n for n in second.findall("note")).findtext("pitch/step") == "C"


def test_chord_symbols():
    c_major = [48, 52, 55, 60]
    d_over_f_sharp = [42, 50, 57, 62]
    notes = [Note(p, beat, beat + 0.9) for beat in range(4) for p in c_major]
    notes += [Note(p, 4 + beat, 4.9 + beat) for beat in range(4) for p in d_over_f_sharp]
    root = parse(notes_to_musicxml(notes, tempo=TEMPO, downbeat=0.0, key=Key.parse("G major")))
    harmonies = root.findall(".//harmony")
    assert [h.findtext("root/root-step") for h in harmonies] == ["C", "D"]
    assert [h.findtext("kind") for h in harmonies] == ["major", "major"]
    assert harmonies[1].findtext("bass/bass-step") == "F"
    assert harmonies[1].findtext("bass/bass-alter") == "1"
    # Each symbol sits right before the first note of its bar.
    for measure in root.findall(".//measure"):
        children = [c.tag for c in measure]
        assert children.index("harmony") < children.index("note")


@pytest.mark.parametrize("tempo", [0.0, float("nan")])
def test_unusable_tempo_falls_back(tempo):
    root = parse(notes_to_musicxml([Note(60, 0.0, 1.0)], tempo=tempo))
    assert root.findtext(".//metronome/per-minute") == "120"


def test_empty_transcription_is_a_single_rest_measure():
    root = parse(notes_to_musicxml([], title="Boş"))
    assert len(root.findall(".//measure")) == 1
    assert root.find(".//rest").get("measure") == "yes"
    assert root.findtext(".//work-title") == "Boş"
