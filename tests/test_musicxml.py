import xml.etree.ElementTree as ET

import pytest

from music_core.musicxml import MEASURE_UNITS, notes_to_musicxml
from music_core.notes import Note

# At 60 BPM a quarter note is 1 s and one grid unit (a 16th) is 0.25 s.
TEMPO = 60.0


def parse(data: bytes) -> ET.Element:
    assert data.startswith(b'<?xml version="1.0" encoding="UTF-8"?>')
    return ET.fromstring(data.split(b"\n", 2)[2])  # skip declaration + DOCTYPE


def staff_notes(root: ET.Element, staff: str) -> list[ET.Element]:
    return [n for n in root.iter("note") if n.findtext("staff") == staff]


def test_every_measure_is_full_on_both_staves():
    notes = [Note(60 + i % 5, i * 0.3, i * 0.3 + 0.2, string=3, fret=5) for i in range(30)]
    root = parse(notes_to_musicxml(notes, tempo=TEMPO))
    for measure in root.iter("measure"):
        for staff in ("1", "2"):
            total = sum(
                int(n.findtext("duration"))
                for n in measure.findall("note")
                if n.findtext("staff") == staff and n.find("chord") is None
            )
            assert total == MEASURE_UNITS, (measure.get("number"), staff, total)


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
    root = parse(notes_to_musicxml(notes, tempo=TEMPO))
    treble = staff_notes(root, "1")
    assert treble[1].find("chord") is not None  # second chord tone
    assert any(n.find("rest") is not None for n in treble)  # beats 2-3 are silent
    tied = [n for n in treble if n.findtext("pitch/step") == "C"]
    assert [t.get("type") for n in tied for t in n.findall("tie")] == ["start", "stop"]
    assert sum(int(n.findtext("duration")) for n in tied) == 12  # 3 beats


def test_accidentals_and_tempo():
    root = parse(notes_to_musicxml([Note(61, 0.0, 1.0)], tempo=97.4))
    pitch = next(iter(root.iter("pitch")))
    assert (pitch.findtext("step"), pitch.findtext("alter"), pitch.findtext("octave")) == (
        "C",
        "1",
        "4",
    )
    assert root.findtext(".//metronome/per-minute") == "97"


@pytest.mark.parametrize("tempo", [0.0, float("nan")])
def test_unusable_tempo_falls_back(tempo):
    root = parse(notes_to_musicxml([Note(60, 0.0, 1.0)], tempo=tempo))
    assert root.findtext(".//metronome/per-minute") == "120"


def test_empty_transcription_is_a_single_rest_measure():
    root = parse(notes_to_musicxml([], title="Boş"))
    assert len(root.findall(".//measure")) == 1
    assert root.find(".//rest").get("measure") == "yes"
    assert root.findtext(".//work-title") == "Boş"
