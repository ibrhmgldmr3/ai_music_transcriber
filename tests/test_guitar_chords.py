import pytest

from music_core.analysis import Key
from music_core.guitar_chords import QUALITIES, ChordSymbol, suggest_capo, voicing
from music_core.tab import STANDARD_TUNING, open_strings


def shape(label: str, tuning=STANDARD_TUNING) -> str:
    return voicing(ChordSymbol.parse(label), tuning).diagram


@pytest.mark.parametrize(
    "label, expected",
    [
        ("C", "x32010"), ("G", "320003"), ("D", "xx0232"), ("A", "x02220"), ("E", "022100"),
        ("A:min", "x02210"), ("E:min", "022000"), ("D:min", "xx0231"),
        ("E:7", "020100"), ("B:7", "x21202"), ("C:maj7", "x32000"), ("A:min7", "x02010"),
        ("F", "133211"), ("B:min", "x24432"), ("F#", "244322"), ("C:min", "x35543"),
    ],
)  # fmt: skip
def test_standard_tuning_uses_the_shapes_guitarists_know(label, expected):
    assert shape(label) == expected


def test_every_chord_of_every_root_has_a_playable_shape():
    for root in range(12):
        for quality in QUALITIES:
            v = voicing(ChordSymbol(root, quality))
            assert v is not None, (root, quality)
            played = [STANDARD_TUNING[i] + f for i, f in enumerate(v.frets) if f >= 0]
            tones = {(root + i) % 12 for i in QUALITIES[quality][0]}
            assert {p % 12 for p in played} <= tones and played[0] % 12 == root


def test_open_chords_are_easier_than_barre_chords():
    assert voicing(ChordSymbol.parse("E")).difficulty < voicing(ChordSymbol.parse("F")).difficulty


def test_other_tunings_are_searched_for_playable_shapes():
    drop_d = open_strings("drop_d")
    assert shape("D", drop_d) == "000232"
    v = voicing(ChordSymbol.parse("G"), drop_d)
    played = [drop_d[i] + f for i, f in enumerate(v.frets) if f >= 0]
    assert played[0] % 12 == 7 and {p % 12 for p in played} <= {7, 11, 2}


def test_labels_names_and_transposition():
    chord = ChordSymbol.parse("A#:min7")
    assert (chord.root, chord.quality) == (10, "min7")
    assert chord.name(Key(5, "major")) == "Bbm7" and chord.name(Key(11, "major")) == "A#m7"
    assert ChordSymbol.parse("Bb") == ChordSymbol(10, "maj")
    assert ChordSymbol.parse("N") is None and ChordSymbol.parse("X") is None
    assert chord.transposed(-3).name() == "Gm7" and chord.harte == "Bb:min7"
    with pytest.raises(ValueError):
        ChordSymbol.parse("H:maj")


def test_slash_chords():
    d_over_f_sharp = ChordSymbol.parse("D:maj/3")
    assert d_over_f_sharp == ChordSymbol(2, "maj", bass=6) == ChordSymbol.parse("D/F#")
    assert d_over_f_sharp.name() == "D/F#" and d_over_f_sharp.harte == "D:maj/3"
    assert ChordSymbol.parse("C:maj/1") == ChordSymbol(0, "maj")  # the root is no slash
    # A capo moves the bass with the chord: Eb/G is played as a C/E shape at fret 3.
    assert ChordSymbol.parse("D#:maj/3").transposed(-3) == ChordSymbol.parse("C/E")
    # The bass note is the lowest string, under a full chord.
    shapes = {"D/F#": "200232", "G/B": "x20003", "C/E": "032010", "A:min/b7": "302010",
              "C/B": "x22010", "E:min/b3": "322000"}  # fmt: skip
    for label, shape in shapes.items():
        assert voicing(ChordSymbol.parse(label)).diagram == shape, label
    with pytest.raises(ValueError):
        ChordSymbol.parse("C:maj/H")


def test_capo_turns_a_song_into_open_shapes():
    ab_major = [(ChordSymbol.parse(c), 4.0) for c in ("G#", "C#", "D#", "F:min")]
    capo, costs = suggest_capo(ab_major)
    assert capo == 1  # G, C, D, Em
    assert costs[1] < costs[0]
    g_major = [(ChordSymbol.parse(c), 4.0) for c in ("G", "C", "D", "E:min")]
    assert suggest_capo(g_major)[0] == 0  # already easy: no capo


def test_chord_sheet_shows_the_shapes_bar_by_bar():
    from music_core.guitar_chords import chord_sheet

    chords = [
        (0.0, 4.0, ChordSymbol.parse("C:min")),
        (4.0, 6.0, ChordSymbol.parse("A#")),
        (6.0, 8.0, ChordSymbol.parse("F")),
    ]
    text = chord_sheet(
        chords, [0.0, 2.0, 4.0, 6.0], title="Song", key=Key(5, "major"), capo=3, tempo=120
    )
    lines = text.splitlines()
    assert lines[0] == "Song" and lines[1] == "Ton: F major · Capo 3 · 120 BPM · 4/4"
    assert lines[3].split("|")[1:5] == [" Am ", " Am ", " G  ", " D  "]
    assert "Am       x02210   (= Cm)" in text and "G        320003   (= Bb)" in text
    assert "Cm" in chord_sheet(chords, [0.0], key=Key(5, "major"))  # no capo: real names
    pickup = chord_sheet(chords, [1.0, 5.0], capo=3).splitlines()[2]
    assert pickup.split("|")[1:4] == [" Am   ", " Am G ", " G D  "]
    assert chord_sheet(chords, []).splitlines()[2].count("|") == 5  # 2 s bars
