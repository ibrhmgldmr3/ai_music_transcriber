from music_core.notes import Note
from music_core.tab import assign_tab, fret_to_pitch, pitch_to_positions, tab_to_ascii


def test_pitch_to_positions_e4():
    assert set(pitch_to_positions(64)) == {(5, 0), (4, 5), (3, 9), (2, 14), (1, 19)}


def test_chord_uses_distinct_strings():
    # Open E major: E2 B2 E3 G#3 B3 E4
    notes = [Note(p, 0.0, 1.0) for p in (40, 47, 52, 56, 59, 64)]
    placed = assign_tab(notes)
    assert len({n.string for n in placed}) == 6
    for note in placed:
        assert fret_to_pitch(note.string, note.fret) == note.pitch
    assert [n.fret for n in placed] == [0, 2, 2, 1, 0, 0]


def test_valid_existing_positions_are_kept():
    (note,) = assign_tab([Note(64, 0.0, 1.0, string=2, fret=14)])
    assert (note.string, note.fret) == (2, 14)


def test_melody_stays_in_one_position():
    # A-minor pentatonic run around the 5th fret: no jumps to open strings far away.
    pitches = [57, 60, 62, 64, 67, 69]
    notes = [Note(p, i * 0.25, i * 0.25 + 0.2) for i, p in enumerate(pitches)]
    frets = [n.fret for n in assign_tab(notes)]
    fretted = [f for f in frets if f > 0]
    assert max(fretted) - min(fretted) <= 4


def test_model_evidence_steers_the_choice():
    note = Note(64, 0.0, 1.0)
    default = assign_tab([note])[0]
    believes_g9 = [{(5, 0): 0.05, (4, 5): 0.05, (3, 9): 0.8, (2, 14): 0.05, (1, 19): 0.05}]
    steered = assign_tab([note], position_probs=believes_g9)[0]
    assert (default.string, default.fret) != (3, 9)
    assert (steered.string, steered.fret) == (3, 9)


def test_model_evidence_cannot_break_chords():
    # The model wants both notes on the B string; a chord still needs distinct strings.
    chord = [Note(64, 0.0, 1.0), Note(71, 0.0, 1.0)]
    probs = [{(4, 5): 0.9, (5, 0): 0.1}, {(4, 12): 0.9, (5, 7): 0.1}]
    placed = assign_tab(chord, position_probs=probs)
    assert placed[0].string != placed[1].string
    for note in placed:
        assert fret_to_pitch(note.string, note.fret) == note.pitch


def test_unplayable_note_is_left_unplaced():
    (note,) = assign_tab([Note(30, 0.0, 1.0)])
    assert note.string is None and note.fret is None


def test_tab_to_ascii_keeps_close_notes_apart():
    # 5 then 6 on the G string, 0.1 s apart (< one column): must not read as fret 56.
    notes = [Note(60, 0.0, 0.1, string=3, fret=5), Note(61, 0.1, 0.2, string=3, fret=6)]
    g_line = next(line for line in tab_to_ascii(notes).splitlines() if line.startswith("G"))
    assert "5-6" in g_line and "56" not in g_line


def test_tab_to_ascii_aligns_chords():
    chord = [Note(52, 1.0, 2.0, string=2, fret=2), Note(57, 1.0, 2.0, string=3, fret=2)]
    lines = tab_to_ascii(chord).splitlines()
    d_line = next(line for line in lines if line.startswith("D"))
    g_line = next(line for line in lines if line.startswith("G"))
    assert d_line.index("2") == g_line.index("2")


def test_tab_to_ascii_never_splits_a_fret_across_lines():
    notes = [Note(64 + 10, i * 0.5, i * 0.5 + 0.4, string=5, fret=10) for i in range(40)]
    for block in tab_to_ascii(notes, line_width=13).split("\n\n"):
        e_line = block.splitlines()[0]
        body = e_line[e_line.index("|") + 1 : -1]
        assert "10" in body and not body.startswith("0") and not body.endswith("1")


def test_tab_to_ascii_renders_frets():
    text = tab_to_ascii(
        [Note(64, 0.0, 0.5, string=5, fret=0), Note(66, 0.5, 1.0, string=5, fret=2)]
    )
    first_line = text.splitlines()[0]
    assert first_line.startswith("e")
    assert "0" in first_line and "2" in first_line
