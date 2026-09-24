"""Core music primitives shared by the ML pipeline and the API.

Installed as the ``music_core`` package (see ``[tool.setuptools]`` in the root pyproject).
"""

from music_core.notes import Note, hz_to_midi, midi_to_hz, midi_to_name, name_to_midi, sort_notes
from music_core.tab import STANDARD_TUNING, assign_tab, pitch_to_positions, tab_to_ascii

__all__ = [
    "Note",
    "STANDARD_TUNING",
    "assign_tab",
    "hz_to_midi",
    "midi_to_hz",
    "midi_to_name",
    "name_to_midi",
    "pitch_to_positions",
    "sort_notes",
    "tab_to_ascii",
]
