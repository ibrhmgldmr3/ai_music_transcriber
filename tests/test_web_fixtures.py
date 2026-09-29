"""Shared expectations for code the web app mirrors in TypeScript (apps/web/lib/music.ts).

The fixture holds what the Python implementation produces; apps/web/lib/music.test.ts
checks the TypeScript copy against the same file. After changing either side on
purpose, regenerate it with ``python tests/test_web_fixtures.py``.
"""

import json
import math
from pathlib import Path

from music_core.analysis import KEY_NAMES, Key
from music_core.musicxml import DIVISIONS, _first_bar_line, _spell
from music_core.notes import Note
from music_core.tab import TUNINGS

FIXTURE = Path(__file__).parent / "fixtures" / "music_web.json"

# (tempo, beats per measure, downbeat, first note onset)
GRID_CASES = [
    (120.0, 4, 0.0, 0.0),
    (120.0, 4, 0.3, 0.05),
    (120.0, 4, 1.7, 0.4),
    (97.5, 3, 0.9, 2.6),
    (60.0, 4, 1.0, 0.0),
    (140.0, 7, 0.21, 5.3),
]


def _name(midi: int, key: Key | None) -> str:
    step, alter, octave = _spell(midi, key)
    return f"{step}{'#' * alter if alter > 0 else 'b' * -alter}{octave}"


def build() -> dict:
    keys = [None, *(Key.parse(name) for name in KEY_NAMES)]
    spelling = {
        (key.name if key else "none"): {
            "tonic": key.tonic if key else None,
            "mode": key.mode if key else None,
            "fifths": key.fifths if key else 0,
            "names": [_name(midi, key) for midi in range(36, 97)],
        }
        for key in keys
    }
    first_bars = []
    for tempo, beats, downbeat, first in GRID_CASES:
        unit = 60.0 / tempo / DIVISIONS
        origin = _first_bar_line([Note(60, first, first + 1)], downbeat, unit, DIVISIONS * beats)
        first_bars.append(
            {
                "tempo": tempo,
                "beats_per_measure": beats,
                "downbeat": downbeat,
                "first_note": first,
                "bar1_time": round(origin, 9),
            }
        )
    return {
        "key_names": list(KEY_NAMES),
        "tunings": {name: list(strings) for name, strings in TUNINGS.items()},
        "midi_range": [36, 96],
        "spelling": spelling,
        "first_bars": first_bars,
    }


def test_fixture_matches_python():
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    current = build()
    for case, expected in zip(current["first_bars"], stored["first_bars"]):
        assert math.isclose(case.pop("bar1_time"), expected.pop("bar1_time"), abs_tol=1e-6)
    assert stored == current, "Regenerate with: python tests/test_web_fixtures.py"


if __name__ == "__main__":
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE}")
