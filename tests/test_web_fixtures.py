"""Shared expectations for code the web app mirrors in TypeScript (apps/web/lib/music.ts).

The fixture holds what the Python implementation produces; apps/web/lib/music.test.ts
checks the TypeScript copy against the same file. After changing either side on
purpose, regenerate it with ``python tests/test_web_fixtures.py``.
"""

import json
from pathlib import Path

from music_core.analysis import KEY_NAMES, Key
from music_core.musicxml import _spell
from music_core.tab import TUNINGS

FIXTURE = Path(__file__).parent / "fixtures" / "music_web.json"


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
    return {
        "key_names": list(KEY_NAMES),
        "tunings": {name: list(strings) for name, strings in TUNINGS.items()},
        "midi_range": [36, 96],
        "spelling": spelling,
    }


def test_fixture_matches_python():
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert stored == build(), "Regenerate with: python tests/test_web_fixtures.py"


if __name__ == "__main__":
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE}")
