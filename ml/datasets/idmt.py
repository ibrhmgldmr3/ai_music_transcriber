"""IDMT-SMT-Guitar (Kehling et al., DAFx 2014): guitar notes labeled with how they were played.

Used to measure playing-technique detection (``ml.inference.techniques``), not for
training. Subset 2 has 261 takes on three electric guitars (AR: Aristides 010, FS: Fender
Strat, LP: Gibson Les Paul), one player: chromatic runs on every string, isolated
bends, slides, vibrato, harmonics and dead notes, and licks mixing them. Every note is
labeled with its string and fret, a plucking ("excitation") style and an expression
style:

    NO normal   BE bend   SL slide   VI vibrato   HA harmonic   DN dead note

``pitch`` is the fretted note (a bent note keeps the pitch it starts from). A slide is
labeled on the note that slides, with ``modulationFrequencyRange`` the slide in cents
(+200: two semitones up); where it lands is not a note of its own. Bends and vibrato
carry no amount. Strings are numbered from the low E (1) up.

Layout after unzipping the Zenodo archive (record 7544110) into ``ml/data/raw/idmt``::

    IDMT-SMT-GUITAR_V2/dataset2/annotation/AR_Lick11_FNVSBHD.xml
    IDMT-SMT-GUITAR_V2/dataset2/audio/AR_Lick11_FNVSBHD.wav

License CC BY-NC-ND 4.0 (non-commercial, no derivatives).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from music_core.notes import Note
from music_core.tab import STANDARD_TUNING

ROOT = Path("ml/data/raw/idmt/IDMT-SMT-GUITAR_V2")
GUITARS = ("AR", "FS", "LP")
EXPRESSIONS = {
    "NO": "normal",
    "BE": "bend",
    "SL": "slide",
    "VI": "vibrato",
    "HA": "harmonic",
    "DN": "dead note",
}


@dataclass(frozen=True)
class IdmtNote:
    note: Note  # pitch (fretted), onset, offset, string (0 = low E) and fret
    expression: str  # NO, BE, SL, VI, HA or DN
    excitation: str  # PK pick, FS finger, MU muted
    slide_cents: int | None = None  # SL only: how far the note slides, + = up


@dataclass(frozen=True)
class IdmtTake:
    name: str  # "AR_Lick11_FNVSBHD"
    audio_path: Path
    annotation_path: Path

    @property
    def guitar(self) -> str:
        return self.name.split("_", 1)[0]


def find_takes(root: str | Path = ROOT, guitars: tuple[str, ...] = GUITARS) -> list[IdmtTake]:
    """Subset 2's takes with both audio and annotation."""
    root = Path(root) / "dataset2"
    audio = {p.stem: p for p in (root / "audio").glob("*.wav")}
    takes = []
    for xml in sorted((root / "annotation").glob("*.xml")):
        # Lick 11's annotations name its techniques (AR_Lick11_FNVSBHD), its audio
        # doesn't (AR_Lick11_FN).
        match = audio.get(xml.stem) or next(
            (path for stem, path in audio.items() if xml.stem.startswith(stem + "VS")), None
        )
        if match is not None and xml.stem.split("_", 1)[0] in guitars:
            takes.append(IdmtTake(xml.stem, match, xml))
    return takes


def load_annotation(path: str | Path) -> list[IdmtNote]:
    """The notes of one take, sorted by onset."""
    notes = []
    for event in ET.parse(path).getroot().iter("event"):
        fields = {child.tag: (child.text or "").strip() for child in event}
        string = int(fields["stringNumber"]) - 1
        fret = int(fields["fretNumber"])
        pitch = int(fields["pitch"])
        placed = 0 <= string < len(STANDARD_TUNING) and STANDARD_TUNING[string] + fret == pitch
        expression = fields.get("expressionStyle", "NO")
        cents = fields.get("modulationFrequencyRange")
        notes.append(
            IdmtNote(
                note=Note(
                    pitch=pitch,
                    start=float(fields["onsetSec"]),
                    end=max(float(fields["offsetSec"]), float(fields["onsetSec"])),
                    string=string if placed else None,
                    fret=fret if placed else None,
                ),
                expression=expression,
                excitation=fields.get("excitationStyle", ""),
                slide_cents=int(float(cents)) if expression == "SL" and cents else None,
            )
        )
    return sorted(notes, key=lambda n: (n.note.start, n.note.pitch))
