"""EGDB (Chen et al., ICASSP 2022): electric guitar clips with per-string labels.

240 clips played by one guitarist through a hexaphonic pickup, each as a direct-input
(DI) recording and rendered through five amplifiers. Layout, as written by
``scripts/download_egdb.py``::

    audio_DI/1.wav  audio_Marshall/1.wav  ...  audio_label/1.midi

The five ``RealData`` recordings are not used: their MIDI tracks are named
``clip1_pred`` etc. and carry no strings, so they look like the paper's model output
rather than reference transcriptions.

Labels: MIDI channel 0-5 is string 1 (high e) to 6. Track names are missing or
inconsistent in most files, while the channel gives a playable fret for 99.8% of notes;
the rest keep their pitch but no position.

Timing: many clips carry the tablature's quantized timing rather than the performance's
(in the median clip half of the onsets sit exactly on a 16th-note grid, versus 2% in
Guitar-TECHS' pickup MIDI), and a few clips (e.g. 40, 69, 116) don't match their audio at
all. Training the note model on these labels lowered note F1, EGDB's own included.
``align_notes`` moves the onsets to a GuitarSet-trained note model's onsets in the direct
input (``scripts/align_egdb.py`` stores the result); the ``egdb_aligned`` dataset uses
those labels and leaves out the clips that don't match their audio. The original labels
keep training the tab model, whose per-frame string/fret targets tolerate the timing
(see ml/configs/guitar_tab_mixed.yaml).

Splits follow the dataset's README: clips 216 and up are the test clips, and the
JCjazz and Plexi amplifiers are test-only tones. Clips 201-215 are held out here for
validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ml.preprocessing.annotations import load_string_midi
from ml.preprocessing.audio import load_audio
from music_core.notes import Note

TRAIN_TONES = ("DI", "Marshall", "Ftwin", "Mesa")
TEST_TONES = ("JCjazz", "Plexi")
VAL_TONES = ("DI", "Marshall")
FIRST_VAL_CLIP = 201
FIRST_TEST_CLIP = 216


@dataclass(frozen=True)
class EGDBTrack:
    track_id: str  # "egdb_001_DI"
    audio_path: Path
    label_path: Path
    clip: int
    tone: str  # amplifier or "DI"


def find_tracks(raw_dir: str | Path) -> list[EGDBTrack]:
    """Every clip in every tone that has a label."""
    raw_dir = Path(raw_dir)
    tracks: list[EGDBTrack] = []
    labels = {int(p.stem): p for p in (raw_dir / "audio_label").glob("*.midi") if p.stem.isdigit()}
    for tone in TRAIN_TONES + TEST_TONES:
        for audio in sorted((raw_dir / f"audio_{tone}").glob("*.wav")):
            clip = int(audio.stem) if audio.stem.isdigit() else None
            if clip in labels:
                tracks.append(EGDBTrack(f"egdb_{clip:03d}_{tone}", audio, labels[clip], clip, tone))
    tracks.sort(key=lambda t: (t.clip, t.tone))
    return tracks


def split_tracks(tracks: list[EGDBTrack]) -> dict[str, list[str]]:
    """train / val / test (unseen clips, every tone)."""
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for track in tracks:
        if track.clip >= FIRST_TEST_CLIP:
            splits["test"].append(track.track_id)
        elif track.tone in TEST_TONES:
            continue  # test tones are never trained or validated on
        elif track.clip >= FIRST_VAL_CLIP:
            if track.tone in VAL_TONES:
                splits["val"].append(track.track_id)
        else:
            splits["train"].append(track.track_id)
    return splits


def load_track(track: EGDBTrack, cfg: dict[str, Any]) -> tuple[np.ndarray, list[Note]]:
    audio = cfg["audio"]
    y = load_audio(track.audio_path, audio["sample_rate"], mono=True, normalize=audio["normalize"])
    return y, load_string_midi(track.label_path, cfg["tab"]["tuning"], string_from="channel")


# ------------------------------------------------------------------------ aligned labels

ALIGNED_DIR = Path("ml/data/interim/egdb_aligned")  # <clip>.npz from scripts/align_egdb.py
ALIGN_WINDOW = 0.06  # s: a label moves to a transcribed onset of its pitch this close


def align_notes(
    notes: list[Note], transcribed: list[Note], window: float = ALIGN_WINDOW
) -> tuple[list[Note], float]:
    """Move each labeled onset to the nearest onset of the same pitch in ``transcribed``
    (a note model's output for the clip) within ``window``, keeping the duration.

    The labels' timing scatters around the audio's (against a GuitarSet-trained model:
    median +3 ms, but a quarter of the onsets more than 25 ms away), while the note
    model's onsets follow the precisely timed GuitarSet and Guitar-TECHS labels. Notes
    the model missed keep their labeled time, and no transcribed note is added, so the
    labels still decide which notes there are. Returns the notes and the share moved.
    """
    by_pitch: dict[int, list[float]] = {}
    for note in transcribed:
        by_pitch.setdefault(note.pitch, []).append(note.start)
    aligned, moved = [], 0
    for note in notes:
        candidates = [t for t in by_pitch.get(note.pitch, []) if abs(t - note.start) <= window]
        start = min(candidates, key=lambda t: abs(t - note.start)) if candidates else note.start
        moved += bool(candidates)
        shift = start - note.start
        aligned.append(
            Note(note.pitch, start, max(start + 0.03, note.end + shift), note.velocity,
                 note.string, note.fret)
        )  # fmt: skip
    return aligned, moved / len(notes) if notes else 0.0


def load_aligned_track(
    track: EGDBTrack, cfg: dict[str, Any], aligned_dir: Path = ALIGNED_DIR
) -> tuple[np.ndarray, list[Note]]:
    """Like ``load_track`` with the aligned labels of ``scripts/align_egdb.py``."""
    from ml.preprocessing.annotations import array_to_notes

    audio = cfg["audio"]
    y = load_audio(track.audio_path, audio["sample_rate"], mono=True, normalize=audio["normalize"])
    with np.load(aligned_dir / f"{track.clip}.npz") as data:
        return y, array_to_notes(data["notes"])


def aligned_clips(aligned_dir: Path = ALIGNED_DIR) -> set[int]:
    """Clips whose aligned labels match their audio (``scripts/align_egdb.py`` keeps them)."""
    path = aligned_dir / "kept.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/align_egdb.py first")
    return {int(line) for line in path.read_text(encoding="utf-8").split()}


def find_aligned_tracks(raw_dir: str | Path) -> list[EGDBTrack]:
    """The kept clips; ``raw_dir`` is ``.../raw/egdb_aligned``, the audio stays in ``raw/egdb``."""
    keep = aligned_clips()
    return [t for t in find_tracks(Path(raw_dir).parent / "egdb") if t.clip in keep]


def augmentable(track: EGDBTrack) -> bool:
    """Room variants make sense for an amplifier, not for the direct input."""
    return track.tone != "DI"


# The ``egdb_aligned`` dataset of scripts/prepare_dataset.py: aligned timing, for the note model.
ALIGNED = SimpleNamespace(
    find_tracks=find_aligned_tracks,
    split_tracks=split_tracks,
    load_track=load_aligned_track,
    augmentable=augmentable,
)
