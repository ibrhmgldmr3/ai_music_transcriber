"""Guitar-TECHS (ICASSP 2025): electric guitar recorded four ways, per-string MIDI labels.

Three guitarists; chords, scales, single notes and techniques (P1, P2) and musical
excerpts (P3). Every take exists as direct input, a microphoned amplifier and two room
recordings from the videos: head-mounted ("ego") and 1.5 m in front ("exo"). Labels come
from a Fishman Triple Play MIDI pickup, one track per string named e, B, G, D, A, E
(the channels are not reliable: some takes put every string on channel 0). Layout after
unzipping the Zenodo archives::

    P1_chords/midi/midi_01.mid
    P1_chords/audio/directinput/directinput_01.wav   .../audio/micamp/micamp_01.wav
    P1_chords/video/ego/ego_01.mp3                   .../video/exo/exo_01.mp3

Timing: the MIDI is up to ~65 ms off the direct input in some takes and the video
soundtracks start up to ~50 ms off it too, so labels are first aligned to the direct
input (onsets against its onset envelope), then shifted by each recording's offset from
the direct input. Splits: P3 (unseen player, real music) is the test set; every tenth
take of P1/P2 is validation.

Known label noise: in the chord takes the pickup reports the lower strings late (on the
A and low E strings 10% of the onsets come 120-130 ms after the strum, against 20-40 ms
elsewhere). Training without the chord takes was tried and was worse on every test set,
so they stay in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.preprocessing.annotations import load_string_midi
from ml.preprocessing.audio import load_audio
from music_core.notes import Note

# version -> (folder under the take's section, file prefix, extension)
VERSIONS: dict[str, tuple[str, str, str]] = {
    "di": ("audio/directinput", "directinput", ".wav"),
    "amp": ("audio/micamp", "micamp", ".wav"),
    "ego": ("video/ego", "ego", ".mp3"),
    "exo": ("video/exo", "exo", ".mp3"),
}
SYNCED_VERSIONS = ("di", "amp")  # recorded on the same interface as the direct input
TEST_PLAYER = "P3"
VAL_EVERY = 10
MAX_OFFSET = 0.25  # seconds; larger measured offsets mean the alignment failed
_SECTION = re.compile(r"^(P\d)_(\w+)$")


@dataclass(frozen=True)
class GuitarTechsTrack:
    track_id: str  # "gtechs_P1_chords_01_exo"
    audio_path: Path
    label_path: Path
    reference_path: Path  # direct input of the same take, for aligning room recordings
    player: str
    section: str  # chords, scales, singlenotes, techniques, music
    take: str
    version: str  # di | amp | ego | exo


def find_tracks(raw_dir: str | Path) -> list[GuitarTechsTrack]:
    raw_dir = Path(raw_dir)
    tracks: list[GuitarTechsTrack] = []
    for section_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        match = _SECTION.match(section_dir.name)
        if not match:
            continue
        player, section = match.groups()
        for label in sorted((section_dir / "midi").glob("midi_*.mid")):
            take = label.stem.removeprefix("midi_")
            reference = _audio_path(section_dir, "di", take)
            if not reference.exists():
                continue
            for version in VERSIONS:
                audio = _audio_path(section_dir, version, take)
                if audio.exists():
                    track_id = f"gtechs_{player}_{section}_{take}_{version}"
                    tracks.append(
                        GuitarTechsTrack(
                            track_id, audio, label, reference, player, section, take, version
                        )
                    )
    return tracks


def _audio_path(section_dir: Path, version: str, take: str) -> Path:
    folder, prefix, extension = VERSIONS[version]
    return section_dir / folder / f"{prefix}_{take}{extension}"


def split_tracks(tracks: list[GuitarTechsTrack]) -> dict[str, list[str]]:
    """test: the unseen player; val: every VAL_EVERY-th take (all versions); train: rest."""
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    takes = sorted({(t.player, t.section, t.take) for t in tracks if t.player != TEST_PLAYER})
    val_takes = set(takes[VAL_EVERY - 1 :: VAL_EVERY])
    for track in tracks:
        if track.player == TEST_PLAYER:
            splits["test"].append(track.track_id)
        elif (track.player, track.section, track.take) in val_takes:
            splits["val"].append(track.track_id)
        else:
            splits["train"].append(track.track_id)
    return splits


def load_track(track: GuitarTechsTrack, cfg: dict[str, Any]) -> tuple[np.ndarray, list[Note]]:
    audio = cfg["audio"]
    sample_rate = audio["sample_rate"]
    y = load_audio(track.audio_path, sample_rate, mono=True, normalize=audio["normalize"])
    notes = load_string_midi(track.label_path, cfg["tab"]["tuning"], string_from="track")
    reference = (
        y
        if track.version == "di"
        else load_audio(track.reference_path, sample_rate, mono=True, normalize=True)
    )
    offset = label_offset(notes, reference, sample_rate)
    if track.version not in SYNCED_VERSIONS:
        offset += estimate_offset(reference, y, sample_rate)
    if abs(offset) > 2 * MAX_OFFSET:
        raise ValueError(f"{track.track_id}: could not align the labels to the audio")
    return y, _shift(notes, offset)


_HOP = 128


def _envelope(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=_HOP)


def _best_lag(a: np.ndarray, b: np.ndarray, sample_rate: int) -> float:
    """Seconds by which ``b`` lags ``a``, within +/- MAX_OFFSET (FFT cross-correlation)."""
    size = max(len(a), len(b))
    a, b = (np.pad(x - x.mean(), (0, 2 * size - len(x))) for x in (a, b))
    correlation = np.fft.irfft(np.fft.rfft(b) * np.conj(np.fft.rfft(a)), n=2 * size)
    max_lag = int(MAX_OFFSET * sample_rate / _HOP)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    values = np.concatenate([correlation[: max_lag + 1], correlation[-max_lag:]])
    return float(lags[int(np.argmax(values))] * _HOP / sample_rate)


def estimate_offset(reference: np.ndarray, other: np.ndarray, sample_rate: int) -> float:
    """Seconds by which ``other`` lags ``reference`` (positive: events happen later).

    Cross-correlates the onset-strength envelopes, which is robust to the very different
    timbre of a direct input and a room microphone.
    """
    return _best_lag(_envelope(reference, sample_rate), _envelope(other, sample_rate), sample_rate)


def label_offset(notes: list[Note], audio: np.ndarray, sample_rate: int) -> float:
    """Seconds to add to the note times so their onsets line up with the audio's."""
    envelope = _envelope(audio, sample_rate)
    impulses = np.zeros_like(envelope)
    for note in notes:
        frame = int(round(note.start * sample_rate / _HOP))
        if 0 <= frame < len(impulses):
            impulses[frame] = 1.0
    return _best_lag(impulses, envelope, sample_rate) if impulses.any() else 0.0


def _shift(notes: list[Note], offset: float) -> list[Note]:
    shifted = []
    for note in notes:
        start, end = note.start + offset, note.end + offset
        if end > 0:
            shifted.append(
                Note(
                    note.pitch,
                    max(0.0, start),
                    end,
                    note.velocity,
                    note.string,
                    note.fret,
                )
            )
    return shifted
