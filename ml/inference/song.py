"""Song mode: a whole song -> its chords, bars and sung melody, set for guitar.

The chords and beats come from the full mix (``ml.inference.chords``: BTC and Beat This!,
the conditions they were trained on); the melody from the vocals Demucs separates,
through the voice transcriber (``ml.inference.voice``). The key is estimated from both,
section by section when the song modulates (``music_core.analysis.segment_keys``).
Unless a capo is given, the one that turns the chords into the easiest shapes is chosen
(``music_core.guitar_chords.suggest_capo``), and the melody is placed on the fretboard
in that position, moved by octaves if it doesn't fit.

A chord held over its third, fifth or seventh in the bass for two beats or more, the
bass clear in each beat, becomes a slash chord (D/F#): on AAM, whose bass plays the
root, that adds no wrong ones, and on GuitarSet it matches a few of the inversions
played (scripts/benchmark_chords.py --bass). When a guitar note model is given and the
song has a guitar, the chords it plays in the guitar stem are the strums of the
strumming pattern (``music_core.rhythm``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ml.inference.chords import (
    BeatTracker,
    ChordRecognizer,
    ChordSegment,
    add_bass,
    beat_bass,
    beat_chords,
)
from ml.inference.predict import ProgressCallback, TranscriptionResult
from ml.inference.voice import VoiceSettings, fit_to_range, pitch_curve, segment_notes, track_pitch
from ml.preprocessing.audio import load_audio
from music_core.analysis import Key, estimate_key, segment_keys
from music_core.guitar_chords import QUALITIES, ChordSymbol, suggest_capo
from music_core.notes import Note
from music_core.rhythm import strums_from_notes
from music_core.tab import DEFAULT_NUM_FRETS, assign_tab, open_strings

# Stored as the project's model_version; bump it when the method changes the result.
SONG_VERSION = "song-btc-beatthis@2"
SAMPLE_RATE = 44100
CHORD_CHANGE_PENALTY = 1.0  # best on GuitarSet comp and AAM (scripts/benchmark_chords.py)
# Slash chords: a bass held this clearly (share of the beat's bass) for this many beats.
# Looser (0.5, any length) added wrong slash chords on both AAM and GuitarSet.
SLASH_SHARE = 0.8
SLASH_BEATS = 2
GUITAR_LEVEL = 0.1  # the guitar stem's share of the mix's level for a strumming pattern


@dataclass
class SongResult(TranscriptionResult):
    chords: list[dict[str, Any]] = field(default_factory=list)  # {start, end, label}
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    capo: int = 0
    beats_per_measure: int = 4
    downbeat: float | None = None
    key: str | None = None  # e.g. "A minor", the song's main key
    keys: list[dict[str, Any]] = field(default_factory=list)  # {start, end, key} if it modulates
    strums: list[tuple[float, float]] | None = None  # (time, strength) of the guitar's chords


def transcribe_song(
    audio_path: str | Path,
    separator: Callable[..., Any],
    chord_recognizer: ChordRecognizer,
    beat_tracker: BeatTracker,
    tuning_name: str = "standard",
    capo: int | None = None,
    num_frets: int = DEFAULT_NUM_FRETS,
    progress: ProgressCallback | None = None,
    s: VoiceSettings = VoiceSettings(),
    guitar_model: Any | None = None,
) -> SongResult:
    """Chords, bars, keys and the sung melody of a song; ``capo=None`` picks one.

    ``separator`` has a ``stems(y, sample_rate, names)`` method (``GuitarSeparator``);
    ``guitar_model`` (a ``Predictor``) transcribes the guitar stem for the strums.
    """
    import librosa

    report = progress or (lambda fraction, stage: None)
    report(0.0, "loading")
    y = load_audio(audio_path, SAMPLE_RATE, mono=True, normalize=True)
    duration = len(y) / SAMPLE_RATE
    report(0.05, "separating")
    stems = separator.stems(y, SAMPLE_RATE, ["vocals", "guitar"])
    report(0.5, "transcribing")
    beats, downbeats = beat_tracker(y, SAMPLE_RATE)
    report(0.6, "transcribing")
    probs = chord_recognizer.probabilities(
        y, SAMPLE_RATE, lambda f: report(0.6 + 0.15 * f, "transcribing")
    )
    segments = beat_chords(
        probs, chord_recognizer.frame_rate, beats, duration, CHORD_CHANGE_PENALTY
    )
    segments = add_bass(
        segments,
        beat_bass(y, SAMPLE_RATE, beats, duration),
        beats,
        duration,
        min_beats=SLASH_BEATS,
        min_share=SLASH_SHARE,
    )
    voice = librosa.resample(stems["vocals"], orig_sr=SAMPLE_RATE, target_sr=s.sample_rate)
    peak = float(np.abs(voice).max()) if voice.size else 0.0
    track = track_pitch(
        voice / peak if peak > 0 else voice,
        s,
        lambda f: report(0.75 + 0.15 * f, "transcribing"),
    )
    report(0.9, "finishing")
    strums = None
    if guitar_model is not None and _level(stems["guitar"], y) >= GUITAR_LEVEL:
        guitar = guitar_model.transcribe_signal(stems["guitar"], estimate_tempo=False)
        strums = strums_from_notes(guitar.notes)
    report(0.97, "finishing")

    symbols = _symbols(segments)
    sung = segment_notes(track, s)
    key = song_key(sung, symbols)
    keys = segment_keys([*sung, *_chord_tones(symbols)], downbeats)
    if capo is None:
        capo = suggest_capo(
            [(c, seg.end - seg.start) for seg, c in symbols], open_strings(tuning_name)
        )[0]
    tuning = list(open_strings(tuning_name, capo))
    melody, transpose = fit_to_range(segment_notes(track, s), min(tuning), max(tuning) + num_frets)
    melody = assign_tab(melody, tuning, num_frets)
    tempo, beats_per_measure, downbeat = meter(beats, downbeats)
    return SongResult(
        notes=melody,
        duration=duration,
        tempo=tempo,
        tuning=tuning,
        transpose=transpose,
        pitch_curve=pitch_curve(track, s, transpose),
        chords=[
            {"start": round(seg.start, 3), "end": round(seg.end, 3), "label": seg.label}
            for seg in segments
            if seg.label not in ("N", "X")
        ],
        beats=[round(float(b), 3) for b in beats],
        downbeats=[round(float(b), 3) for b in downbeats],
        capo=capo,
        beats_per_measure=beats_per_measure,
        downbeat=downbeat,
        key=key.name if key else None,
        keys=[{"start": round(k.start, 3), "end": round(k.end, 3), "key": k.key.name} for k in keys]
        if len(keys) > 1
        else [],
        strums=strums,
    )


def _level(stem: np.ndarray, mix: np.ndarray) -> float:
    """A stem's RMS level as a share of the mix's."""
    mix_rms = float(np.sqrt(np.mean(mix**2)))
    return float(np.sqrt(np.mean(stem**2))) / mix_rms if mix_rms > 0 else 0.0


def _symbols(segments: Sequence[ChordSegment]) -> list[tuple[ChordSegment, ChordSymbol]]:
    pairs = []
    for segment in segments:
        symbol = ChordSymbol.parse(segment.label)
        if symbol is not None:
            pairs.append((segment, symbol))
    return pairs


def _chord_tones(chords: Sequence[tuple[ChordSegment, ChordSymbol]]) -> list[Note]:
    """The chords' tones as notes held as long as each chord sounds."""
    return [
        Note(48 + (symbol.root + interval) % 12, segment.start, segment.end)
        for segment, symbol in chords
        for interval in QUALITIES[symbol.quality][0]
    ]


def song_key(
    melody: Sequence[Note], chords: Sequence[tuple[ChordSegment, ChordSymbol]]
) -> Key | None:
    """The key from the melody and the chords' tones, each held as long as it sounds."""
    return estimate_key([*melody, *_chord_tones(chords)])


def meter(
    beats: Sequence[float], downbeats: Sequence[float]
) -> tuple[float | None, int, float | None]:
    """Tempo (BPM, from the median beat), beats per bar (the most common count between bar
    lines, 2-7) and the first bar line."""
    beats = np.asarray(beats, dtype=float)
    tempo = round(60.0 / float(np.median(np.diff(beats))), 2) if len(beats) > 2 else None
    counts = Counter(
        int(np.sum((beats >= a - 1e-3) & (beats < b - 1e-3)))
        for a, b in zip(downbeats, downbeats[1:])
    )
    beats_per_measure = next((n for n, _ in counts.most_common() if 2 <= n <= 7), 4)
    first = float(downbeats[0]) if len(downbeats) else (float(beats[0]) if len(beats) else None)
    return tempo, beats_per_measure, first
