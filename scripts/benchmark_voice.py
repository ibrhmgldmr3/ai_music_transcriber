"""Voice mode on real singing: VocalSet with the Annotated-VocalSet note labels.

VocalSet (Wilkins et al., 2018; CC BY 4.0) has 20 singers performing scales, arpeggios,
long tones and three short songs in several techniques; Annotated-VocalSet (Faghih &
Timoney, 2022; CC BY 4.0) marks every sung note's start, end and written pitch.

    https://zenodo.org/records/1442513  -> VocalSet11.zip
    https://zenodo.org/records/7061507  -> Annotated VocalSet.zip

Both zips are read in place. The labels give the *written* pitch, and many takes are
sung a semitone (or an octave) away from it, so notes are scored key-independently:
each take at the whole-semitone transposition that fits it best (``key_f1``, onsets
within 50 ms; ``key_f1_100ms`` allows 100 ms). The labels start a legato note where its
pitch has settled, after the slide into it, which is a matter of convention within
about 50 ms. ``f1`` is the plain score and ``onset_f1`` ignores pitch. Trills, lip
trills and speech are
left out: their notes aren't defined by the labels the way the others are. Singers
f1-f3 and m1-m3 are for trying settings (``--split dev``), the other 14 for reporting.

python scripts/benchmark_voice.py --vocalset ml/data/raw/vocalset/VocalSet11.zip \
    --annotations ml/data/raw/vocalset/annotated_vocalset.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm import tqdm

from ml.evaluation.metrics import note_metrics
from ml.inference.voice import PitchTrack, VoiceSettings, segment_notes, track_pitch
from music_core.notes import Note

LABELS = "Annotated VocalSet/extended 1/with file header/"
SKIPPED = {"trill", "trillo", "lip_trill", "spoken"}
DEV_SINGERS = {"f1", "f2", "f3", "m1", "m2", "m3"}
SONGS = {"row": "songs", "caro": "songs", "dona": "songs"}


@dataclass
class Take:
    name: str  # e.g. f1_arpeggios_belt_c_a
    singer: str
    kind: str  # scales, arpeggios, long tones, songs
    technique: str
    notes: list[Note]


def read_labels(path: Path) -> list[Take]:
    takes = []
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not (member.startswith(LABELS) and member.endswith(".csv")):
                continue
            technique = member[len(LABELS) :].split("/")[0]
            if technique in SKIPPED:
                continue
            lines = z.read(member).decode("utf-8", "replace").splitlines()
            header = next(i for i, line in enumerate(lines) if line.startswith("Sequence"))
            columns = [c.strip() for c in lines[header].split(",")]
            notes = []
            for row in csv.reader(lines[header + 1 :]):
                cell = dict(zip(columns, (value.strip() for value in row)))
                if cell.get("Type") != "Sound":
                    continue
                try:
                    pitch = int(float(cell["Ground Truth MIDI code"]))
                    start, end = float(cell["Start time"]), float(cell["End time"])
                except (KeyError, ValueError):
                    continue
                if end > start:
                    notes.append(Note(pitch, start, end))
            name = PurePosixPath(member).stem
            singer, kind = name.split("_")[:2]
            kind = SONGS.get(kind.strip(), "long tones" if kind.strip() == "long" else kind)
            if notes:
                takes.append(
                    Take(name, singer, kind.replace("arepggios", "arpeggios"), technique, notes)
                )
    return takes


def audio_index(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        return {
            PurePosixPath(m).stem: m
            for m in z.namelist()
            if m.lower().endswith(".wav") and "__MACOSX" not in m
        }


def load_take(z: zipfile.ZipFile, member: str, sample_rate: int) -> np.ndarray:
    import librosa
    import soundfile as sf

    y, sr = sf.read(io.BytesIO(z.read(member)), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
    peak = float(np.abs(y).max()) or 1.0
    return (0.95 * y / peak).astype(np.float32)


def score(reference: list[Note], estimated: list[Note]) -> dict[str, float]:
    import mir_eval

    plain = note_metrics(reference, estimated)["f1"]
    # The take's transposition: the most common pitch difference of notes with nearby
    # onsets (trying all 27 shifts gives the same result, much slower).
    shifts = Counter(
        ref.pitch - est.pitch
        for ref in reference
        for est in estimated
        if abs(ref.start - est.start) <= 0.15
    )
    k = shifts.most_common(1)[0][0] if shifts else 0
    shifted = [replace(n, pitch=n.pitch + k) for n in estimated]
    key = max(plain, note_metrics(reference, shifted)["f1"])
    key_100ms = note_metrics(reference, shifted, onset_tolerance=0.1)["f1"]
    if estimated:
        onsets = mir_eval.transcription.onset_precision_recall_f1(
            np.array([[n.start, n.end] for n in reference]),
            np.array([[n.start, max(n.end, n.start + 1e-3)] for n in estimated]),
            onset_tolerance=0.05,
        )[2]
    else:
        onsets = 0.0
    return {"key_f1": key, "key_f1_100ms": key_100ms, "f1": plain, "onset_f1": float(onsets)}


def pitch_tracks(
    takes: list[Take], vocalset: Path, s: VoiceSettings, cache: Path | None = None
) -> dict[str, PitchTrack]:
    """Pitch tracks of the takes (the slow part), optionally cached as .npz files."""
    index = audio_index(vocalset)
    tracks: dict[str, PitchTrack] = {}
    with zipfile.ZipFile(vocalset) as z:
        for take in tqdm(takes, desc="pitch"):
            cached = cache / f"{take.name}.npz" if cache else None
            if cached and cached.exists():
                data = np.load(cached)
                tracks[take.name] = PitchTrack(
                    data["midi"], data["prob"], data["db"], data["onset"]
                )
                continue
            if take.name not in index:
                continue
            track = track_pitch(load_take(z, index[take.name], s.sample_rate), s)
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cached,
                    midi=track.midi,
                    prob=track.voiced_prob,
                    db=track.loudness_db,
                    onset=track.onset,
                )
            tracks[take.name] = track
    return tracks


def benchmark(
    takes: list[Take], tracks: dict[str, PitchTrack], s: VoiceSettings
) -> dict[str, dict[str, float]]:
    """Mean scores per kind of take and overall."""
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for take in takes:
        if take.name not in tracks:
            continue
        result = score(take.notes, segment_notes(tracks[take.name], s))
        groups[take.kind].append(result)
        groups["all"].append(result)
    return {
        kind: {
            **{m: float(np.mean([r[m] for r in results])) for m in results[0]},
            "takes": len(results),
        }
        for kind, results in sorted(groups.items())
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Voice mode on VocalSet.")
    parser.add_argument("--vocalset", type=Path, required=True, help="VocalSet11.zip")
    parser.add_argument("--annotations", type=Path, required=True, help="Annotated VocalSet zip")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--cache", type=Path, help="folder for cached pitch tracks")
    parser.add_argument("--output", type=Path, default=Path("benchmark_voice.json"))
    args = parser.parse_args(argv)

    s = VoiceSettings()
    takes = [
        t
        for t in read_labels(args.annotations)
        if (t.singer in DEV_SINGERS) == (args.split == "dev")
    ]
    results = benchmark(takes, pitch_tracks(takes, args.vocalset, s, args.cache), s)

    print(f"{'takes':<12} {'n':>5} {'key F1':>7} {'@100ms':>7} {'F1':>7} {'onset F1':>9}")
    for kind, m in results.items():
        print(
            f"{kind:<12} {m['takes']:>5} {m['key_f1']:>7.3f} {m['key_f1_100ms']:>7.3f}"
            f" {m['f1']:>7.3f} {m['onset_f1']:>9.3f}"
        )
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
