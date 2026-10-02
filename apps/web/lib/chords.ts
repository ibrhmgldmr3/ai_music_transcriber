import type { Rhythm, SongChord, Voicing } from "@music-transcriber/shared-types";
import { beatPosition, beatTime } from "./music";

/** Roots and chord types the editor offers (Harte labels, as the API stores them). */
export const CHORD_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export const CHORD_QUALITIES: { value: string; label: string }[] = [
  { value: "maj", label: "majör" },
  { value: "min", label: "m" },
  { value: "7", label: "7" },
  { value: "maj7", label: "maj7" },
  { value: "min7", label: "m7" },
  { value: "sus2", label: "sus2" },
  { value: "sus4", label: "sus4" },
  { value: "maj6", label: "6" },
  { value: "min6", label: "m6" },
  { value: "dim", label: "dim" },
  { value: "dim7", label: "dim7" },
  { value: "hdim7", label: "m7b5" },
  { value: "aug", label: "aug" },
  { value: "5", label: "5" },
];

/** Bass notes of slash chords: Harte intervals above the root and their semitones. */
export const BASS_INTERVALS: { value: string; semitones: number }[] = [
  { value: "b2", semitones: 1 },
  { value: "2", semitones: 2 },
  { value: "b3", semitones: 3 },
  { value: "3", semitones: 4 },
  { value: "4", semitones: 5 },
  { value: "b5", semitones: 6 },
  { value: "5", semitones: 7 },
  { value: "b6", semitones: 8 },
  { value: "6", semitones: 9 },
  { value: "b7", semitones: 10 },
  { value: "7", semitones: 11 },
];

/** "A#:min7" -> { root: "A#", quality: "min7", bass: null }; "D:maj/3" has bass "3"
 * (D over its third, F#); a bare root is major. */
export function splitLabel(label: string): { root: string; quality: string; bass: string | null } {
  const [head, bass] = label.split("/");
  const [root, quality] = head.split(":");
  return { root, quality: quality || "maj", bass: bass && bass !== "1" ? bass : null };
}

export const makeLabel = (root: string, quality: string, bass: string | null = null) =>
  bass ? `${root}:${quality}/${bass}` : quality === "maj" ? root : `${root}:${quality}`;

/** The note a bass interval names above a root: ("D", "3") -> "F#". */
export function bassNote(root: string, interval: string): string {
  const semitones = BASS_INTERVALS.find((b) => b.value === interval)?.semitones ?? 0;
  return CHORD_ROOTS[(CHORD_ROOTS.indexOf(root) + semitones) % 12];
}

export interface Bar {
  start: number;
  end: number;
  /** Chords sounding in the bar, in order, with their index in the song's chord list. */
  chords: { index: number; label: string }[];
}

/**
 * The song's bars from its bar lines (``downbeats``) and the chords sounding in each.
 * Without bar lines, a bar is `beatsPerMeasure` beats, or 2 s without beats either.
 */
export function songBars(
  chords: SongChord[],
  downbeats: number[],
  beats: number[] = [],
  beatsPerMeasure = 4,
): Bar[] {
  if (!chords.length) return [];
  const end = Math.max(...chords.map((c) => c.end));
  let lines = downbeats.filter((t) => t < end);
  if (!lines.length) lines = beats.filter((_, i) => i % beatsPerMeasure === 0).filter((t) => t < end);
  if (!lines.length) lines = Array.from({ length: Math.ceil(end / 2) }, (_, i) => i * 2);
  if (chords[0].start < lines[0] - 0.05) lines = [chords[0].start, ...lines]; // a pickup bar
  const bounds = [...lines, Math.max(end, lines[lines.length - 1] + 1e-3)];
  return lines.map((start, i) => {
    const stop = bounds[i + 1];
    const inside = chords
      .map((c, index) => ({ index, label: c.label, start: c.start, end: c.end }))
      .filter((c) => c.start < stop - 1e-3 && c.end > start + 1e-3);
    return { start, end: stop, chords: inside.map(({ index, label }) => ({ index, label })) };
  });
}

export interface Strum {
  time: number;
  /** MIDI pitches from the lowest string up. */
  pitches: number[];
  /** A downstroke (low string first); an upstroke plays them high string first. */
  down: boolean;
}

/**
 * The chords' shapes strummed: with a `rhythm`, the song's pattern in every bar that
 * starts at one of `barTimes` (down or up by the pendulum rule), else a downstroke on
 * every beat. The pattern rather than each bar's own detected strums: those miss strums
 * here and there, which leaves gaps that sound like missing beats. `beats` are the
 * grid's (the analysis'); `tuning` includes the capo, as the shapes' frets count from
 * it. Without beats, strums fall every 0.5 s.
 */
export function strums(
  chords: SongChord[],
  beats: number[],
  voicings: ReadonlyMap<string, Voicing>,
  tuning: number[],
  rhythm: Rhythm | null = null,
  barTimes: number[] = rhythm?.bar_times ?? [],
): Strum[] {
  const shape = (label: string) => {
    const frets = voicings.get(label)?.frets;
    return frets?.flatMap((fret, string) => (fret >= 0 && string < tuning.length ? [tuning[string] + fret] : []));
  };
  const at = (time: number) => chords.find((c) => c.start - 1e-3 <= time && time < c.end - 1e-3);
  const out: Strum[] = [];
  if (rhythm && rhythm.pattern.some(Boolean) && beats.length >= 2) {
    for (const bar of barTimes) {
      const first = Math.round(beatPosition(beats, bar));
      rhythm.pattern.forEach((hit, slot) => {
        if (!hit) return;
        const time = beatTime(beats, first + slot / rhythm.per_beat);
        const chord = at(time);
        const pitches = chord && shape(chord.label);
        if (pitches) out.push({ time, pitches, down: slot % 2 === 0 });
      });
    }
    return out;
  }
  for (const chord of chords) {
    const pitches = shape(chord.label);
    if (!pitches) continue;
    const times = beats.filter((t) => t >= chord.start - 1e-3 && t < chord.end - 1e-3);
    if (!times.length) for (let t = chord.start; t < chord.end - 1e-3; t += 0.5) times.push(t);
    for (const time of times) out.push({ time, pitches, down: true });
  }
  return out.sort((a, b) => a.time - b.time);
}

/** "D-DU-UDU" as arrows under the beat count: ↓ · ↓ ↑ · ↑ ↓ ↑ with 1 & 2 & ... */
export function rhythmCells(rhythm: Rhythm): { count: string; stroke: string }[] {
  const names = rhythm.per_beat === 4 ? ["", "e", "&", "a"] : ["", "&"];
  return rhythm.pattern.map((hit, slot) => ({
    count: slot % rhythm.per_beat === 0 ? String(slot / rhythm.per_beat + 1) : names[slot % rhythm.per_beat],
    stroke: hit ? (slot % 2 === 0 ? "↓" : "↑") : "·",
  }));
}
