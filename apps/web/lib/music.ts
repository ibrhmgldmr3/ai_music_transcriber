export const STANDARD_TUNING = [40, 45, 50, 55, 59, 64];
export const MAX_FRET = 24;

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

const pitchClass = (midi: number) => ((midi % 12) + 12) % 12;

export function midiToName(midi: number): string {
  return `${NOTE_NAMES[pitchClass(midi)]}${Math.floor(midi / 12) - 1}`;
}

export function isBlackKey(midi: number): boolean {
  return BLACK_KEYS.has(pitchClass(midi));
}

/** Pitch-class names per string, lowest string first; the highest is lower-cased ("e"). */
export function stringLabels(tuning: number[]): string[] {
  const labels = tuning.map((p) => NOTE_NAMES[pitchClass(p)]);
  if (labels.length > 0) labels[labels.length - 1] = labels[labels.length - 1].toLowerCase();
  return labels;
}

export interface Position {
  string: number;
  fret: number;
}

// Mirrors TabCostWeights.preferred_fret in packages/music-core/tab.py.
const PREFERRED_FRET = 5;

/** Every (string, fret) that produces `pitch`. */
export function playablePositions(pitch: number, tuning: number[], maxFret = MAX_FRET): Position[] {
  return tuning
    .map((open, string) => ({ string, fret: pitch - open }))
    .filter((p) => p.fret >= 0 && p.fret <= maxFret);
}

/** A sensible single-note position: closest to the hand's preferred area. */
export function defaultPosition(pitch: number, tuning: number[], maxFret = MAX_FRET): Position | null {
  const options = playablePositions(pitch, tuning, maxFret);
  if (options.length === 0) return null;
  return options.reduce((best, p) =>
    Math.abs(p.fret - PREFERRED_FRET) < Math.abs(best.fret - PREFERRED_FRET) ? p : best,
  );
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.0";
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}
