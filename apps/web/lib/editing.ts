import type { Note } from "@music-transcriber/shared-types";
import { MAX_FRET, beatPosition, beatTime, clamp, defaultPosition } from "./music";

/** Shortest note a resize can leave (seconds). */
export const MIN_DURATION = 0.02;

/** A pointer drag or keyboard nudge applied to the selected notes. */
export interface Drag {
  /** move: time and pitch · resize: the end only · string: time and string (TAB). */
  kind: "move" | "resize" | "string";
  /** Seconds. */
  dt: number;
  /** Semitones for "move", strings for "string" (up = higher string). */
  steps: number;
}

/** Hand edits replace the model's guess, so its confidence no longer applies. */
function edited(note: Note, patch: Partial<Note>): Note {
  return { ...note, ...patch, confidence: null };
}

/** The note at another pitch: on the same string when it can still play it, else moved. */
export function withPitch(note: Note, pitch: number, tuning: number[]): Note {
  const value = clamp(Math.round(pitch), 0, 127);
  if (note.string !== null) {
    const fret = value - tuning[note.string];
    if (fret >= 0 && fret <= MAX_FRET) return edited(note, { pitch: value, fret });
  }
  const position = defaultPosition(value, tuning);
  return edited(note, { pitch: value, string: position?.string ?? null, fret: position?.fret ?? null });
}

/** The same pitch played on another string, if that string can reach it. */
function onString(note: Note, string: number, tuning: number[]): Note {
  const fret = note.pitch - (tuning[string] ?? Infinity);
  return string >= 0 && string < tuning.length && fret >= 0 && fret <= MAX_FRET
    ? edited(note, { string, fret })
    : note;
}

/** Apply a drag to the notes at `selection`; the others are returned as they were. */
export function applyDrag(notes: Note[], selection: readonly number[], drag: Drag, tuning: number[]): Note[] {
  const chosen = new Set(selection.filter((i) => i >= 0 && i < notes.length));
  if (chosen.size === 0 || (drag.dt === 0 && drag.steps === 0)) return notes;
  // The earliest selected note stops at 0 s; the group keeps its spacing.
  const earliest = Math.min(...[...chosen].map((i) => notes[i].start));
  const dt = drag.kind === "resize" ? drag.dt : Math.max(drag.dt, -earliest);
  return notes.map((note, i) => {
    if (!chosen.has(i)) return note;
    if (drag.kind === "resize") {
      return edited(note, { end: Math.max(note.start + MIN_DURATION, note.end + dt) });
    }
    const moved = dt ? edited(note, { start: note.start + dt, end: note.end + dt }) : note;
    if (!drag.steps) return moved;
    if (drag.kind === "string") {
      return moved.string === null ? moved : onString(moved, moved.string + drag.steps, tuning);
    }
    return withPitch(moved, note.pitch + drag.steps, tuning);
  });
}

export function deleteNotes(notes: Note[], selection: readonly number[]): Note[] {
  const chosen = new Set(selection);
  return notes.filter((_, i) => !chosen.has(i));
}

/** Selected notes with times relative to the earliest one, for pasting elsewhere. */
export function copyNotes(notes: Note[], selection: readonly number[]): Note[] {
  const picked = [...new Set(selection)].filter((i) => i >= 0 && i < notes.length).map((i) => notes[i]);
  if (picked.length === 0) return [];
  const origin = Math.min(...picked.map((n) => n.start));
  return picked
    .map((n) => ({ ...n, start: n.start - origin, end: n.end - origin }))
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

/** Append a copied group starting at `at` seconds; returns the new notes and their indices. */
export function pasteNotes(notes: Note[], clip: Note[], at: number): { notes: Note[]; selection: number[] } {
  const start = Math.max(0, at);
  const pasted = clip.map((n) => edited(n, { start: n.start + start, end: n.end + start }));
  return {
    notes: [...notes, ...pasted],
    selection: pasted.map((_, i) => notes.length + i),
  };
}

/** Indices of notes that start inside [t0, t1] with a pitch inside [p0, p1]. */
export function notesInRange(notes: Note[], t0: number, t1: number, p0: number, p1: number): number[] {
  const [a, b] = t0 <= t1 ? [t0, t1] : [t1, t0];
  const [lo, hi] = p0 <= p1 ? [p0, p1] : [p1, p0];
  return notes.flatMap((n, i) => (n.start >= a && n.start <= b && n.pitch >= lo && n.pitch <= hi ? [i] : []));
}

/** The selected notes, or all of them when nothing is selected. */
function targets(notes: Note[], selection: readonly number[]): Set<number> {
  const chosen = selection.filter((i) => i >= 0 && i < notes.length);
  return new Set(chosen.length ? chosen : notes.map((_, i) => i));
}

/**
 * Starts and ends moved to the nearest line of a rhythm grid: `division` lines per beat
 * of `beats` (the analysis' beat times); a note keeps at least one step.
 */
export function quantizeNotes(
  notes: Note[],
  selection: readonly number[],
  beats: number[],
  division: number,
): Note[] {
  if (beats.length < 2 || !(division > 0)) return notes;
  const chosen = targets(notes, selection);
  // Grid steps are 1/division of whichever beat a time falls in, so the grid can follow
  // a tempo that drifts (tracked beats) as well as a steady one.
  const step = (t: number) => Math.round(beatPosition(beats, t) * division);
  const at = (k: number) => Math.max(0, beatTime(beats, k / division));
  const round = (t: number) => Math.round(t * 1e4) / 1e4;
  return notes.map((note, i) => {
    if (!chosen.has(i)) return note;
    const first = step(note.start);
    const start = round(at(first));
    const end = round(at(Math.max(step(note.end), first + 1)));
    return start === note.start && end === note.end ? note : edited(note, { start, end });
  });
}

const MAJOR = [0, 2, 4, 5, 7, 9, 11];
const MINOR = [0, 2, 3, 5, 7, 8, 10, 11]; // natural minor plus the leading tone (harmonic minor)

/** Pitch classes of a key's scale. */
export function scalePitchClasses(key: { tonic: number; mode: "major" | "minor" }): Set<number> {
  return new Set((key.mode === "major" ? MAJOR : MINOR).map((step) => (key.tonic + step) % 12));
}

/**
 * Notes outside the key's scale moved a semitone to the scale note on the side they
 * were sung (`sung`: the pitch actually sung, e.g. from the pitch curve), else towards
 * the previous note. Returns the notes and how many moved.
 */
export function snapToScale(
  notes: Note[],
  selection: readonly number[],
  key: { tonic: number; mode: "major" | "minor" },
  tuning: number[],
  sung?: (note: Note) => number | null,
): { notes: Note[]; moved: number } {
  const scale = scalePitchClasses(key);
  const inScale = (pitch: number) => scale.has(((pitch % 12) + 12) % 12);
  const chosen = targets(notes, selection);
  const byTime = notes.map((_, i) => i).sort((a, b) => notes[a].start - notes[b].start);
  let moved = 0;
  const result = [...notes];
  byTime.forEach((i, order) => {
    const note = notes[i];
    if (!chosen.has(i) || inScale(note.pitch)) return;
    const up = inScale(note.pitch + 1);
    const down = inScale(note.pitch - 1);
    let direction = up && !down ? 1 : -1;
    if (up && down) {
      const heard = sung?.(note) ?? null;
      const previous = order > 0 ? result[byTime[order - 1]].pitch : null;
      if (heard !== null && heard !== note.pitch) direction = heard > note.pitch ? 1 : -1;
      else if (previous !== null && previous !== note.pitch) direction = previous > note.pitch ? 1 : -1;
    }
    result[i] = withPitch(note, note.pitch + direction, tuning);
    moved++;
  });
  return { notes: moved ? result : notes, moved };
}
