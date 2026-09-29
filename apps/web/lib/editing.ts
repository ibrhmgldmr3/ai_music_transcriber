import type { Note } from "@music-transcriber/shared-types";
import { MAX_FRET, clamp, defaultPosition } from "./music";

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
