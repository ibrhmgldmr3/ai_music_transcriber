import type { Note } from "@music-transcriber/shared-types";

// The shapes of playing techniques in playback, as in the MIDI export: keep them in sync
// with packages/music-core/midi.py.
export const BEND_DELAY = 0.04; // seconds after the pick before a bend starts (at most 1/5 of the note)
export const BEND_RISE = 0.1; // seconds to reach the bent pitch (at most 2/5 of the note)
export const VIBRATO_RATE = 5.5; // Hz
export const VIBRATO_DEPTH = 20; // cents each way
export const VIBRATO_DELAY = 0.1; // seconds after the pick (or the bend) before the vibrato starts
export const SLIDE_GLIDE = 0.06; // seconds a slide takes, ending where the next note starts
/** A hammer-on or pull-off follows this soon after the previous note ends, or is picked. */
export const LEGATO_GAP = 0.1;

/** A pitch change of a voice: `cents` off its first note, set at `time` or ramped to by it. */
export interface PitchPoint {
  time: number;
  cents: number;
  ramp: boolean;
}

/**
 * One picked string: its first note's pitch and level, and the notes reached from it
 * without picking again (hammer-ons, pull-offs, slides) as changes of its pitch.
 */
export interface SynthVoice {
  pitch: number;
  velocity: number;
  start: number;
  end: number;
  points: PitchPoint[];
  vibratos: { start: number; end: number }[];
}

/** The voices that play `notes`: one per pick, with bends, slides and vibrato. */
export function synthVoices(notes: Note[]): SynthVoice[] {
  const order = notes
    .map((n, i) => i)
    .filter((i) => notes[i].end > notes[i].start)
    .sort((a, b) => notes[a].start - notes[b].start || notes[a].pitch - notes[b].pitch);
  const voices: SynthVoice[] = [];
  const onString = new Map<number, { voice: SynthVoice; note: Note }>();
  for (const i of order) {
    const note = notes[i];
    const before = note.string !== null ? onString.get(note.string) : undefined;
    const legato =
      before !== undefined &&
      note.technique &&
      note.start - before.note.end <= LEGATO_GAP &&
      note.start > before.note.start;
    let voice: SynthVoice;
    if (legato && before) {
      voice = before.voice;
      const cents = (note.pitch - voice.pitch) * 100;
      if (note.technique === "slide") {
        const glide = Math.min(SLIDE_GLIDE, (note.start - before.note.start) / 2);
        voice.points.push({ time: note.start - glide, cents: lastCents(voice, note.start - glide), ramp: false });
        voice.points.push({ time: note.start, cents, ramp: true });
      } else {
        voice.points.push({ time: note.start, cents, ramp: false });
      }
      voice.end = note.end;
    } else {
      voice = { pitch: note.pitch, velocity: note.velocity, start: note.start, end: note.end, points: [], vibratos: [] };
      voices.push(voice);
    }
    shape(voice, note);
    if (note.string !== null) onString.set(note.string, { voice, note });
  }
  return voices;
}

/** A note's own bend and vibrato on its voice. */
function shape(voice: SynthVoice, note: Note) {
  const length = note.end - note.start;
  const base = (note.pitch - voice.pitch) * 100;
  const bendStart = note.start + Math.min(BEND_DELAY, length / 5);
  const bendEnd = bendStart + Math.min(BEND_RISE, (2 * length) / 5);
  if (note.bend) {
    voice.points.push({ time: bendStart, cents: base, ramp: false });
    voice.points.push({ time: bendEnd, cents: base + note.bend * 100, ramp: true });
  }
  if (note.vibrato) {
    const start = (note.bend ? bendEnd : note.start) + Math.min(VIBRATO_DELAY, length / 5);
    if (start < note.end) voice.vibratos.push({ start, end: note.end });
  }
}

/** The voice's pitch (cents) at `time` from its points so far. */
function lastCents(voice: SynthVoice, time: number): number {
  let cents = 0;
  let from = { time: voice.start, cents: 0 };
  for (const p of voice.points) {
    if (p.time <= time) {
      cents = p.cents;
      from = p;
    } else {
      if (p.ramp) cents = from.cents + ((p.cents - from.cents) * (time - from.time)) / (p.time - from.time);
      break;
    }
  }
  return cents;
}
