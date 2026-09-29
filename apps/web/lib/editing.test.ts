import { describe, expect, it } from "vitest";
import type { Note } from "@music-transcriber/shared-types";
import { MIN_DURATION, applyDrag, copyNotes, deleteNotes, notesInRange, pasteNotes, withPitch } from "./editing";
import { STANDARD_TUNING } from "./music";

const note = (pitch: number, start: number, end: number, string: number | null = null, fret: number | null = null): Note => ({
  pitch,
  start,
  end,
  velocity: 80,
  string,
  fret,
  confidence: 0.9,
});

// E3 on the D string (fret 2), G3 on the G string (fret 0), A3 on the G string (fret 2).
const notes = [note(52, 1, 1.5, 2, 2), note(55, 2, 2.5, 3, 0), note(57, 3, 3.5, 3, 2)];

describe("applyDrag", () => {
  it("moves the selected notes in time and pitch, keeping their strings", () => {
    const moved = applyDrag(notes, [0, 2], { kind: "move", dt: 0.5, steps: 2 }, STANDARD_TUNING);
    expect(moved[0]).toMatchObject({ pitch: 54, start: 1.5, end: 2, string: 2, fret: 4, confidence: null });
    expect(moved[2]).toMatchObject({ pitch: 59, start: 3.5, string: 3, fret: 4 });
    expect(moved[1]).toBe(notes[1]); // untouched notes are the same objects
  });

  it("changes string when the pitch leaves the current one", () => {
    // G3 open on the G string, two semitones down: the G string can't play F3.
    const [moved] = applyDrag([notes[1]], [0], { kind: "move", dt: 0, steps: -2 }, STANDARD_TUNING);
    expect(moved.pitch).toBe(53);
    expect(STANDARD_TUNING[moved.string!] + moved.fret!).toBe(53);
  });

  it("stops the group at 0 s", () => {
    const moved = applyDrag(notes, [0, 1], { kind: "move", dt: -5, steps: 0 }, STANDARD_TUNING);
    expect(moved[0].start).toBe(0);
    expect(moved[1].start).toBe(1); // spacing kept
  });

  it("resizes but never below the minimum length", () => {
    const [longer] = applyDrag([notes[0]], [0], { kind: "resize", dt: 0.25, steps: 0 }, STANDARD_TUNING);
    expect(longer.end).toBe(1.75);
    const [shortest] = applyDrag([notes[0]], [0], { kind: "resize", dt: -9, steps: 0 }, STANDARD_TUNING);
    expect(shortest.end).toBeCloseTo(1 + MIN_DURATION);
  });

  it("moves a note to another string at the same pitch, only where it fits", () => {
    const [up] = applyDrag([notes[0]], [0], { kind: "string", dt: 0, steps: -1 }, STANDARD_TUNING);
    expect(up).toMatchObject({ pitch: 52, string: 1, fret: 7 }); // E3 on the A string
    const [same] = applyDrag([notes[0]], [0], { kind: "string", dt: 0, steps: 3 }, STANDARD_TUNING);
    expect(same).toBe(notes[0]); // the B string can't play E3
  });

  it("is a no-op without a selection or movement", () => {
    expect(applyDrag(notes, [], { kind: "move", dt: 1, steps: 1 }, STANDARD_TUNING)).toBe(notes);
    expect(applyDrag(notes, [0], { kind: "move", dt: 0, steps: 0 }, STANDARD_TUNING)).toBe(notes);
  });
});

describe("clipboard", () => {
  it("copies relative to the first note and pastes at a time", () => {
    const clip = copyNotes(notes, [2, 1]);
    expect(clip.map((n) => [n.pitch, n.start])).toEqual([
      [55, 0],
      [57, 1],
    ]);
    const { notes: result, selection } = pasteNotes(notes, clip, 10);
    expect(result).toHaveLength(5);
    expect(selection).toEqual([3, 4]);
    expect(result[3]).toMatchObject({ pitch: 55, start: 10, end: 10.5, confidence: null });
  });

  it("deletes and finds notes in a range", () => {
    expect(deleteNotes(notes, [1]).map((n) => n.pitch)).toEqual([52, 57]);
    expect(notesInRange(notes, 3.2, 0.5, 50, 56)).toEqual([0, 1]);
  });
});

describe("withPitch", () => {
  it("clamps to MIDI and drops positions nothing can play", () => {
    expect(withPitch(notes[0], 200, STANDARD_TUNING).pitch).toBe(127);
    expect(withPitch(notes[0], 30, STANDARD_TUNING)).toMatchObject({ pitch: 30, string: null, fret: null });
  });
});
