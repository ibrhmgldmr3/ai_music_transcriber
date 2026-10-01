import { describe, expect, it } from "vitest";
import type { Voicing } from "@music-transcriber/shared-types";
import { bassNote, makeLabel, rhythmCells, songBars, splitLabel, strums } from "./chords";

const chords = [
  { start: 0.5, end: 4.0, label: "C:min" },
  { start: 4.0, end: 5.0, label: "A#" },
  { start: 5.0, end: 6.0, label: "F" },
];

describe("songBars", () => {
  it("groups the chords by bar, with a pickup bar before the first bar line", () => {
    const bars = songBars(chords, [2.0, 4.0, 6.0]);
    expect(bars.map((b) => [b.start, b.end])).toEqual([
      [0.5, 2],
      [2, 4],
      [4, 6],
    ]);
    expect(bars.map((b) => b.chords.map((c) => c.label))).toEqual([["C:min"], ["C:min"], ["A#", "F"]]);
    expect(bars[2].chords[1].index).toBe(2);
  });

  it("falls back to beats, then to 2 s bars", () => {
    expect(songBars(chords, [], [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], 4).map((b) => b.start)).toEqual([0.5, 2.5]);
    expect(songBars(chords, []).map((b) => b.start)).toEqual([0, 2, 4]);
    expect(songBars([], [0, 2])).toEqual([]);
  });
});

describe("strums", () => {
  it("strums each chord's shape on its beats, from the capo", () => {
    const am: Voicing = { label: "C:min", name: "Cm", shape: "Am", frets: [-1, 0, 2, 2, 1, 0], difficulty: 1.2 };
    const tuning = [43, 48, 53, 58, 62, 67]; // standard with capo 3
    const out = strums(chords, [0.5, 1.0, 4.0], new Map([["C:min", am]]), tuning);
    expect(out.map((s) => s.time)).toEqual([0.5, 1.0]); // A# and F have no shape here
    expect(out[0].pitches).toEqual([48, 55, 60, 63, 67]); // Cm sounding: C G C Eb G
  });

  it("follows a strumming pattern: struck slots, down on the beat and up between", () => {
    const e: Voicing = { label: "E", name: "E", shape: "E", frets: [0, 2, 2, 1, 0, 0], difficulty: 1 };
    const beats = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    const rhythm = {
      per_beat: 2,
      pattern: [true, false, true, true, false, true, true, true],
      text: "D-DU-UDU",
      bars: [[true, false, true, true, false, true, true, true]],
      bar_times: [0],
    };
    const out = strums([{ start: 0, end: 4, label: "E" }], beats, new Map([["E", e]]), [40, 45, 50, 55, 59, 64], rhythm);
    expect(out.map((s) => [s.time, s.down])).toEqual([
      [0, true],
      [0.5, true],
      [0.75, false],
      [1.25, false],
      [1.5, true],
      [1.75, false],
    ]);
    expect(rhythmCells(rhythm).map((c) => c.count + c.stroke).join(" ")).toBe("1↓ &· 2↓ &↑ 3· &↑ 4↓ &↑");
  });

  it("strums every half second without beats", () => {
    const e: Voicing = { label: "E", name: "E", shape: "E", frets: [0, 2, 2, 1, 0, 0], difficulty: 1 };
    expect(strums([{ start: 0, end: 1.2, label: "E" }], [], new Map([["E", e]]), [40, 45, 50, 55, 59, 64]).length).toBe(3);
  });
});

describe("labels", () => {
  it("splits and builds Harte labels", () => {
    expect(splitLabel("A#:min7")).toEqual({ root: "A#", quality: "min7", bass: null });
    expect(splitLabel("F")).toEqual({ root: "F", quality: "maj", bass: null });
    expect(makeLabel("F", "maj")).toBe("F");
    expect(makeLabel("F", "sus4")).toBe("F:sus4");
  });

  it("keeps the bass of slash chords", () => {
    expect(splitLabel("D:maj/3")).toEqual({ root: "D", quality: "maj", bass: "3" });
    expect(splitLabel("C:maj/1").bass).toBeNull();
    expect(makeLabel("D", "maj", "3")).toBe("D:maj/3");
    expect(makeLabel("A", "min", "b7")).toBe("A:min/b7");
    expect(bassNote("D", "3")).toBe("F#");
    expect(bassNote("A", "b7")).toBe("G");
  });
});
