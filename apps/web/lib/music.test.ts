import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { MusicKey } from "@music-transcriber/shared-types";
import { KEY_NAMES, beatGrid, midiToName, playablePositions, STANDARD_TUNING } from "./music";

// Written by the Python implementation: python tests/test_web_fixtures.py
interface Fixture {
  key_names: string[];
  midi_range: [number, number];
  spelling: Record<string, { tonic: number | null; mode: "major" | "minor" | null; fifths: number; names: string[] }>;
  first_bars: { tempo: number; beats_per_measure: number; downbeat: number; first_note: number; bar1_time: number }[];
}
const fixture: Fixture = JSON.parse(
  readFileSync(resolve(__dirname, "../../../tests/fixtures/music_web.json"), "utf-8"),
);

describe("mirrors music_core", () => {
  it("lists the same keys", () => {
    expect(KEY_NAMES).toEqual(fixture.key_names);
  });

  it("spells every pitch in every key like the MusicXML export", () => {
    const [low] = fixture.midi_range;
    for (const [name, entry] of Object.entries(fixture.spelling)) {
      const key: MusicKey | null =
        entry.mode === null ? null : { name, tonic: entry.tonic ?? 0, mode: entry.mode, fifths: entry.fifths };
      const names = entry.names.map((_, i) => midiToName(low + i, key));
      expect(names, name).toEqual(entry.names);
    }
  });

  it("numbers bars like the MusicXML measures", () => {
    for (const c of fixture.first_bars) {
      const bar = (60 / c.tempo) * c.beats_per_measure;
      const grid = beatGrid(c.tempo, c.beats_per_measure, c.downbeat, c.first_note, c.first_note + 30);
      // Bar 1 can start before the recording (a pickup), and the grid only has lines from
      // 0 s on; then bar 2 is checked. Lines before bar 1 are numbered 0, -1, ...
      const number = c.bar1_time < -1e-9 ? 2 : 1;
      const line = grid.bars.find((b) => b.number === number);
      expect(line?.time).toBeCloseTo(c.bar1_time + (number - 1) * bar, 6);
      expect(grid.bars.every((b) => b.number < 1 || b.time >= c.bar1_time - 1e-9)).toBe(true);
    }
  });
});

describe("beatGrid", () => {
  it("puts bar lines every bar and beats in between", () => {
    const grid = beatGrid(120, 4, 0.25, 0.25, 4.3);
    expect(grid.bars.map((b) => b.time)).toEqual([0.25, 2.25, 4.25]);
    expect(grid.bars.map((b) => b.number)).toEqual([1, 2, 3]);
    expect(grid.beats).toEqual([0.75, 1.25, 1.75, 2.75, 3.25, 3.75]);
  });

  it("is empty for an unusable tempo", () => {
    expect(beatGrid(0, 4, 0, 0, 10)).toEqual({ beats: [], bars: [] });
  });
});

describe("playablePositions", () => {
  it("finds every string/fret for a pitch", () => {
    expect(playablePositions(64, STANDARD_TUNING)).toEqual([
      { string: 0, fret: 24 },
      { string: 1, fret: 19 },
      { string: 2, fret: 14 },
      { string: 3, fret: 9 },
      { string: 4, fret: 5 },
      { string: 5, fret: 0 },
    ]);
    expect(playablePositions(39, STANDARD_TUNING)).toEqual([]);
  });
});
