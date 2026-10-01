import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { MusicKey } from "@music-transcriber/shared-types";
import { KEY_NAMES, TUNINGS, beatPosition, beatTime, gridLines, midiToName, playablePositions, STANDARD_TUNING } from "./music";

// Written by the Python implementation: python tests/test_web_fixtures.py
interface Fixture {
  key_names: string[];
  tunings: Record<string, number[]>;
  midi_range: [number, number];
  spelling: Record<string, { tonic: number | null; mode: "major" | "minor" | null; fifths: number; names: string[] }>;
}
const fixture: Fixture = JSON.parse(
  readFileSync(resolve(__dirname, "../../../tests/fixtures/music_web.json"), "utf-8"),
);

describe("mirrors music_core", () => {
  it("lists the same keys", () => {
    expect(KEY_NAMES).toEqual(fixture.key_names);
  });

  it("offers the same tunings", () => {
    expect(Object.fromEntries(Object.entries(TUNINGS).map(([name, t]) => [name, t.strings]))).toEqual(fixture.tunings);
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

});

describe("grid", () => {
  it("draws the analysis' beats between its numbered bar lines", () => {
    const grid = gridLines({
      beats: [0.25, 0.75, 1.25, 1.75, 2.25],
      bars: [
        { time: 0.25, number: 1 },
        { time: 2.25, number: 2 },
      ],
    });
    expect(grid.beats).toEqual([0.75, 1.25, 1.75]);
    expect(grid.bars.map((b) => b.number)).toEqual([1, 2]);
  });

  it("converts between time and beat position on drifting beats", () => {
    const beats = [1, 1.5, 2.1, 2.8];
    expect(beatPosition(beats, 1.8)).toBeCloseTo(1.5);
    expect(beatTime(beats, 1.5)).toBeCloseTo(1.8);
    // Past the ends at the edge beats' tempo.
    expect(beatPosition(beats, 0.75)).toBeCloseTo(-0.5);
    expect(beatTime(beats, 4)).toBeCloseTo(3.5);
    for (const t of [0.3, 1.2, 2.5, 3.9]) expect(beatTime(beats, beatPosition(beats, t))).toBeCloseTo(t);
    expect(beatPosition([1], 2)).toBeNaN();
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
