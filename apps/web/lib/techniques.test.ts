import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { Note } from "@music-transcriber/shared-types";
import {
  BEND_DELAY,
  BEND_RISE,
  SLIDE_GLIDE,
  VIBRATO_DELAY,
  VIBRATO_DEPTH,
  VIBRATO_RATE,
  synthVoices,
} from "./techniques";

const note = (pitch: number, start: number, end: number, extra: Partial<Note> = {}): Note => ({
  pitch,
  start,
  end,
  velocity: 80,
  string: 4,
  fret: pitch - 59,
  confidence: null,
  ...extra,
});

it("shapes techniques like the MIDI export", () => {
  // Written by the Python implementation: python tests/test_web_fixtures.py
  const fixture = JSON.parse(readFileSync(resolve(__dirname, "../../../tests/fixtures/music_web.json"), "utf-8"));
  expect({
    bend_delay: BEND_DELAY,
    bend_rise: BEND_RISE,
    vibrato_rate: VIBRATO_RATE,
    vibrato_depth_cents: VIBRATO_DEPTH,
    vibrato_delay: VIBRATO_DELAY,
    slide_glide: SLIDE_GLIDE,
  }).toEqual(fixture.technique_shapes);
});

describe("synthVoices", () => {
  it("gives every picked note a voice of its own", () => {
    const voices = synthVoices([note(64, 0, 0.5), note(66, 0.5, 1)]);
    expect(voices.map((v) => [v.pitch, v.start, v.end])).toEqual([
      [64, 0, 0.5],
      [66, 0.5, 1],
    ]);
    expect(voices.every((v) => v.points.length === 0 && v.vibratos.length === 0)).toBe(true);
  });

  it("bends after the pick and holds the bent pitch", () => {
    const [v] = synthVoices([note(64, 0, 1, { bend: 2 })]);
    expect(v.points).toEqual([
      { time: BEND_DELAY, cents: 0, ramp: false },
      { time: BEND_DELAY + BEND_RISE, cents: 200, ramp: true },
    ]);
  });

  it("plays a hammer-on as a step of the same voice, without a new pick", () => {
    const voices = synthVoices([note(64, 0, 0.5), note(66, 0.5, 1, { technique: "hammer_on" })]);
    expect(voices).toHaveLength(1);
    expect(voices[0].end).toBe(1);
    expect(voices[0].points).toEqual([{ time: 0.5, cents: 200, ramp: false }]);
  });

  it("glides into a slide's note, ending where it starts", () => {
    const [v] = synthVoices([note(64, 0, 0.5), note(67, 0.5, 1, { technique: "slide" })]);
    expect(v.points).toEqual([
      { time: 0.5 - SLIDE_GLIDE, cents: 0, ramp: false },
      { time: 0.5, cents: 300, ramp: true },
    ]);
  });

  it("picks again when the technique's note is on another string or long after", () => {
    const other = synthVoices([note(64, 0, 0.5), note(66, 0.5, 1, { technique: "hammer_on", string: 3 })]);
    const late = synthVoices([note(64, 0, 0.5), note(66, 1, 1.5, { technique: "pull_off" })]);
    expect(other).toHaveLength(2);
    expect(late).toHaveLength(2);
  });

  it("adds vibrato after the bend", () => {
    const [v] = synthVoices([note(64, 0, 1, { bend: 1, vibrato: true })]);
    expect(v.vibratos).toHaveLength(1);
    expect(v.vibratos[0].start).toBeGreaterThan(BEND_DELAY + BEND_RISE);
    expect(v.vibratos[0].end).toBe(1);
  });
});
