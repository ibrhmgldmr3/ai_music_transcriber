import { describe, expect, it } from "vitest";
import type { ModelInfo, Project } from "@music-transcriber/shared-types";
import { isOutdated, stageLabel } from "./api";

const models: ModelInfo = {
  version: "guitar_v9@1+guitar_v7@2",
  voice_version: "voice-pyin@1",
  song_version: "song-btc-beatthis@1+voice-pyin@1",
  notes_model: "guitar_v9",
  tab_model: "guitar_v7",
};

const project = (overrides: Partial<Project>): Project => ({
  id: "p",
  name: "p",
  filename: "p.wav",
  status: "completed",
  source: "guitar",
  separate_guitar: false,
  tuning_name: "standard",
  capo: 0,
  capo_auto: false,
  model_version: models.version,
  edited: false,
  progress: null,
  stage: null,
  error: null,
  duration: 1,
  created_at: "",
  updated_at: "",
  ...overrides,
});

describe("isOutdated", () => {
  it("compares guitar projects with the models, voice and song projects with their methods", () => {
    expect(isOutdated(project({}), models)).toBe(false);
    expect(isOutdated(project({ model_version: "guitar_v8@0+guitar_v7@2" }), models)).toBe(true);
    expect(isOutdated(project({ source: "voice", model_version: "voice-pyin@1" }), models)).toBe(false);
    expect(isOutdated(project({ source: "voice", model_version: "voice-pyin@0" }), models)).toBe(true);
    expect(isOutdated(project({ source: "song", model_version: models.song_version }), models)).toBe(false);
    expect(isOutdated(project({ source: "song", model_version: "song-btc-beatthis@1+voice-pyin@0" }), models)).toBe(
      true,
    );
    expect(isOutdated(project({ status: "processing", model_version: null }), models)).toBe(false);
    expect(isOutdated(project({ model_version: null }), null)).toBe(false);
  });
});

describe("stageLabel", () => {
  it("names what the job separates", () => {
    expect(stageLabel(project({ stage: null }))).toBe("Sırada bekliyor");
    expect(stageLabel(project({ stage: "separating" }))).toMatch(/^Gitar/);
    expect(stageLabel(project({ stage: "separating", source: "voice" }))).toMatch(/^Vokal/);
  });
});
