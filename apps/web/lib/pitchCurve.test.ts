import { describe, expect, it } from "vitest";
import { curvePath, sungPitch } from "./pitchCurve";

describe("curvePath", () => {
  it("draws row centers and breaks at silence and jumps", () => {
    const curve = { frame_rate: 10, values: [60, 60.5, null, 62, 74.5, 74] };
    // x = t * 100 px/s, y = (maxPitch 64 - pitch + 0.5) * 10 px
    expect(curvePath(curve, 100, 10, 64)).toBe("M0.0 45.0L10.0 40.0M30.0 25.0M40.0 -100.0L50.0 -95.0");
    expect(curvePath({ frame_rate: 50, values: [null, null] }, 100, 10, 64)).toBe("");
  });
});

describe("sungPitch", () => {
  it("is the median of the sung frames in the span", () => {
    const curve = { frame_rate: 10, values: [60, 60.2, null, 60.4, 61, 70] };
    expect(sungPitch(curve, 0, 0.5)).toBeCloseTo(60.3); // 60, 60.2, 60.4, 61
    expect(sungPitch(curve, 0.2, 0.3)).toBeNull();
  });
});
