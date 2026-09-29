import { describe, expect, it } from "vitest";
import { encodeWav } from "./wav";

describe("encodeWav", () => {
  it("writes a 16-bit mono PCM header and clipped samples", async () => {
    const blob = encodeWav([new Float32Array([0, 0.5]), new Float32Array([-1, 2])], 44100);
    const view = new DataView(await blob.arrayBuffer());
    const text = (offset: number, n: number) =>
      String.fromCharCode(...Array.from({ length: n }, (_, i) => view.getUint8(offset + i)));
    expect(blob.type).toBe("audio/wav");
    expect(blob.size).toBe(44 + 4 * 2);
    expect([text(0, 4), text(8, 4), text(12, 4), text(36, 4)]).toEqual(["RIFF", "WAVE", "fmt ", "data"]);
    expect(view.getUint32(4, true)).toBe(36 + 8);
    expect([view.getUint16(20, true), view.getUint16(22, true), view.getUint32(24, true)]).toEqual([1, 1, 44100]);
    expect(view.getUint16(34, true)).toBe(16);
    expect([0, 1, 2, 3].map((i) => view.getInt16(44 + 2 * i, true))).toEqual([0, 16383, -32768, 32767]);
  });
});
