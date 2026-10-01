import type { PitchCurve } from "@music-transcriber/shared-types";

/**
 * SVG path of a pitch curve (row centers, fractional pitches), broken where nothing is
 * sung or the pitch jumps by more than a fifth between frames (e.g. an octave error).
 */
export function curvePath(curve: PitchCurve, pixelsPerSecond: number, rowHeight: number, maxPitch: number): string {
  const parts: string[] = [];
  let previous: number | null = null;
  curve.values.forEach((pitch, i) => {
    if (pitch === null) {
      previous = null;
      return;
    }
    const x = ((i / curve.frame_rate) * pixelsPerSecond).toFixed(1);
    const y = ((maxPitch - pitch + 0.5) * rowHeight).toFixed(1);
    parts.push(`${previous === null || Math.abs(pitch - previous) > 7 ? "M" : "L"}${x} ${y}`);
    previous = pitch;
  });
  return parts.join("");
}

/** Median sung pitch between `start` and `end` seconds; null if nothing was sung. */
export function sungPitch(curve: PitchCurve, start: number, end: number): number | null {
  const values = curve.values
    .slice(Math.max(0, Math.floor(start * curve.frame_rate)), Math.ceil(end * curve.frame_rate))
    .filter((v): v is number => v !== null)
    .sort((a, b) => a - b);
  if (!values.length) return null;
  const mid = values.length >> 1;
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}
