"use client";

import type { Rhythm } from "@music-transcriber/shared-types";
import { rhythmCells } from "@/lib/chords";

/**
 * The strumming pattern of a bar: the count (1 & 2 & ... or 1 e & a ...) and a stroke
 * under every struck eighth or sixteenth. Directions follow the pendulum rule (the hand
 * goes down on the beat and up between), which is a suggestion, not something heard.
 */
export default function RhythmPattern({ rhythm }: { rhythm: Rhythm }) {
  const cells = rhythmCells(rhythm);
  const struck = rhythm.bars.filter((bar) => bar.some(Boolean)).length;
  return (
    <section className="rhythm-pattern row" aria-label="Ritim kalıbı">
      <span className="section-title small">Ritim</span>
      <div className="rhythm-cells" style={{ gridTemplateColumns: `repeat(${cells.length}, 1.6em)` }}>
        {cells.map((cell, i) => (
          <span key={`c${i}`} className={cell.count && !/[e&a]/.test(cell.count) ? "count beat" : "count"}>
            {cell.count}
          </span>
        ))}
        {cells.map((cell, i) => (
          <span key={`s${i}`} className={cell.stroke === "·" ? "stroke rest" : "stroke"}>
            {cell.stroke}
          </span>
        ))}
      </div>
      <span
        className="muted small"
        title="Ölçülerin en az yarısında vurulan 8'likler / 16'lıklar. Yönler önerilen sarkaç hareketi: el vuruşta aşağı, arada yukarı iner."
      >
        {struck} ölçünün çoğunda · ↓ aşağı, ↑ yukarı (önerilen)
      </span>
    </section>
  );
}
