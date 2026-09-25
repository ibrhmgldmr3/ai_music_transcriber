"use client";

import { useMemo, useRef, type MouseEvent, type ReactElement } from "react";
import type { MusicKey, Note } from "@music-transcriber/shared-types";
import { isBlackKey, midiToName, type BeatGrid } from "@/lib/music";
import { useFollowPlayhead, type LoopRange } from "@/lib/usePlayback";

const KEY_WIDTH = 44;

interface PianoRollProps {
  notes: Note[];
  duration: number;
  currentTime: number;
  pixelsPerSecond: number;
  selectedIndex: number | null;
  onSelect: (index: number | null) => void;
  onSeek?: (time: number) => void;
  follow?: boolean;
  rowHeight?: number;
  /** Beat and bar lines; without one, seconds are marked. */
  grid?: BeatGrid | null;
  loop?: LoopRange | null;
  /** Spells note names (Bb rather than A# in F major). */
  musicKey?: MusicKey | null;
}

/** Guitar range by default, widened to fit every note. */
function pitchRange(notes: Note[]): [number, number] {
  let lo = 40;
  let hi = 88;
  for (const n of notes) {
    lo = Math.min(lo, n.pitch);
    hi = Math.max(hi, n.pitch);
  }
  return [Math.max(0, lo - 1), Math.min(127, hi + 1)];
}

export default function PianoRoll({
  notes,
  duration,
  currentTime,
  pixelsPerSecond,
  selectedIndex,
  onSelect,
  onSeek,
  follow = true,
  rowHeight = 9,
  grid = null,
  loop = null,
  musicKey = null,
}: PianoRollProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useFollowPlayhead(scrollRef, currentTime, pixelsPerSecond, follow);

  const [minPitch, maxPitch] = useMemo(() => pitchRange(notes), [notes]);
  const height = (maxPitch - minPitch + 1) * rowHeight;
  const width = Math.max(1, Math.ceil(duration * pixelsPerSecond)) + 24;

  const keys = useMemo(() => {
    const items: ReactElement[] = [];
    for (let p = minPitch; p <= maxPitch; p++) {
      const y = (maxPitch - p) * rowHeight;
      items.push(
        <rect key={`k${p}`} x={0} y={y} width={KEY_WIDTH} height={rowHeight} className={isBlackKey(p) ? "key-black" : "key-white"} />,
      );
      if (p % 12 === 0) {
        items.push(
          <text key={`l${p}`} x={KEY_WIDTH - 4} y={y + rowHeight - 1} className="key-label" textAnchor="end">
            {midiToName(p)}
          </text>,
        );
      }
    }
    return items;
  }, [minPitch, maxPitch, rowHeight]);

  const gridLines = useMemo(() => {
    const items: ReactElement[] = [];
    for (let p = minPitch; p <= maxPitch; p++) {
      const y = (maxPitch - p) * rowHeight;
      if (isBlackKey(p)) items.push(<rect key={`r${p}`} x={0} y={y} width={width} height={rowHeight} className="pr-row-black" />);
      if (p % 12 === 0) items.push(<line key={`o${p}`} x1={0} x2={width} y1={y + rowHeight} y2={y + rowHeight} className="pr-octave" />);
    }
    if (grid) {
      for (const t of grid.beats) {
        const x = t * pixelsPerSecond;
        items.push(<line key={`b${t}`} x1={x} x2={x} y1={0} y2={height} className="pr-second" />);
      }
      for (const { time } of grid.bars) {
        const x = time * pixelsPerSecond;
        items.push(<line key={`m${time}`} x1={x} x2={x} y1={0} y2={height} className="pr-second strong" />);
      }
    } else {
      for (let s = 0; s <= duration; s++) {
        const x = s * pixelsPerSecond;
        items.push(<line key={`t${s}`} x1={x} x2={x} y1={0} y2={height} className={s % 5 === 0 ? "pr-second strong" : "pr-second"} />);
      }
    }
    return items;
  }, [minPitch, maxPitch, rowHeight, width, height, duration, pixelsPerSecond, grid]);

  const noteRects = useMemo(
    () =>
      notes.map((n, i) => (
        <rect
          key={i}
          x={n.start * pixelsPerSecond}
          y={(maxPitch - n.pitch) * rowHeight + 1}
          width={Math.max(2, (n.end - n.start) * pixelsPerSecond)}
          height={rowHeight - 2}
          rx={2}
          className={i === selectedIndex ? "pr-note selected" : "pr-note"}
          style={{ opacity: n.confidence === null ? 1 : 0.45 + 0.55 * n.confidence }}
          onClick={(e) => {
            e.stopPropagation();
            onSelect(i);
          }}
        >
          <title>
            {`${midiToName(n.pitch, musicKey)} · ${n.start.toFixed(2)}–${n.end.toFixed(2)} sn`}
            {n.confidence !== null ? ` · güven %${Math.round(n.confidence * 100)}` : ""}
          </title>
        </rect>
      )),
    [notes, pixelsPerSecond, maxPitch, rowHeight, selectedIndex, onSelect, musicKey],
  );

  const onBackgroundClick = (e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek?.((e.clientX - rect.left) / pixelsPerSecond);
    onSelect(null);
  };

  const playheadX = currentTime * pixelsPerSecond;

  return (
    <section className="stack tight">
      <h3 className="section-title">Piyano rulosu</h3>
      <div className="timeline">
        <svg className="timeline-labels" width={KEY_WIDTH} height={height}>
          {keys}
        </svg>
        <div className="timeline-scroll" ref={scrollRef}>
          <svg width={width} height={height} onClick={onBackgroundClick}>
            {loop && (
              <rect
                x={loop.start * pixelsPerSecond}
                width={(loop.end - loop.start) * pixelsPerSecond}
                y={0}
                height={height}
                className="loop-shade"
              />
            )}
            {gridLines}
            {noteRects}
            <line x1={playheadX} x2={playheadX} y1={0} y2={height} className="playhead" />
          </svg>
        </div>
      </div>
    </section>
  );
}
