"use client";

import { useMemo, useRef, type MouseEvent } from "react";
import type { ChordSymbol, Note } from "@music-transcriber/shared-types";
import { playablePositions, stringLabels, type BeatGrid } from "@/lib/music";
import { useFollowPlayhead, type LoopRange } from "@/lib/usePlayback";

const SPACING = 22;
const PAD = 16;
const LABEL_WIDTH = 32;
const HEADER = 30; // bar numbers and chord symbols above the strings

interface GuitarTabProps {
  notes: Note[];
  tuning: number[];
  duration: number;
  currentTime: number;
  pixelsPerSecond: number;
  selectedIndex: number | null;
  onSelect: (index: number | null) => void;
  onSeek?: (time: number) => void;
  follow?: boolean;
  grid?: BeatGrid | null;
  chords?: ChordSymbol[];
  /** Chords from before the latest edit, while the new analysis is on its way. */
  chordsStale?: boolean;
  loop?: LoopRange | null;
}

/** Tablature with the highest string on top; one fret number per note at its onset. */
export default function GuitarTab({
  notes,
  tuning,
  duration,
  currentTime,
  pixelsPerSecond,
  selectedIndex,
  onSelect,
  onSeek,
  follow = true,
  grid = null,
  chords = [],
  chordsStale = false,
  loop = null,
}: GuitarTabProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useFollowPlayhead(scrollRef, currentTime, pixelsPerSecond, follow);

  const strings = tuning.length;
  const height = HEADER + PAD * 2 + Math.max(0, strings - 1) * SPACING;
  const width = Math.max(1, Math.ceil(duration * pixelsPerSecond)) + 24;
  const labels = useMemo(() => stringLabels(tuning), [tuning]);
  const lineY = (string: number) => HEADER + PAD + (strings - 1 - string) * SPACING;
  const top = lineY(strings - 1);
  const bottom = lineY(0);

  // Unplaced notes are either outside the guitar's range or simply not assigned yet
  // (the server's optimizer places those on save).
  const { unplayable, unassigned } = useMemo(() => {
    const unplaced = notes.filter((n) => n.string === null || n.fret === null);
    const outOfRange = unplaced.filter((n) => playablePositions(n.pitch, tuning).length === 0).length;
    return { unplayable: outOfRange, unassigned: unplaced.length - outOfRange };
  }, [notes, tuning]);

  const frets = useMemo(
    () =>
      notes.map((n, i) => {
        if (n.string === null || n.fret === null || n.string < 0 || n.string >= strings) return null;
        const x = n.start * pixelsPerSecond;
        const y = HEADER + PAD + (strings - 1 - n.string) * SPACING;
        const label = String(n.fret);
        return (
          <g
            key={i}
            className={i === selectedIndex ? "tab-fret selected" : "tab-fret"}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(i);
            }}
          >
            <rect x={x - 2} y={y - 8} width={label.length * 8 + 4} height={16} rx={3} className="tab-fret-bg" />
            <text x={x} y={y + 4}>
              {label}
            </text>
          </g>
        );
      }),
    [notes, pixelsPerSecond, selectedIndex, onSelect, strings],
  );

  const gridLines = useMemo(() => {
    if (!grid) return null;
    return (
      <g>
        {grid.beats.map((t) => (
          <line key={`b${t}`} x1={t * pixelsPerSecond} x2={t * pixelsPerSecond} y1={top} y2={bottom} className="grid-beat" />
        ))}
        {grid.bars.map(({ time, number }) => (
          <g key={`m${time}`}>
            <line x1={time * pixelsPerSecond} x2={time * pixelsPerSecond} y1={top} y2={bottom} className="grid-bar" />
            {number > 0 && (
              <text x={time * pixelsPerSecond + 3} y={10} className="bar-number">
                {number}
              </text>
            )}
          </g>
        ))}
      </g>
    );
  }, [grid, pixelsPerSecond, top, bottom]);

  const chordLabels = useMemo(
    () => (
      <g className={chordsStale ? "chord-lane stale" : "chord-lane"}>
        {chords.map((c) => (
          <text key={`${c.start}-${c.label}`} x={c.start * pixelsPerSecond + 3} y={24}>
            {c.label}
          </text>
        ))}
      </g>
    ),
    [chords, chordsStale, pixelsPerSecond],
  );

  const onBackgroundClick = (e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek?.((e.clientX - rect.left) / pixelsPerSecond);
    onSelect(null);
  };

  const playheadX = currentTime * pixelsPerSecond;

  return (
    <section className="stack tight">
      <h3 className="section-title">Gitar TAB</h3>
      <div className="timeline">
        <svg className="timeline-labels" width={LABEL_WIDTH} height={height}>
          {labels.map((label, s) => (
            <text key={s} x={LABEL_WIDTH / 2} y={lineY(s) + 4} textAnchor="middle" className="tab-label">
              {label}
            </text>
          ))}
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
            {chordLabels}
            {tuning.map((_, s) => (
              <line key={s} x1={0} x2={width} y1={lineY(s)} y2={lineY(s)} className="tab-line" />
            ))}
            {frets}
            <line x1={playheadX} x2={playheadX} y1={0} y2={height} className="playhead" />
          </svg>
        </div>
      </div>
      {unplayable > 0 && (
        <p className="muted small">{unplayable} nota gitarın çalınabilir aralığı dışında, tab&apos;a yerleştirilemedi.</p>
      )}
      {unassigned > 0 && (
        <p className="muted small">{unassigned} notanın teli/perdesi yok; kaydedince otomatik atanacak.</p>
      )}
    </section>
  );
}
