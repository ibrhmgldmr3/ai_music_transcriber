"use client";

import { useMemo, useRef, type MouseEvent } from "react";
import type { Note } from "@music-transcriber/shared-types";
import { playablePositions, stringLabels } from "@/lib/music";
import { useFollowPlayhead } from "@/lib/usePlayback";

const SPACING = 22;
const PAD = 16;
const LABEL_WIDTH = 32;

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
}: GuitarTabProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  useFollowPlayhead(scrollRef, currentTime, pixelsPerSecond, follow);

  const strings = tuning.length;
  const height = PAD * 2 + Math.max(0, strings - 1) * SPACING;
  const width = Math.max(1, Math.ceil(duration * pixelsPerSecond)) + 24;
  const labels = useMemo(() => stringLabels(tuning), [tuning]);
  const lineY = (string: number) => PAD + (strings - 1 - string) * SPACING;

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
        const y = PAD + (strings - 1 - n.string) * SPACING;
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
