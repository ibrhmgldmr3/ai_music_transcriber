"use client";

import { useMemo, useRef, useState, type MouseEvent, type PointerEvent } from "react";
import type { ChordSymbol, Note } from "@music-transcriber/shared-types";
import type { Drag } from "@/lib/editing";
import { TECHNIQUE_MARKS, playablePositions, stringLabels, tabLabel, type BeatGrid } from "@/lib/music";
import { startPointerDrag } from "@/lib/pointerDrag";
import { useFollowPlayhead, type LoopRange } from "@/lib/usePlayback";

const SPACING = 22;
const PAD = 16;
const LABEL_WIDTH = 32;
const HEADER = 30; // bar numbers and chord symbols above the strings
const CHAR_WIDTH = 8; // of the monospaced fret labels

/** Where a fret label starts and how wide its background is. */
const labelBox = (x: number, label: string) => ({ x: x - 2, width: label.length * CHAR_WIDTH + 4 });

interface GuitarTabProps {
  notes: Note[];
  /** Open-string pitches the frets count from, capo included. */
  tuning: number[];
  /** Capo fret (0: none); only labels the strings with their names without it. */
  capo?: number;
  duration: number;
  currentTime: number;
  pixelsPerSecond: number;
  selection: ReadonlySet<number>;
  onSelect: (index: number | null, additive: boolean) => void;
  onSelectMany: (indices: number[], additive: boolean) => void;
  /** Drag of the selected notes (time and string); `done` on release. */
  onDrag: (drag: Drag, done: boolean) => void;
  onSeek?: (time: number) => void;
  follow?: boolean;
  grid?: BeatGrid | null;
  chords?: ChordSymbol[];
  /** Chords from before the latest edit, while the new analysis is on its way. */
  chordsStale?: boolean;
  loop?: LoopRange | null;
}

interface Marquee {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Tablature with the highest string on top; one fret number per note at its onset, with
 * its bend ("7b9") and vibrato ("7~"). Hammer-ons and pull-offs arc from the previous
 * note on the string with an "h" / "p"; slides are a line rising or falling to the note.
 * Drag a fret number sideways to move it in time, up or down to play the same pitch on
 * another string; drag on empty space to select an area.
 */
export default function GuitarTab({
  notes,
  tuning,
  capo = 0,
  duration,
  currentTime,
  pixelsPerSecond,
  selection,
  onSelect,
  onSelectMany,
  onDrag,
  onSeek,
  follow = true,
  grid = null,
  chords = [],
  chordsStale = false,
  loop = null,
}: GuitarTabProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const suppressClick = useRef(false);
  const [marquee, setMarquee] = useState<Marquee | null>(null);
  useFollowPlayhead(scrollRef, currentTime, pixelsPerSecond, follow);

  const strings = tuning.length;
  const height = HEADER + PAD * 2 + Math.max(0, strings - 1) * SPACING;
  const width = Math.max(1, Math.ceil(duration * pixelsPerSecond)) + 24;
  const labels = useMemo(() => stringLabels(tuning.map((t) => t - capo)), [tuning, capo]);
  const lineY = (string: number) => HEADER + PAD + (strings - 1 - string) * SPACING;
  const stringAt = (y: number) => strings - 1 - Math.round((y - HEADER - PAD) / SPACING);
  const top = lineY(strings - 1);
  const bottom = lineY(0);

  // Unplaced notes are either outside the guitar's range or simply not assigned yet
  // (the server's optimizer places those on save).
  const { unplayable, unassigned } = useMemo(() => {
    const unplaced = notes.filter((n) => n.string === null || n.fret === null);
    const outOfRange = unplaced.filter((n) => playablePositions(n.pitch, tuning).length === 0).length;
    return { unplayable: outOfRange, unassigned: unplaced.length - outOfRange };
  }, [notes, tuning]);

  const pressFret = (e: PointerEvent, index: number) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    const wasSelected = selection.has(index);
    if (!wasSelected || additive) onSelect(index, additive);
    const toDrag = (dx: number, dy: number): Drag => ({
      kind: "string",
      dt: dx / pixelsPerSecond,
      steps: -Math.round(dy / SPACING), // up the screen = higher string
    });
    startPointerDrag(
      e,
      ({ dx, dy }) => onDrag(toDrag(dx, dy), false),
      (delta) => {
        if (delta) onDrag(toDrag(delta.dx, delta.dy), true);
        else if (wasSelected && !additive) onSelect(index, false);
      },
    );
  };

  const frets = useMemo(
    () =>
      notes.map((n, i) => {
        if (n.string === null || n.fret === null || n.string < 0 || n.string >= strings) return null;
        const x = n.start * pixelsPerSecond;
        const y = HEADER + PAD + (strings - 1 - n.string) * SPACING;
        const label = tabLabel(n);
        const box = labelBox(x, label);
        return (
          <g
            key={i}
            className={selection.has(i) ? "tab-fret selected" : "tab-fret"}
            onPointerDown={(e) => pressFret(e, i)}
            onClick={(e) => e.stopPropagation()}
          >
            <rect x={box.x} y={y - 8} width={box.width} height={16} rx={3} className="tab-fret-bg" />
            <text x={x} y={y + 4}>
              {label}
            </text>
          </g>
        );
      }),
    // pressFret reads only selection, onSelect, onDrag and pixelsPerSecond.
    [notes, pixelsPerSecond, selection, onSelect, onDrag, strings],
  );

  // Each technique joins a note to the previous placed note on its string.
  const techniques = useMemo(() => {
    const order = notes
      .map((n, i) => i)
      .filter((i) => notes[i].string !== null && notes[i].fret !== null)
      .sort((a, b) => notes[a].start - notes[b].start);
    const previous = new Map<number, number>();
    const marks = [];
    for (const i of order) {
      const n = notes[i];
      const string = n.string as number;
      const before = previous.get(string);
      previous.set(string, i);
      if (!n.technique || before === undefined || string < 0 || string >= strings) continue;
      const prev = notes[before];
      const y = HEADER + PAD + (strings - 1 - string) * SPACING;
      const from = labelBox(prev.start * pixelsPerSecond, tabLabel(prev));
      const x0 = from.x + from.width;
      const x1 = n.start * pixelsPerSecond - 2;
      if (x1 <= x0) continue;
      if (n.technique === "slide") {
        const up = n.pitch > prev.pitch;
        marks.push(
          <line key={i} x1={x0 + 1} x2={x1 - 1} y1={y + (up ? 4 : -4)} y2={y + (up ? -4 : 4)} className="tab-technique" />,
        );
      } else {
        const mid = (x0 + x1) / 2;
        const left = from.x + from.width / 2;
        const right = x1 + 2 + labelBox(0, tabLabel(n)).width / 2;
        marks.push(
          <g key={i}>
            <path d={`M ${left} ${y - 9} Q ${mid} ${y - 19} ${right} ${y - 9}`} className="tab-technique" />
            <text x={mid} y={y - 14} textAnchor="middle" className="tab-technique-letter">
              {TECHNIQUE_MARKS[n.technique]}
            </text>
          </g>,
        );
      }
    }
    return <g>{marks}</g>;
  }, [notes, pixelsPerSecond, strings]);

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

  const toLocal = (e: { clientX: number; clientY: number }, svg: SVGSVGElement) => {
    const rect = svg.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onBackgroundDown = (e: PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    const start = toLocal(e, e.currentTarget);
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    startPointerDrag(
      e,
      ({ dx, dy }) => setMarquee({ x0: start.x, y0: start.y, x1: start.x + dx, y1: start.y + dy }),
      (delta) => {
        setMarquee(null);
        if (!delta) return; // a click: seeks (onClick)
        suppressClick.current = true;
        const [t0, t1] = [start.x, start.x + delta.dx].map((x) => x / pixelsPerSecond).sort((a, b) => a - b);
        const [s0, s1] = [stringAt(start.y), stringAt(start.y + delta.dy)].sort((a, b) => a - b);
        onSelectMany(
          notes.flatMap((n, i) =>
            n.string !== null && n.string >= s0 && n.string <= s1 && n.start >= t0 && n.start <= t1 ? [i] : [],
          ),
          additive,
        );
      },
    );
  };

  const onBackgroundClick = (e: MouseEvent<SVGSVGElement>) => {
    if (suppressClick.current) {
      suppressClick.current = false;
      return;
    }
    onSeek?.(toLocal(e, e.currentTarget).x / pixelsPerSecond);
    onSelect(null, false);
  };

  const playheadX = currentTime * pixelsPerSecond;

  return (
    <section className="stack tight">
      <h3 className="section-title">Gitar TAB{capo > 0 && <span className="muted"> · Capo {capo}</span>}</h3>
      <div className="timeline">
        <svg className="timeline-labels" width={LABEL_WIDTH} height={height}>
          {labels.map((label, s) => (
            <text key={s} x={LABEL_WIDTH / 2} y={lineY(s) + 4} textAnchor="middle" className="tab-label">
              {label}
            </text>
          ))}
        </svg>
        <div className="timeline-scroll" ref={scrollRef}>
          <svg width={width} height={height} onPointerDown={onBackgroundDown} onClick={onBackgroundClick}>
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
            {techniques}
            {frets}
            {marquee && (
              <rect
                x={Math.min(marquee.x0, marquee.x1)}
                y={Math.min(marquee.y0, marquee.y1)}
                width={Math.abs(marquee.x1 - marquee.x0)}
                height={Math.abs(marquee.y1 - marquee.y0)}
                className="marquee"
              />
            )}
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
