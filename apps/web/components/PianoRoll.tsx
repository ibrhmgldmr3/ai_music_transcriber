"use client";

import { useMemo, useRef, useState, type MouseEvent, type PointerEvent, type ReactElement } from "react";
import type { MusicKey, Note, PitchCurve } from "@music-transcriber/shared-types";
import type { Drag } from "@/lib/editing";
import { notesInRange } from "@/lib/editing";
import { isBlackKey, midiToName, type BeatGrid } from "@/lib/music";
import { curvePath } from "@/lib/pitchCurve";
import { startPointerDrag } from "@/lib/pointerDrag";
import { useFollowPlayhead, type LoopRange } from "@/lib/usePlayback";

const KEY_WIDTH = 44;
const HANDLE = 6; // pixels at a note's end that resize it

interface PianoRollProps {
  notes: Note[];
  duration: number;
  currentTime: number;
  pixelsPerSecond: number;
  selection: ReadonlySet<number>;
  /** A note clicked (null: none); `additive` toggles it in the selection. */
  onSelect: (index: number | null, additive: boolean) => void;
  onSelectMany: (indices: number[], additive: boolean) => void;
  /** Drag of the selected notes; `done` on release (one undo step). */
  onDrag: (drag: Drag, done: boolean) => void;
  onSeek?: (time: number) => void;
  follow?: boolean;
  rowHeight?: number;
  /** Beat and bar lines; without one, seconds are marked. */
  grid?: BeatGrid | null;
  loop?: LoopRange | null;
  /** Spells note names (Bb rather than A# in F major). */
  musicKey?: MusicKey | null;
  /** Voice projects: the sung pitch, drawn behind the notes. */
  pitchCurve?: PitchCurve | null;
}

/** Guitar range by default, widened to fit every note. */
function pitchRange(notes: Note[]): [number, number] {
  let lo = 36;
  let hi = 88;
  for (const n of notes) {
    lo = Math.min(lo, n.pitch);
    hi = Math.max(hi, n.pitch);
  }
  return [Math.max(0, lo - 1), Math.min(127, hi + 1)];
}

interface Marquee {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Piano roll. Click a note to select it (Shift/Ctrl: add or remove), drag to move the
 * selection in time and pitch, drag a note's end to resize, drag on empty space to
 * select an area; a plain click on empty space seeks.
 */
export default function PianoRoll({
  notes,
  duration,
  currentTime,
  pixelsPerSecond,
  selection,
  onSelect,
  onSelectMany,
  onDrag,
  onSeek,
  follow = true,
  rowHeight = 9,
  grid = null,
  loop = null,
  musicKey = null,
  pitchCurve = null,
}: PianoRollProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const suppressClick = useRef(false);
  const [marquee, setMarquee] = useState<Marquee | null>(null);
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

  const curve = useMemo(
    () => (pitchCurve ? curvePath(pitchCurve, pixelsPerSecond, rowHeight, maxPitch) : null),
    [pitchCurve, pixelsPerSecond, rowHeight, maxPitch],
  );

  const pressNote = (e: PointerEvent, index: number, kind: Drag["kind"]) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    const wasSelected = selection.has(index);
    if (!wasSelected || additive) onSelect(index, additive);
    const toDrag = (dx: number, dy: number): Drag => ({
      kind,
      dt: dx / pixelsPerSecond,
      steps: kind === "move" ? -Math.round(dy / rowHeight) : 0,
    });
    startPointerDrag(
      e,
      ({ dx, dy }) => onDrag(toDrag(dx, dy), false),
      (delta) => {
        if (delta) onDrag(toDrag(delta.dx, delta.dy), true);
        else if (wasSelected && !additive) onSelect(index, false); // click in a group: just this one
      },
    );
  };

  const noteRects = useMemo(
    () =>
      notes.map((n, i) => {
        const x = n.start * pixelsPerSecond;
        const w = Math.max(2, (n.end - n.start) * pixelsPerSecond);
        const y = (maxPitch - n.pitch) * rowHeight + 1;
        return (
          <g key={i}>
            <rect
              x={x}
              y={y}
              width={w}
              height={rowHeight - 2}
              rx={2}
              className={selection.has(i) ? "pr-note selected" : "pr-note"}
              style={{ opacity: n.confidence === null ? 1 : 0.45 + 0.55 * n.confidence }}
              onPointerDown={(e) => pressNote(e, i, "move")}
              onClick={(e) => e.stopPropagation()}
            >
              <title>
                {`${midiToName(n.pitch, musicKey)} · ${n.start.toFixed(2)}–${n.end.toFixed(2)} sn`}
                {n.confidence !== null ? ` · güven %${Math.round(n.confidence * 100)}` : ""}
              </title>
            </rect>
            {w > 3 * HANDLE && (
              <rect
                x={x + w - HANDLE}
                y={y}
                width={HANDLE}
                height={rowHeight - 2}
                className="pr-note-handle"
                onPointerDown={(e) => pressNote(e, i, "resize")}
                onClick={(e) => e.stopPropagation()}
              />
            )}
          </g>
        );
      }),
    // pressNote reads only selection, onSelect, onDrag, pixelsPerSecond and rowHeight.
    [notes, pixelsPerSecond, maxPitch, rowHeight, selection, onSelect, onDrag, musicKey],
  );

  const toLocal = (e: { clientX: number; clientY: number }, svg: SVGSVGElement) => {
    const rect = svg.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onBackgroundDown = (e: PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    const svg = e.currentTarget;
    const start = toLocal(e, svg);
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    startPointerDrag(
      e,
      ({ dx, dy }) => setMarquee({ x0: start.x, y0: start.y, x1: start.x + dx, y1: start.y + dy }),
      (delta) => {
        setMarquee(null);
        if (!delta) return; // a click: seeks (onClick)
        suppressClick.current = true;
        const pitchAt = (y: number) => maxPitch - Math.floor(y / rowHeight);
        onSelectMany(
          notesInRange(
            notes,
            start.x / pixelsPerSecond,
            (start.x + delta.dx) / pixelsPerSecond,
            pitchAt(start.y),
            pitchAt(start.y + delta.dy),
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
      <div className="row between">
        <h3 className="section-title">Piyano rulosu</h3>
        {pitchCurve && (
          <span className="muted small">
            <span className="pr-curve-key" /> söylediğiniz perde
          </span>
        )}
      </div>
      <div className="timeline">
        <svg className="timeline-labels" width={KEY_WIDTH} height={height}>
          {keys}
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
            {noteRects}
            {curve && <path d={curve} className="pr-curve" />}
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
    </section>
  );
}
