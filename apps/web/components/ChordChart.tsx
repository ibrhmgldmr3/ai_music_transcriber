"use client";

import { useMemo, useState } from "react";
import type { KeySpan, SongChord, Voicing } from "@music-transcriber/shared-types";
import ChordDiagram from "@/components/ChordDiagram";
import {
  type Bar,
  BASS_INTERVALS,
  CHORD_QUALITIES,
  CHORD_ROOTS,
  bassNote,
  makeLabel,
  splitLabel,
} from "@/lib/chords";

const BARS_PER_ROW = 4;

interface ChordChartProps {
  chords: SongChord[];
  bars: Bar[];
  /** A modulating song's keys; each change is marked on the bar it starts in. */
  keys: KeySpan[];
  /** Shapes and names by chord label (from the API). */
  voicings: ReadonlyMap<string, Voicing>;
  capo: number;
  currentTime: number;
  onSeek: (time: number) => void;
  /** A chord relabeled by the user. */
  onRelabel: (index: number, label: string) => void;
}

/**
 * The song's chords bar by bar, as the shapes to play (from the capo), with a chord box
 * for each shape. Click a bar to play from it, a chord to change it (root, type and a
 * bass note for slash chords such as D/F#).
 */
export default function ChordChart({
  chords,
  bars,
  keys,
  voicings,
  capo,
  currentTime,
  onSeek,
  onRelabel,
}: ChordChartProps) {
  const [editing, setEditing] = useState<number | null>(null);
  const shape = (label: string) => voicings.get(label)?.shape ?? label;
  const unique = useMemo(() => [...new Set(chords.map((c) => c.label))], [chords]);
  const current = bars.findIndex((b) => currentTime >= b.start && currentTime < b.end);
  // The bar each key change after the first starts in.
  const keyChanges = useMemo(() => {
    const marks = new Map<number, string>();
    for (const span of keys.slice(1)) {
      const bar = bars.findIndex((b) => span.start >= b.start - 0.05 && span.start < b.end - 0.05);
      if (bar >= 0) marks.set(bar, span.key);
    }
    return marks;
  }, [keys, bars]);
  const edited = editing !== null ? chords[editing] : null;
  const parts = edited ? splitLabel(edited.label) : null;

  return (
    <section className="stack tight">
      <div className="row between">
        <h3 className="section-title">Akorlar{keys.length > 1 ? ` · ${keys[0].key} ile başlar` : ""}</h3>
        <span className="muted small">
          {capo ? `Capo ${capo}: çalınacak şekiller; duyulan akor altında` : "Capo yok"} · Bara tıkla: oradan çal ·
          Akora tıkla: değiştir
        </span>
      </div>
      <div className="chord-shapes">
        {unique.map((label) => {
          const v = voicings.get(label);
          return (
            <ChordDiagram
              key={label}
              frets={v?.frets ?? null}
              name={v?.shape ?? label}
              subtitle={capo && v ? v.name : undefined}
            />
          );
        })}
      </div>
      <div className="chord-bars" style={{ gridTemplateColumns: `repeat(${BARS_PER_ROW}, minmax(0, 1fr))` }}>
        {bars.map((bar, i) => (
          <div
            key={bar.start}
            className={i === current ? "chord-bar current" : "chord-bar"}
            onClick={() => onSeek(bar.start)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && onSeek(bar.start)}
          >
            <span className="bar-number">{i + 1}</span>
            {keyChanges.has(i) && (
              <span className="key-change" title="Ton burada değişiyor">
                {keyChanges.get(i)}
              </span>
            )}
            {bar.chords.map((c) => (
              <button
                key={c.index}
                className={c.index === editing ? "chord-chip selected" : "chord-chip"}
                title={voicings.get(c.label)?.name}
                onClick={(e) => {
                  e.stopPropagation();
                  setEditing(c.index === editing ? null : c.index);
                }}
              >
                {shape(c.label)}
              </button>
            ))}
          </div>
        ))}
      </div>
      {edited && parts && editing !== null && (
        <div className="card row compact chord-editor">
          <span>
            {edited.start.toFixed(2)}–{edited.end.toFixed(2)} sn, duyulan akor:
          </span>
          <select
            value={parts.root}
            onChange={(e) => onRelabel(editing, makeLabel(e.target.value, parts.quality, parts.bass))}
            aria-label="Kök"
          >
            {CHORD_ROOTS.map((root) => (
              <option key={root} value={root}>
                {root}
              </option>
            ))}
          </select>
          <select
            value={parts.quality}
            onChange={(e) => onRelabel(editing, makeLabel(parts.root, e.target.value, parts.bass))}
            aria-label="Akor türü"
          >
            {CHORD_QUALITIES.map((q) => (
              <option key={q.value} value={q.value}>
                {q.label}
              </option>
            ))}
          </select>
          <label className="row compact" title="Basta kökten başka bir nota çalıyorsa (D/F# gibi)">
            Bas:
            <select
              value={parts.bass ?? ""}
              onChange={(e) => onRelabel(editing, makeLabel(parts.root, parts.quality, e.target.value || null))}
              aria-label="Bas notası"
            >
              <option value="">Kök ({parts.root})</option>
              {BASS_INTERVALS.map((b) => (
                <option key={b.value} value={b.value}>
                  {bassNote(parts.root, b.value)}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => setEditing(null)}>Kapat</button>
        </div>
      )}
    </section>
  );
}
