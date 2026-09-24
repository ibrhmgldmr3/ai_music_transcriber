"use client";

import type { Note } from "@music-transcriber/shared-types";
import { MAX_FRET, clamp, defaultPosition, midiToName, stringLabels } from "@/lib/music";

interface NoteEditorProps {
  note: Note | null;
  tuning: number[];
  maxFret?: number;
  onChange: (note: Note) => void;
  onDelete: () => void;
}

const round = (value: number) => Math.round(value * 1000) / 1000;

/**
 * Edits the selected note. Pitch and position stay consistent: changing the fret changes
 * the pitch; changing the pitch moves the fret, or the string if the current one can't
 * play it. Choosing "—" as the string lets the optimizer pick a position on save.
 */
export default function NoteEditor({ note: selected, tuning, maxFret = MAX_FRET, onChange, onDelete }: NoteEditorProps) {
  if (!selected) {
    return (
      <aside className="card note-editor">
        <h3>Nota</h3>
        <p className="muted">Düzenlemek için tab veya piyano rulosu üzerinde bir notaya tıklayın.</p>
      </aside>
    );
  }

  const note: Note = selected;
  const labels = stringLabels(tuning);
  const playableOn = (s: number) => {
    const fret = note.pitch - tuning[s];
    return fret >= 0 && fret <= maxFret;
  };
  // Hand edits replace the model's guess, so the confidence no longer applies.
  const edit = (patch: Partial<Note>) => onChange({ ...note, ...patch, confidence: null });

  const setPitch = (value: number) => {
    if (Number.isNaN(value)) return;
    const pitch = clamp(Math.round(value), 0, 127);
    if (note.string !== null) {
      const fret = pitch - tuning[note.string];
      if (fret >= 0 && fret <= maxFret) return edit({ pitch, fret });
    }
    // The current string can't play the new pitch: move it (unplaced if nothing can).
    const position = defaultPosition(pitch, tuning, maxFret);
    edit({ pitch, string: position?.string ?? null, fret: position?.fret ?? null });
  };

  const setString = (value: string) => {
    if (value === "") return edit({ string: null, fret: null });
    const string = Number(value);
    edit({ string, fret: note.pitch - tuning[string] });
  };

  const setFret = (value: number) => {
    if (note.string === null || Number.isNaN(value)) return;
    const fret = clamp(Math.round(value), 0, maxFret);
    edit({ fret, pitch: tuning[note.string] + fret });
  };

  const setStart = (value: number) => {
    if (!Number.isNaN(value)) edit({ start: clamp(value, 0, note.end - 0.01) });
  };

  const setEnd = (value: number) => {
    if (!Number.isNaN(value)) edit({ end: Math.max(value, note.start + 0.01) });
  };

  const setVelocity = (value: number) => {
    if (!Number.isNaN(value)) edit({ velocity: clamp(Math.round(value), 1, 127) });
  };

  return (
    <aside className="card note-editor stack">
      <div className="row between">
        <h3>Nota: {midiToName(note.pitch)}</h3>
        <button className="danger" onClick={onDelete}>
          Sil
        </button>
      </div>

      <div className="grid-2">
        <label className="field">
          Pitch (MIDI)
          <input type="number" min={0} max={127} value={note.pitch} onChange={(e) => setPitch(e.target.valueAsNumber)} />
        </label>
        <label className="field">
          Velocity
          <input type="number" min={1} max={127} value={note.velocity} onChange={(e) => setVelocity(e.target.valueAsNumber)} />
        </label>
        <label className="field">
          Onset (sn)
          <input type="number" min={0} step={0.01} value={round(note.start)} onChange={(e) => setStart(e.target.valueAsNumber)} />
        </label>
        <label className="field">
          Offset (sn)
          <input type="number" min={0} step={0.01} value={round(note.end)} onChange={(e) => setEnd(e.target.valueAsNumber)} />
        </label>
        <label className="field">
          Tel
          <select value={note.string ?? ""} onChange={(e) => setString(e.target.value)}>
            <option value="">—</option>
            {tuning.map((_, s) => (
              <option key={s} value={s} disabled={!playableOn(s)}>
                {tuning.length - s}. tel ({labels[s]})
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Perde
          <input
            type="number"
            min={0}
            max={maxFret}
            value={note.fret ?? ""}
            disabled={note.string === null}
            onChange={(e) => setFret(e.target.valueAsNumber)}
          />
        </label>
      </div>

      <p className="muted small">
        Süre: {(note.end - note.start).toFixed(2)} sn
        {note.confidence !== null && ` · Model güveni: %${Math.round(note.confidence * 100)}`}
      </p>
    </aside>
  );
}
