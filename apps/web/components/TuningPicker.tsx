"use client";

import { MAX_CAPO, TUNINGS } from "@/lib/music";

interface TuningPickerProps {
  tuning: string;
  capo: number;
  onChange: (tuning: string, capo: number) => void;
  disabled?: boolean;
}

/** The guitar's tuning and capo: they decide which string and fret play each note. */
export default function TuningPicker({ tuning, capo, onChange, disabled }: TuningPickerProps) {
  return (
    <>
      <label className="row compact" title="Gitarın akordu: notaların hangi tel ve perdede çalındığını belirler">
        Akort
        <select value={tuning} disabled={disabled} onChange={(e) => onChange(e.target.value, capo)}>
          {Object.entries(TUNINGS).map(([name, { label }]) => (
            <option key={name} value={name}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <label className="row compact" title="Capo'lu çalımda perdeler capodan itibaren sayılır">
        Capo
        <select value={capo} disabled={disabled} onChange={(e) => onChange(tuning, Number(e.target.value))}>
          {Array.from({ length: MAX_CAPO + 1 }, (_, fret) => (
            <option key={fret} value={fret}>
              {fret === 0 ? "Yok" : `${fret}. perde`}
            </option>
          ))}
        </select>
      </label>
    </>
  );
}
