"use client";

const STRING_GAP = 11;
const FRET_GAP = 14;
const FRETS = 5;
const LEFT = 18;
const TOP = 30;

interface ChordDiagramProps {
  /** Per string, lowest first: -1 muted, 0 open; null: no playable shape. */
  frets: number[] | null;
  name: string;
  /** Small text under the name, e.g. the sounding chord with a capo. */
  subtitle?: string;
}

/** A chord box: strings vertical (low E left), frets down, dots where to press. */
export default function ChordDiagram({ frets, name, subtitle }: ChordDiagramProps) {
  const strings = frets?.length ?? 6;
  const width = LEFT + (strings - 1) * STRING_GAP + 10;
  const height = TOP + FRETS * FRET_GAP + (subtitle ? 16 : 6);
  const fretted = (frets ?? []).filter((f) => f > 0);
  const highest = fretted.length ? Math.max(...fretted) : 0;
  const first = highest <= FRETS ? 1 : Math.min(...fretted);
  const x = (s: number) => LEFT + s * STRING_GAP;
  return (
    <svg className="chord-diagram" width={width} height={height} role="img" aria-label={`${name} akor şeması`}>
      <text x={LEFT + ((strings - 1) * STRING_GAP) / 2} y={11} textAnchor="middle" className="chord-name">
        {name}
      </text>
      {Array.from({ length: FRETS + 1 }, (_, i) => (
        <line
          key={`f${i}`}
          x1={x(0)}
          x2={x(strings - 1)}
          y1={TOP + i * FRET_GAP}
          y2={TOP + i * FRET_GAP}
          className={i === 0 && first === 1 ? "nut" : "fret"}
        />
      ))}
      {Array.from({ length: strings }, (_, s) => (
        <line key={`s${s}`} x1={x(s)} x2={x(s)} y1={TOP} y2={TOP + FRETS * FRET_GAP} className="fret" />
      ))}
      {first > 1 && (
        <text x={LEFT - 5} y={TOP + FRET_GAP - 3} textAnchor="end" className="base-fret">
          {first}
        </text>
      )}
      {frets === null ? (
        <text x={LEFT + ((strings - 1) * STRING_GAP) / 2} y={TOP + 2.5 * FRET_GAP} textAnchor="middle" className="base-fret">
          ?
        </text>
      ) : (
        frets.map((fret, s) =>
          fret <= 0 ? (
            <text key={s} x={x(s)} y={TOP - 4} textAnchor="middle" className="open-mark">
              {fret === 0 ? "o" : "×"}
            </text>
          ) : (
            <circle key={s} cx={x(s)} cy={TOP + (fret - first + 0.5) * FRET_GAP} r={4} className="dot" />
          ),
        )
      )}
      {subtitle && (
        <text x={LEFT + ((strings - 1) * STRING_GAP) / 2} y={height - 3} textAnchor="middle" className="chord-sub">
          {subtitle}
        </text>
      )}
    </svg>
  );
}
