"use client";

import { formatTime } from "@/lib/music";
import type { LoopRange } from "@/lib/usePlayback";

const RATES = [0.5, 0.75, 1, 1.25, 1.5];

/** What is heard: the recording, the transcription played by a synth, or both. */
export type ListenMode = "audio" | "notes" | "both";

const LISTEN_LABELS: Record<ListenMode, string> = {
  audio: "Kayıt",
  notes: "Notalar",
  both: "Kayıt + notalar",
};

interface PlaybackControlsProps {
  playing: boolean;
  currentTime: number;
  duration: number;
  rate: number;
  onToggle: () => void;
  onSeek: (time: number) => void;
  onRateChange: (rate: number) => void;
  listen: ListenMode;
  onListenChange: (mode: ListenMode) => void;
  metronome: boolean;
  onMetronomeChange: (on: boolean) => void;
  loop: LoopRange | null;
  onLoopStart: () => void;
  onLoopEnd: () => void;
  onLoopClear: () => void;
}

export default function PlaybackControls({
  playing,
  currentTime,
  duration,
  rate,
  onToggle,
  onSeek,
  onRateChange,
  listen,
  onListenChange,
  metronome,
  onMetronomeChange,
  loop,
  onLoopStart,
  onLoopEnd,
  onLoopClear,
}: PlaybackControlsProps) {
  const max = duration || 0;
  return (
    <div className="card stack tight">
      <div className="row playback">
        <button className="primary play-button" onClick={onToggle} aria-label={playing ? "Duraklat" : "Oynat"}>
          {playing ? "❚❚ Duraklat" : "▶ Oynat"}
        </button>
        <button onClick={() => onSeek(loop?.start ?? 0)} aria-label="Başa sar" title="Başa sar">
          ⏮
        </button>
        <span className="time">
          {formatTime(currentTime)} / {formatTime(max)}
        </span>
        <input
          className="seek"
          type="range"
          min={0}
          max={max}
          step={0.01}
          value={Math.min(currentTime, max)}
          onChange={(e) => onSeek(Number(e.target.value))}
          aria-label="Konum"
        />
        <label className="row compact">
          Hız
          <select value={rate} onChange={(e) => onRateChange(Number(e.target.value))}>
            {RATES.map((r) => (
              <option key={r} value={r}>
                {r}×
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row playback-options">
        <label className="row compact" title="Transkripsiyonu sentezlenmiş gitar sesiyle çal">
          Dinle
          <select value={listen} onChange={(e) => onListenChange(e.target.value as ListenMode)}>
            {(Object.keys(LISTEN_LABELS) as ListenMode[]).map((mode) => (
              <option key={mode} value={mode}>
                {LISTEN_LABELS[mode]}
              </option>
            ))}
          </select>
        </label>
        <label className="row compact" title="Ölçü başları vurgulu tık (M tuşu)">
          <input type="checkbox" checked={metronome} onChange={(e) => onMetronomeChange(e.target.checked)} />
          Metronom
        </label>
        <div className="row compact loop-controls" role="group" aria-label="Döngü">
          <span className="muted">Döngü</span>
          <button onClick={onLoopStart} title="Döngü başı: şu anki konum (A tuşu)">
            A
          </button>
          <button onClick={onLoopEnd} title="Döngü sonu: şu anki konum (B tuşu)">
            B
          </button>
          {loop ? (
            <>
              <span className="time">
                {formatTime(loop.start)} – {formatTime(loop.end)}
              </span>
              <button onClick={onLoopClear} aria-label="Döngüyü kaldır" title="Döngüyü kaldır (L tuşu)">
                ✕
              </button>
            </>
          ) : (
            <span className="muted small">yok · dalga formunda sürükleyerek de seçebilirsiniz</span>
          )}
        </div>
      </div>
    </div>
  );
}
