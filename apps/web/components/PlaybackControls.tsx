"use client";

import { formatTime } from "@/lib/music";

const RATES = [0.5, 0.75, 1, 1.25, 1.5];

interface PlaybackControlsProps {
  playing: boolean;
  currentTime: number;
  duration: number;
  rate: number;
  onToggle: () => void;
  onSeek: (time: number) => void;
  onRateChange: (rate: number) => void;
}

export default function PlaybackControls({
  playing,
  currentTime,
  duration,
  rate,
  onToggle,
  onSeek,
  onRateChange,
}: PlaybackControlsProps) {
  const max = duration || 0;
  return (
    <div className="card row playback">
      <button className="primary play-button" onClick={onToggle} aria-label={playing ? "Duraklat" : "Oynat"}>
        {playing ? "❚❚ Duraklat" : "▶ Oynat"}
      </button>
      <button onClick={() => onSeek(0)} aria-label="Başa sar" title="Başa sar">
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
  );
}
