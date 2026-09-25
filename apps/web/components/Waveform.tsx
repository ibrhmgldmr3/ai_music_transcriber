"use client";

import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { formatTime } from "@/lib/music";
import type { LoopRange } from "@/lib/usePlayback";

const BUCKETS = 2000;
const DRAG_PIXELS = 4; // less movement than this is a click (seek)

interface WaveformProps {
  audioUrl: string;
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
  /** Dragging across the waveform selects a loop. */
  loop?: LoopRange | null;
  onLoopChange?: (range: LoopRange | null) => void;
  height?: number;
}

/** Decode the audio once and reduce it to normalized peak values for drawing. */
async function loadPeaks(url: string, signal: AbortSignal): Promise<Float32Array> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.arrayBuffer();
  const ctx = new AudioContext();
  try {
    const channel = (await ctx.decodeAudioData(data)).getChannelData(0);
    const peaks = new Float32Array(BUCKETS);
    const size = Math.max(1, Math.floor(channel.length / BUCKETS));
    let maxPeak = 0;
    for (let b = 0; b < BUCKETS; b++) {
      let peak = 0;
      const end = Math.min(channel.length, (b + 1) * size);
      for (let i = b * size; i < end; i++) {
        const v = Math.abs(channel[i]);
        if (v > peak) peak = v;
      }
      peaks[b] = peak;
      if (peak > maxPeak) maxPeak = peak;
    }
    if (maxPeak > 0) for (let b = 0; b < BUCKETS; b++) peaks[b] /= maxPeak;
    return peaks;
  } finally {
    void ctx.close();
  }
}

function tickStep(duration: number): number {
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300];
  return steps.find((s) => duration / s <= 12) ?? 600;
}

export default function Waveform({
  audioUrl,
  currentTime,
  duration,
  onSeek,
  loop = null,
  onLoopChange,
  height = 80,
}: WaveformProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; time: number } | null>(null);
  const [selection, setSelection] = useState<LoopRange | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [peaks, setPeaks] = useState<Float32Array | null>(null);
  const [failed, setFailed] = useState(false);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setPeaks(null);
    setFailed(false);
    loadPeaks(audioUrl, controller.signal)
      .then((p) => {
        if (!controller.signal.aborted) setPeaks(p);
      })
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => controller.abort();
  }, [audioUrl]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // Measure now; the observer only reports after the next layout/paint.
    setWidth(Math.floor(el.getBoundingClientRect().width));
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !peaks || width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = getComputedStyle(canvas).getPropertyValue("--accent").trim() || "#6ea8fe";
    const mid = height / 2;
    for (let x = 0; x < width; x++) {
      const b0 = Math.floor((x / width) * peaks.length);
      const b1 = Math.max(b0 + 1, Math.floor(((x + 1) / width) * peaks.length));
      let peak = 0;
      for (let b = b0; b < b1 && b < peaks.length; b++) peak = Math.max(peak, peaks[b]);
      const h = Math.max(1, peak * (height - 4));
      ctx.fillRect(x, mid - h / 2, 1, h);
    }
  }, [peaks, width, height]);

  const ticks = useMemo(() => {
    if (duration <= 0) return [];
    const step = tickStep(duration);
    const result: number[] = [];
    for (let t = 0; t <= duration; t += step) result.push(t);
    return result;
  }, [duration]);

  const timeAt = (e: PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)) * duration;
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (duration <= 0 || e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, time: timeAt(e) };
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || !onLoopChange || Math.abs(e.clientX - drag.x) < DRAG_PIXELS) return;
    const time = timeAt(e);
    setSelection({ start: Math.min(drag.time, time), end: Math.max(drag.time, time) });
  };

  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    if (selection && onLoopChange) onLoopChange(selection);
    else onSeek(drag.time);
    setSelection(null);
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;
  const shaded = selection ?? loop;
  const percent = (time: number) => `${(Math.min(duration, Math.max(0, time)) / duration) * 100}%`;

  return (
    <div className="card waveform-card">
      <div
        ref={wrapRef}
        className="waveform"
        style={{ height }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          dragRef.current = null;
          setSelection(null);
        }}
        title={onLoopChange ? "Tıkla: konuma git · Sürükle: döngü seç" : undefined}
      >
        <canvas ref={canvasRef} style={{ height }} />
        {shaded && duration > 0 && (
          <div
            className={selection ? "loop-region selecting" : "loop-region"}
            style={{ left: percent(shaded.start), width: `calc(${percent(shaded.end)} - ${percent(shaded.start)})` }}
          />
        )}
        {!peaks && <span className="waveform-status muted">{failed ? "Dalga formu yüklenemedi" : "Dalga formu yükleniyor…"}</span>}
        <div className="playhead-bar" style={{ left: `${progress * 100}%` }} />
      </div>
      <div className="time-axis">
        {ticks.map((t) => (
          <span key={t} style={{ left: `${(t / duration) * 100}%` }}>
            {formatTime(t).replace(/\.\d$/, "")}
          </span>
        ))}
      </div>
    </div>
  );
}
