"use client";

import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { formatTime } from "@/lib/music";

const BUCKETS = 2000;

interface WaveformProps {
  audioUrl: string;
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
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

export default function Waveform({ audioUrl, currentTime, duration, onSeek, height = 80 }: WaveformProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
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

  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    if (duration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - rect.left) / rect.width) * duration);
  };

  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;

  return (
    <div className="card waveform-card">
      <div ref={wrapRef} className="waveform" style={{ height }} onClick={onClick}>
        <canvas ref={canvasRef} style={{ height }} />
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
