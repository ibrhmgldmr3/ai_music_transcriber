import { useEffect, useRef } from "react";
import type { Note } from "@music-transcriber/shared-types";
import type { Strum } from "./chords";
import { VIBRATO_DEPTH, VIBRATO_RATE, synthVoices, type SynthVoice } from "./techniques";

/** Seconds of media time scheduled ahead of the playhead. */
const LOOKAHEAD = 0.2;
const TICK_MS = 25;
const PLUCK_SECONDS = 2.5;
const RELEASE = 0.04; // time constant of the fade after a note ends
const STRUM_SPREAD = 0.012; // seconds between strings in a strum
const STRUM_LEVEL = 0.07;
const UPSTROKE_LEVEL = 0.7; // of a downstroke's
const STRUM_RING = 0.45; // seconds a strummed chord rings

/** Playback gain of a note; the velocity shapes it only a little, like picking harder. */
const noteLevel = (velocity: number) => 0.12 + 0.28 * (velocity / 127);

/**
 * A plucked-string tone (Karplus-Strong): a noise burst circulating in a delay line
 * one period long, low-passed on every pass so the upper harmonics die out first like
 * a real string. A first-order all-pass tunes the fractional part of the period.
 */
function pluck(ctx: BaseAudioContext, midi: number): AudioBuffer {
  const sr = ctx.sampleRate;
  const freq = 440 * 2 ** ((midi - 69) / 12);
  const period = sr / freq;
  const size = Math.max(2, Math.floor(period - 0.6));
  const frac = period - 0.5 - size; // averaging filter adds half a sample of delay
  const c = (1 - frac) / (1 + frac);
  // Low strings ring longer: ~4 s decay (60 dB) on the low E, ~1.5 s two octaves up.
  const t60 = 4 * 0.96 ** Math.max(0, midi - 40);
  const gain = 10 ** (-3 / (t60 * freq));

  const line = new Float32Array(size);
  let mean = 0;
  for (let i = 0; i < size; i++) {
    line[i] = Math.random() * 2 - 1;
    mean += line[i] / size;
  }
  for (let i = 0; i < size; i++) line[i] -= mean; // no DC offset
  for (let i = 1; i < size; i++) line[i] = 0.5 * (line[i] + line[i - 1]); // softer attack

  const length = Math.floor(sr * PLUCK_SECONDS);
  const buffer = ctx.createBuffer(1, length, sr);
  const out = buffer.getChannelData(0);
  let index = 0;
  let x1 = 0;
  let y1 = 0;
  let peak = 0;
  for (let n = 0; n < length; n++) {
    const a = line[index];
    const b = line[(index + 1) % size];
    out[n] = a;
    peak = Math.max(peak, Math.abs(a));
    const filtered = gain * 0.5 * (a + b);
    const y = c * filtered + x1 - c * y1;
    x1 = filtered;
    y1 = y;
    line[index] = y;
    index = (index + 1) % size;
  }
  const fade = Math.floor(sr * 0.05);
  for (let n = 0; n < length; n++) {
    out[n] = (out[n] / (peak || 1)) * Math.min(1, (length - n) / fade);
  }
  return buffer;
}

function click(ctx: BaseAudioContext, accent: boolean): AudioBuffer {
  const sr = ctx.sampleRate;
  const length = Math.floor(sr * 0.05);
  const buffer = ctx.createBuffer(1, length, sr);
  const out = buffer.getChannelData(0);
  const freq = accent ? 1760 : 1320;
  for (let n = 0; n < length; n++) out[n] = Math.sin((2 * Math.PI * freq * n) / sr) * Math.exp(-n / (sr * 0.008));
  return buffer;
}

/**
 * Gives a voice its pitch changes (bends, slides, hammer-ons and pull-offs) on the
 * source's detune and its vibrato as a low-frequency oscillator added to it. `when` is
 * the audio time of the voice's start, `rate` the playback rate. Returns the nodes to
 * disconnect when the voice ends.
 */
function applyPitch(ctx: BaseAudioContext, source: AudioBufferSourceNode, v: SynthVoice, when: number, rate: number) {
  const at = (t: number) => when + Math.max(0, t - v.start) / rate;
  source.detune.setValueAtTime(0, when);
  for (const p of [...v.points].sort((a, b) => a.time - b.time)) {
    if (p.ramp) source.detune.linearRampToValueAtTime(p.cents, at(p.time));
    else source.detune.setValueAtTime(p.cents, at(p.time));
  }
  const nodes: AudioNode[] = [];
  for (const vibrato of v.vibratos) {
    const lfo = ctx.createOscillator();
    lfo.frequency.value = VIBRATO_RATE * rate;
    const depth = ctx.createGain();
    depth.gain.value = VIBRATO_DEPTH;
    lfo.connect(depth).connect(source.detune);
    lfo.start(at(vibrato.start));
    lfo.stop(at(vibrato.end));
    nodes.push(lfo, depth);
  }
  return nodes;
}

interface Voice {
  source: AudioBufferSourceNode;
  gain: GainNode;
}

export interface NotePlayerOptions {
  /** The recording being played: its clock drives the synth. */
  audio: HTMLAudioElement | null;
  playing: boolean;
  notes: Note[];
  /** Play the notes. */
  synth: boolean;
  /** Metronome clicks; bar lines get an accent. */
  clicks: { beats: number[]; bars: number[] } | null;
  /** Chord strums (song projects), quieter than the melody. */
  strums?: Strum[];
}

/**
 * Plays the transcription (and a metronome) in sync with the recording.
 *
 * The recording's media element stays the clock: every tick, events due within the
 * next LOOKAHEAD of media time are scheduled on the Web Audio clock at
 * `now + (event - mediaTime) / playbackRate`. A seek, loop jump or rate change
 * cancels what was scheduled and starts over from the new position.
 */
export function useNotePlayer({ audio, playing, notes, synth, clicks, strums }: NotePlayerOptions) {
  const ctxRef = useRef<AudioContext | null>(null);
  const buffers = useRef(new Map<string, AudioBuffer>());

  useEffect(() => {
    if (!audio || !playing || (!synth && !clicks && !strums?.length)) return;
    const ctx = ctxRef.current ?? new AudioContext();
    ctxRef.current = ctx;
    void ctx.resume();

    const bufferFor = (key: string, make: () => AudioBuffer) => {
      let buffer = buffers.current.get(key);
      if (!buffer) {
        buffer = make();
        buffers.current.set(key, buffer);
      }
      return buffer;
    };
    const active = new Set<Voice>(); // scheduled or sounding
    const voice = (
      buffer: AudioBuffer,
      when: number,
      level: number,
      until?: number,
      shape?: (source: AudioBufferSourceNode) => AudioNode[],
    ) => {
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      const gain = ctx.createGain();
      gain.gain.value = level;
      source.connect(gain).connect(ctx.destination);
      const extra = shape?.(source) ?? [];
      source.start(when);
      if (until !== undefined) {
        gain.gain.setTargetAtTime(0, until, RELEASE);
        source.stop(until + RELEASE * 8);
      }
      const v = { source, gain };
      active.add(v);
      source.onended = () => {
        active.delete(v);
        gain.disconnect();
        for (const node of extra) node.disconnect();
      };
    };

    const events: { time: number; play: (when: number, rate: number) => void }[] = [];
    if (synth) {
      // One voice per pick: hammer-ons, pull-offs and slides change its pitch.
      for (const v of synthVoices(notes)) {
        events.push({
          time: v.start,
          play: (when, rate) =>
            voice(
              bufferFor(`n${v.pitch}`, () => pluck(ctx, v.pitch)),
              when,
              noteLevel(v.velocity),
              when + Math.max(0.05, (v.end - v.start) / rate),
              (source) => applyPitch(ctx, source, v, when, rate),
            ),
        });
      }
    }
    for (const strum of strums ?? []) {
      // An upstroke hits the high strings first, and usually more lightly.
      const order = strum.down ? strum.pitches : [...strum.pitches].reverse();
      const level = strum.down ? STRUM_LEVEL : STRUM_LEVEL * UPSTROKE_LEVEL;
      events.push({
        time: strum.time,
        play: (when, rate) =>
          order.forEach((pitch, i) =>
            voice(
              bufferFor(`n${pitch}`, () => pluck(ctx, pitch)),
              when + (i * STRUM_SPREAD) / rate,
              level,
              when + STRUM_RING / rate,
            ),
          ),
      });
    }
    if (clicks) {
      for (const [times, accent] of [
        [clicks.beats, false],
        [clicks.bars, true],
      ] as const) {
        for (const time of times) {
          events.push({
            time,
            play: (when) => voice(bufferFor(accent ? "accent" : "click", () => click(ctx, accent)), when, 0.35),
          });
        }
      }
    }
    events.sort((a, b) => a.time - b.time);

    let next = 0; // first event not scheduled yet
    let scheduledUntil = -Infinity; // media time up to which events are scheduled
    let lastMedia = audio.currentTime;
    let lastRate = audio.playbackRate;

    const silence = () => {
      const now = ctx.currentTime;
      for (const { source, gain } of active) {
        gain.gain.cancelScheduledValues(now);
        gain.gain.setTargetAtTime(0, now, 0.01);
        try {
          source.stop(now + 0.05);
        } catch {
          // already stopped
        }
      }
    };
    const restart = (media: number) => {
      silence();
      let lo = 0;
      let hi = events.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (events[mid].time < media) lo = mid + 1;
        else hi = mid;
      }
      next = lo;
      scheduledUntil = media;
    };

    restart(audio.currentTime);
    const tick = () => {
      const media = audio.currentTime;
      const rate = audio.playbackRate || 1;
      // A jump backwards (loop, seek) or forwards past what was scheduled, or a new rate.
      if (media < lastMedia - 0.05 || media > scheduledUntil + 0.25 || rate !== lastRate) restart(media);
      lastMedia = media;
      lastRate = rate;

      const horizon = media + LOOKAHEAD * rate;
      const now = ctx.currentTime;
      while (next < events.length && events[next].time < horizon) {
        const event = events[next++];
        event.play(now + Math.max(0, (event.time - media) / rate), rate);
      }
      scheduledUntil = Math.max(scheduledUntil, horizon);
    };
    tick();
    const timer = window.setInterval(tick, TICK_MS);
    return () => {
      window.clearInterval(timer);
      silence();
    };
  }, [audio, playing, synth, clicks, notes, strums]);

  useEffect(
    () => () => {
      void ctxRef.current?.close();
      ctxRef.current = null;
    },
    [],
  );
}

/**
 * The notes played on the synthesized guitar, as audio (e.g. to download a sung melody
 * as guitar). Same sound and levels as playback, peak-normalized.
 */
export async function renderNotes(notes: Note[], sampleRate = 44100): Promise<AudioBuffer> {
  const tail = RELEASE * 8;
  const voices = synthVoices(notes);
  const end = voices.reduce((last, v) => Math.max(last, Math.min(v.start + PLUCK_SECONDS, v.end + tail)), 0);
  const ctx = new OfflineAudioContext(1, Math.max(1, Math.ceil((end + 0.1) * sampleRate)), sampleRate);
  const buffers = new Map<number, AudioBuffer>();
  for (const v of voices) {
    let buffer = buffers.get(v.pitch);
    if (!buffer) {
      buffer = pluck(ctx, v.pitch);
      buffers.set(v.pitch, buffer);
    }
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    const gain = ctx.createGain();
    gain.gain.value = noteLevel(v.velocity);
    source.connect(gain).connect(ctx.destination);
    applyPitch(ctx, source, v, v.start, 1);
    const until = v.start + Math.max(0.05, v.end - v.start);
    source.start(v.start);
    gain.gain.setTargetAtTime(0, until, RELEASE);
    source.stop(until + tail);
  }
  const rendered = await ctx.startRendering();
  const samples = rendered.getChannelData(0);
  let peak = 0;
  for (const x of samples) peak = Math.max(peak, Math.abs(x));
  if (peak > 0) for (let i = 0; i < samples.length; i++) samples[i] *= 0.9 / peak;
  return rendered;
}
