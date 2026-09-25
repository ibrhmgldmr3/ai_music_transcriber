import { useCallback, useEffect, useRef, useState } from "react";

export interface LoopRange {
  start: number;
  end: number;
}

/** Shortest loop worth repeating (seconds). */
export const MIN_LOOP = 0.1;

/** Audio playback state backed by an HTMLAudioElement, with an optional A-B loop. */
export function usePlayback(src: string | null) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRateState] = useState(1);
  const [muted, setMutedState] = useState(false);
  const [loop, setLoopState] = useState<LoopRange | null>(null);
  const loopRef = useRef<LoopRange | null>(null);
  const outputRef = useRef({ rate: 1, muted: false }); // carried over to a new element

  useEffect(() => {
    if (!src) return;
    const element = new Audio(src);
    element.preload = "auto";
    element.playbackRate = outputRef.current.rate;
    element.muted = outputRef.current.muted;
    audioRef.current = element;
    setAudio(element);

    // Past the loop's end: jump back to its start.
    const wrap = () => {
      const range = loopRef.current;
      if (range && !element.paused && element.currentTime >= range.end) element.currentTime = range.start;
    };
    const onMetadata = () => setDuration(Number.isFinite(element.duration) ? element.duration : 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnded = () => {
      const range = loopRef.current; // a loop reaching the end of the file
      if (range) {
        element.currentTime = range.start;
        element.play().catch(() => setPlaying(false));
      } else setPlaying(false);
    };
    const onTime = () => {
      wrap();
      setCurrentTime(element.currentTime);
    };
    element.addEventListener("loadedmetadata", onMetadata);
    element.addEventListener("play", onPlay);
    element.addEventListener("pause", onPause);
    element.addEventListener("ended", onEnded);
    // ~4 Hz and independent of rendering, so the position stays right even when
    // animation frames are throttled (background tab, hidden window).
    element.addEventListener("timeupdate", onTime);
    element.addEventListener("seeked", onTime);

    // Poll the position every frame for a smooth playhead and a tight loop; React skips
    // renders while the value is unchanged.
    let frame = 0;
    const tick = () => {
      wrap();
      setCurrentTime(element.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frame);
      element.pause();
      element.removeEventListener("loadedmetadata", onMetadata);
      element.removeEventListener("play", onPlay);
      element.removeEventListener("pause", onPause);
      element.removeEventListener("ended", onEnded);
      element.removeEventListener("timeupdate", onTime);
      element.removeEventListener("seeked", onTime);
      audioRef.current = null;
      setAudio(null);
      setPlaying(false);
      setCurrentTime(0);
      setDuration(0);
    };
  }, [src]);

  const start = useCallback((element: HTMLAudioElement) => {
    // With a loop set, playback starts inside it.
    const range = loopRef.current;
    if (range && (element.currentTime < range.start || element.currentTime >= range.end)) {
      element.currentTime = range.start;
    }
    element.play().catch(() => setPlaying(false));
  }, []);

  const play = useCallback(() => {
    if (audioRef.current) start(audioRef.current);
  }, [start]);

  const pause = useCallback(() => audioRef.current?.pause(), []);

  const toggle = useCallback(() => {
    const element = audioRef.current;
    if (!element) return;
    if (element.paused) start(element);
    else element.pause();
  }, [start]);

  const seek = useCallback((time: number) => {
    const element = audioRef.current;
    if (!element) return;
    const max = Number.isFinite(element.duration) ? element.duration : time;
    element.currentTime = Math.min(Math.max(0, time), max);
    setCurrentTime(element.currentTime);
  }, []);

  const setRate = useCallback((value: number) => {
    if (audioRef.current) audioRef.current.playbackRate = value;
    outputRef.current.rate = value;
    setRateState(value);
  }, []);

  const setMuted = useCallback((value: boolean) => {
    if (audioRef.current) audioRef.current.muted = value;
    outputRef.current.muted = value;
    setMutedState(value);
  }, []);

  /** Set or clear the A-B loop; the ends are ordered and a too-short range clears it. */
  const setLoop = useCallback((range: LoopRange | null) => {
    const next =
      range && Math.abs(range.end - range.start) >= MIN_LOOP
        ? { start: Math.max(0, Math.min(range.start, range.end)), end: Math.max(range.start, range.end) }
        : null;
    loopRef.current = next;
    setLoopState(next);
  }, []);

  return {
    audio,
    playing,
    currentTime,
    duration,
    rate,
    muted,
    loop,
    play,
    pause,
    toggle,
    seek,
    setRate,
    setMuted,
    setLoop,
  };
}

/** Keep the playhead visible in a horizontally scrolling container. */
export function useFollowPlayhead(
  ref: { current: HTMLElement | null },
  currentTime: number,
  pixelsPerSecond: number,
  enabled: boolean,
) {
  useEffect(() => {
    const el = ref.current;
    if (!enabled || !el) return;
    const x = currentTime * pixelsPerSecond;
    const margin = el.clientWidth * 0.1;
    if (x < el.scrollLeft || x > el.scrollLeft + el.clientWidth - margin) {
      el.scrollLeft = Math.max(0, x - margin);
    }
  }, [ref, currentTime, pixelsPerSecond, enabled]);
}
