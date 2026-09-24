import { useCallback, useEffect, useRef, useState } from "react";

/** Audio playback state backed by an HTMLAudioElement. */
export function usePlayback(src: string | null) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRateState] = useState(1);

  useEffect(() => {
    if (!src) return;
    const audio = new Audio(src);
    audio.preload = "auto";
    audioRef.current = audio;

    const onMetadata = () => setDuration(Number.isFinite(audio.duration) ? audio.duration : 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onTime = () => setCurrentTime(audio.currentTime);
    audio.addEventListener("loadedmetadata", onMetadata);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onPause);
    // ~4 Hz and independent of rendering, so the position stays right even when
    // animation frames are throttled (background tab, hidden window).
    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("seeked", onTime);

    // Poll the position every frame for a smooth playhead; React skips renders while
    // the value is unchanged.
    let frame = 0;
    const tick = () => {
      setCurrentTime(audio.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frame);
      audio.pause();
      audio.removeEventListener("loadedmetadata", onMetadata);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onPause);
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("seeked", onTime);
      audioRef.current = null;
      setPlaying(false);
      setCurrentTime(0);
      setDuration(0);
    };
  }, [src]);

  const play = useCallback(() => {
    audioRef.current?.play().catch(() => setPlaying(false));
  }, []);

  const pause = useCallback(() => audioRef.current?.pause(), []);

  const toggle = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) audio.play().catch(() => setPlaying(false));
    else audio.pause();
  }, []);

  const seek = useCallback((time: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    const max = Number.isFinite(audio.duration) ? audio.duration : time;
    audio.currentTime = Math.min(Math.max(0, time), max);
    setCurrentTime(audio.currentTime);
  }, []);

  const setRate = useCallback((value: number) => {
    if (audioRef.current) audioRef.current.playbackRate = value;
    setRateState(value);
  }, []);

  return { playing, currentTime, duration, rate, play, pause, toggle, seek, setRate };
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
