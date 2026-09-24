import { useCallback, useEffect, useState } from "react";

export interface EditorSettings {
  /** Horizontal zoom of the piano roll and tab. */
  pixelsPerSecond: number;
  showTab: boolean;
  showPianoRoll: boolean;
  followPlayhead: boolean;
}

export const DEFAULT_SETTINGS: EditorSettings = {
  pixelsPerSecond: 120,
  showTab: true,
  showPianoRoll: true,
  followPlayhead: true,
};

const STORAGE_KEY = "music-transcriber:settings";

const MIN_ZOOM = 20;
const MAX_ZOOM = 400;

/** Keep only known keys with the right types; stored data may be stale or hand-edited. */
function sanitize(value: unknown): EditorSettings {
  const stored = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const flag = (key: keyof EditorSettings) =>
    typeof stored[key] === "boolean" ? (stored[key] as boolean) : (DEFAULT_SETTINGS[key] as boolean);
  const zoom = stored.pixelsPerSecond;
  return {
    pixelsPerSecond:
      typeof zoom === "number" && Number.isFinite(zoom)
        ? Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom))
        : DEFAULT_SETTINGS.pixelsPerSecond,
    showTab: flag("showTab"),
    showPianoRoll: flag("showPianoRoll"),
    followPlayhead: flag("followPlayhead"),
  };
}

function readSettings(): EditorSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? sanitize(JSON.parse(raw)) : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function writeSettings(settings: EditorSettings): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // storage unavailable (private mode) - settings just won't persist
  }
}

/** Per-browser editor preferences persisted in localStorage. */
export function useSettings() {
  const [settings, setSettings] = useState<EditorSettings>(DEFAULT_SETTINGS);

  useEffect(() => setSettings(readSettings()), []);

  const update = useCallback((patch: Partial<EditorSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      writeSettings(next);
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    writeSettings(DEFAULT_SETTINGS);
    setSettings(DEFAULT_SETTINGS);
  }, []);

  return { settings, update, reset };
}
