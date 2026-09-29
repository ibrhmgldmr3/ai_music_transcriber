import { useCallback, useRef, useState } from "react";

interface HistoryState<T> {
  past: T[];
  present: T;
  future: T[];
}

/**
 * Undo/redo over an immutable value.
 *
 * `set(value)` records a step. `set(value, true)` changes the value without recording,
 * for the in-between states of a drag; the next plain `set` records one step from where
 * the drag started. `reset` starts a new history (e.g. after loading or saving).
 */
export function useHistory<T>(initial: T, limit = 200) {
  const [state, setState] = useState<HistoryState<T>>({ past: [], present: initial, future: [] });
  const presentRef = useRef(initial);
  presentRef.current = state.present;
  const dragOrigin = useRef<T | null>(null);

  const set = useCallback(
    (next: T, transient = false) => {
      if (transient) {
        if (dragOrigin.current === null) dragOrigin.current = presentRef.current;
        setState((s) => ({ ...s, present: next }));
        return;
      }
      const origin = dragOrigin.current ?? presentRef.current;
      dragOrigin.current = null;
      setState((s) =>
        next === origin
          ? { ...s, present: next }
          : { past: [...s.past, origin].slice(-limit), present: next, future: [] },
      );
    },
    [limit],
  );

  const undo = useCallback(() => {
    dragOrigin.current = null;
    setState((s) =>
      s.past.length
        ? { past: s.past.slice(0, -1), present: s.past[s.past.length - 1], future: [s.present, ...s.future] }
        : s,
    );
  }, []);

  const redo = useCallback(() => {
    dragOrigin.current = null;
    setState((s) =>
      s.future.length ? { past: [...s.past, s.present], present: s.future[0], future: s.future.slice(1) } : s,
    );
  }, []);

  const reset = useCallback((value: T) => {
    dragOrigin.current = null;
    setState({ past: [], present: value, future: [] });
  }, []);

  return {
    value: state.present,
    set,
    undo,
    redo,
    reset,
    canUndo: state.past.length > 0,
    canRedo: state.future.length > 0,
  };
}
