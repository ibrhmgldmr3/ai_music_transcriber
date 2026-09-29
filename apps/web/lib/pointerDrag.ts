import type { PointerEvent as ReactPointerEvent } from "react";

export interface PointerDelta {
  /** Pixels moved since the pointer went down. */
  dx: number;
  dy: number;
  shiftKey: boolean;
}

/** Pointer travel below which a press counts as a click, not a drag. */
const THRESHOLD = 3;

/**
 * Follow a pointer from `pointerdown` until it is released, on the window so the drag
 * continues outside the element. `onMove` starts once the pointer has moved a few
 * pixels; `onEnd` gets the final delta, or null when it was just a click.
 */
export function startPointerDrag(
  event: ReactPointerEvent,
  onMove: (delta: PointerDelta) => void,
  onEnd: (delta: PointerDelta | null) => void,
): void {
  const x0 = event.clientX;
  const y0 = event.clientY;
  let last: PointerDelta | null = null;

  const move = (e: PointerEvent) => {
    const delta = { dx: e.clientX - x0, dy: e.clientY - y0, shiftKey: e.shiftKey };
    if (!last && Math.hypot(delta.dx, delta.dy) < THRESHOLD) return;
    last = delta;
    onMove(delta);
  };
  const stop = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
    window.removeEventListener("pointercancel", stop);
    onEnd(last);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop);
  window.addEventListener("pointercancel", stop);
}
