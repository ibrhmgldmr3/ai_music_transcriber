"use client";

import { useEffect, useRef, useState } from "react";
import type { RenderRequest } from "@music-transcriber/shared-types";
import { errorMessage, renderMusicXml } from "@/lib/api";

const RENDER_DELAY_MS = 800;

type Osmd = import("opensheetmusicdisplay").OpenSheetMusicDisplay;

/**
 * Standard notation and TAB as the MusicXML export engraves them (OpenSheetMusicDisplay),
 * redrawn shortly after every edit. The library is loaded only when this is shown.
 */
export default function NotationView({ request }: { request: RenderRequest }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const osmdRef = useRef<Osmd | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        setStatus("loading");
        const xml = await renderMusicXml(request, controller.signal);
        if (!osmdRef.current) {
          const { OpenSheetMusicDisplay } = await import("opensheetmusicdisplay");
          if (controller.signal.aborted || !containerRef.current) return;
          osmdRef.current = new OpenSheetMusicDisplay(containerRef.current, {
            autoResize: true,
            backend: "svg",
            drawTitle: false,
            drawPartNames: false,
            drawMeasureNumbers: true,
          });
        }
        if (controller.signal.aborted) return;
        await osmdRef.current.load(xml);
        osmdRef.current.render();
        setStatus("ready");
        setError(null);
      } catch (err) {
        if (controller.signal.aborted) return;
        setStatus("error");
        setError(errorMessage(err));
      }
    }, RENDER_DELAY_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [request]);

  return (
    <div className="stack tight">
      {status === "loading" && <p className="muted small">Nota çiziliyor…</p>}
      {error && <p className="error small">Nota çizilemedi: {error}</p>}
      <div ref={containerRef} className="score" />
    </div>
  );
}
