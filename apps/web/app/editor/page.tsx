"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import type { Analysis, Note, Project, Transcription } from "@music-transcriber/shared-types";
import GuitarTab from "@/components/GuitarTab";
import NoteEditor from "@/components/NoteEditor";
import PianoRoll from "@/components/PianoRoll";
import PlaybackControls, { type ListenMode } from "@/components/PlaybackControls";
import Waveform from "@/components/Waveform";
import {
  STATUS_LABELS,
  analyzeNotes,
  audioUrl,
  errorMessage,
  getProject,
  getTranscription,
  midiUrl,
  musicXmlUrl,
  retranscribe,
  saveNotes,
  tabUrl,
} from "@/lib/api";
import { KEY_NAMES, STANDARD_TUNING, beatGrid, defaultPosition, formatTime } from "@/lib/music";
import { useSettings } from "@/lib/settings";
import { useNotePlayer } from "@/lib/synth";
import { usePlayback } from "@/lib/usePlayback";

// Same bounds as apps/api/app/schemas/project.py.
const MIN_TEMPO = 20;
const MAX_TEMPO = 400;
const METERS = [2, 3, 4, 5, 6, 7];
const ANALYSIS_DELAY_MS = 300;

export default function EditorPage() {
  return (
    <Suspense fallback={<p className="muted">Yükleniyor…</p>}>
      <Editor />
    </Suspense>
  );
}

function Editor() {
  const id = useSearchParams().get("id");
  const { settings } = useSettings();
  const playback = usePlayback(id ? audioUrl(id) : null);

  const [project, setProject] = useState<Project | null>(null);
  const [transcription, setTranscription] = useState<Transcription | null>(null);
  const [notes, setNotes] = useState<Note[]>([]);
  const [tempo, setTempo] = useState<number | null>(null);
  // Notation chosen by the user; null key / downbeat mean "estimate from the notes".
  const [beatsPerMeasure, setBeatsPerMeasure] = useState(4);
  const [keyChoice, setKeyChoice] = useState<string | null>(null);
  const [downbeatChoice, setDownbeatChoice] = useState<number | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysisPending, setAnalysisPending] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [songMode, setSongMode] = useState(false);
  const [listen, setListen] = useState<ListenMode>("audio");
  const [metronome, setMetronome] = useState(false);

  const applyTranscription = useCallback((t: Transcription) => {
    setTranscription(t);
    setNotes(t.notes);
    setTempo(t.tempo);
    setBeatsPerMeasure(t.beats_per_measure);
    setKeyChoice(t.key);
    setDownbeatChoice(t.downbeat);
    setDirty(false);
  }, []);

  // Load the project and poll until the transcription is finished.
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      try {
        const p = await getProject(id);
        if (cancelled) return;
        setProject(p);
        setSongMode(p.separate_guitar);
        if (p.status === "completed") {
          const t = await getTranscription(id);
          if (cancelled) return;
          applyTranscription(t);
          setSelected(null);
        } else if (p.status === "pending" || p.status === "processing") {
          timer = setTimeout(load, 2000);
        }
      } catch (err) {
        if (!cancelled) setError(errorMessage(err));
      }
    };
    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [id, reloadKey, applyTranscription]);

  // Key, bar grid and chords follow every edit (debounced; stale requests are aborted).
  useEffect(() => {
    if (!transcription) return;
    const controller = new AbortController();
    setAnalysisPending(true);
    const timer = setTimeout(() => {
      analyzeNotes(
        { notes, tempo, beats_per_measure: beatsPerMeasure, key: keyChoice, downbeat: downbeatChoice },
        controller.signal,
      )
        .then((result) => {
          setAnalysis(result);
          setAnalysisError(null);
          setAnalysisPending(false);
        })
        .catch((err) => {
          if (controller.signal.aborted) return;
          setAnalysisError(errorMessage(err));
          setAnalysisPending(false);
        });
    }, ANALYSIS_DELAY_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [transcription, notes, tempo, beatsPerMeasure, keyChoice, downbeatChoice]);

  // Warn before leaving with unsaved edits.
  useEffect(() => {
    if (!dirty) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [dirty]);

  const tuning = transcription?.tuning ?? STANDARD_TUNING;
  const duration = useMemo(
    () => notes.reduce((max, n) => Math.max(max, n.end), Math.max(project?.duration ?? 0, playback.duration)),
    [notes, project?.duration, playback.duration],
  );
  const meanConfidence = useMemo(() => {
    const values = notes.map((n) => n.confidence).filter((c): c is number => c !== null);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  }, [notes]);
  const grid = useMemo(() => {
    if (!analysis) return null;
    const firstNote = notes.reduce((min, n) => Math.min(min, n.start), Infinity);
    return beatGrid(
      analysis.tempo,
      analysis.beats_per_measure,
      analysis.downbeat,
      Number.isFinite(firstNote) ? firstNote : 0,
      duration,
    );
  }, [analysis, notes, duration]);
  const clicks = useMemo(
    () => (metronome && grid ? { beats: grid.beats, bars: grid.bars.map((b) => b.time) } : null),
    [metronome, grid],
  );
  const musicKey = analysis?.key ?? null;

  const { setMuted } = playback;
  useEffect(() => setMuted(listen === "notes"), [listen, setMuted]);
  useNotePlayer({ audio: playback.audio, playing: playback.playing, notes, synth: listen !== "audio", clicks });

  const updateNote = useCallback(
    (note: Note) => {
      setNotes((prev) => prev.map((n, i) => (i === selected ? note : n)));
      setDirty(true);
    },
    [selected],
  );

  const deleteSelected = useCallback(() => {
    if (selected === null) return;
    setNotes((prev) => prev.filter((_, i) => i !== selected));
    setSelected(null);
    setDirty(true);
  }, [selected]);

  const addNote = () => {
    const start = playback.currentTime;
    const pitch = selected !== null ? notes[selected]?.pitch ?? 64 : 64;
    const position = defaultPosition(pitch, tuning);
    const note: Note = {
      pitch,
      start,
      end: start + 0.25,
      velocity: 80,
      string: position?.string ?? null,
      fret: position?.fret ?? null,
      confidence: null,
    };
    setNotes((prev) => [...prev, note]);
    setSelected(notes.length);
    setDirty(true);
  };

  const editNotation = (change: () => void) => {
    change();
    setDirty(true);
  };

  /** The selected note's onset (else the playhead) becomes beat 1 of a bar. */
  const markDownbeat = () => {
    const note = selected !== null ? notes[selected] : undefined;
    const time = note ? note.start : playback.currentTime;
    editNotation(() => setDownbeatChoice(Math.round(time * 1000) / 1000));
  };

  const save = async () => {
    if (!id) return;
    setSaving(true);
    setError(null);
    try {
      applyTranscription(
        await saveNotes(id, notes, {
          tempo,
          beats_per_measure: beatsPerMeasure,
          key: keyChoice,
          downbeat: downbeatChoice,
        }),
      );
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const rerun = async () => {
    if (!id) return;
    if (dirty && !confirm("Kaydedilmemiş değişiklikler kaybolacak. Devam edilsin mi?")) return;
    try {
      setProject(await retranscribe(id, songMode));
      setTranscription(null);
      setAnalysis(null);
      setNotes([]);
      setSelected(null);
      setDirty(false);
      setReloadKey((k) => k + 1);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  // Read the position from the element so these (and the key handler) stay stable while playing.
  const { toggle, setLoop, loop, audio } = playback;
  const loopFromHere = useCallback(() => {
    const now = audio?.currentTime ?? 0;
    setLoop({ start: now, end: loop && loop.end > now ? loop.end : duration });
  }, [audio, loop, duration, setLoop]);
  const loopUntilHere = useCallback(() => {
    const now = audio?.currentTime ?? 0;
    setLoop({ start: loop && loop.start < now ? loop.start : 0, end: now });
  }, [audio, loop, setLoop]);

  // Space: play/pause · Delete/Backspace: remove note · Esc: deselect · A/B/L: loop · M: metronome.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (e.code === "Space") {
        e.preventDefault();
        toggle();
      } else if ((e.key === "Delete" || e.key === "Backspace") && selected !== null) {
        e.preventDefault();
        deleteSelected();
      } else if (e.key === "Escape") {
        setSelected(null);
      } else if (key === "a") {
        loopFromHere();
      } else if (key === "b") {
        loopUntilHere();
      } else if (key === "l") {
        setLoop(null);
      } else if (key === "m") {
        setMetronome((on) => !on);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, deleteSelected, selected, loopFromHere, loopUntilHere, setLoop]);

  if (!id) {
    return (
      <div className="card">
        Proje seçilmedi. <Link href="/projects">Projelere dön</Link>
      </div>
    );
  }

  const busy = project?.status === "pending" || project?.status === "processing";
  const canExport = transcription !== null && !dirty;

  return (
    <div className="stack">
      <div className="editor-header">
        <div className="row">
          <h1>{project?.name ?? "Yükleniyor…"}</h1>
          {project && <span className={`badge ${project.status}`}>{STATUS_LABELS[project.status]}</span>}
          {project?.separate_guitar && <span className="badge">Şarkı modu</span>}
        </div>
        <div className="row">
          <button className="primary play-button" onClick={toggle}>
            {playback.playing ? "❚❚ Duraklat" : "▶ Oynat"}
          </button>
          <button onClick={addNote} disabled={!transcription}>
            Nota ekle
          </button>
          <button onClick={() => void save()} disabled={!dirty || saving}>
            {saving ? "Kaydediliyor…" : "Kaydet"}
          </button>
          {[
            { label: "MIDI indir", href: midiUrl(id) },
            { label: "MusicXML indir", href: musicXmlUrl(id), title: "MuseScore / Guitar Pro" },
            { label: "TAB indir", href: tabUrl(id) },
          ].map(({ label, href, title }) =>
            canExport ? (
              <a key={label} className="button" href={href} title={title}>
                {label}
              </a>
            ) : (
              <button key={label} disabled title={dirty ? "Önce kaydedin" : undefined}>
                {label}
              </button>
            ),
          )}
          <label className="row compact small" title="Davul, bas veya vokal içeren kayıtlar için">
            <input type="checkbox" checked={songMode} onChange={(e) => setSongMode(e.target.checked)} />
            Gitarı ayır
          </label>
          <button onClick={() => void rerun()} disabled={busy || !project}>
            Yeniden çözümle
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {busy && <div className="card muted">Model kaydı çözümlüyor… Sayfa otomatik olarak güncellenecek.</div>}
      {project?.status === "failed" && <div className="card error">Çözümleme başarısız: {project.error}</div>}
      {dirty && <p className="muted small">Kaydedilmemiş değişiklikler var. Dışa aktarmadan önce kaydedin.</p>}

      <Waveform
        audioUrl={audioUrl(id)}
        currentTime={playback.currentTime}
        duration={playback.duration || duration}
        onSeek={playback.seek}
        loop={loop}
        onLoopChange={setLoop}
      />
      <PlaybackControls
        playing={playback.playing}
        currentTime={playback.currentTime}
        duration={playback.duration || duration}
        rate={playback.rate}
        onToggle={toggle}
        onSeek={playback.seek}
        onRateChange={playback.setRate}
        listen={listen}
        onListenChange={setListen}
        metronome={metronome}
        onMetronomeChange={setMetronome}
        loop={loop}
        onLoopStart={loopFromHere}
        onLoopEnd={loopUntilHere}
        onLoopClear={() => setLoop(null)}
      />

      {transcription && (
        <>
          <div className="editor-layout">
            <div className="stack">
              {settings.showTab && (
                <GuitarTab
                  notes={notes}
                  tuning={tuning}
                  duration={duration}
                  currentTime={playback.currentTime}
                  pixelsPerSecond={settings.pixelsPerSecond}
                  selectedIndex={selected}
                  onSelect={setSelected}
                  onSeek={playback.seek}
                  follow={settings.followPlayhead}
                  grid={grid}
                  chords={analysis?.chords}
                  chordsStale={analysisPending}
                  loop={loop}
                />
              )}
              {settings.showPianoRoll && (
                <PianoRoll
                  notes={notes}
                  duration={duration}
                  currentTime={playback.currentTime}
                  pixelsPerSecond={settings.pixelsPerSecond}
                  selectedIndex={selected}
                  onSelect={setSelected}
                  onSeek={playback.seek}
                  follow={settings.followPlayhead}
                  grid={grid}
                  loop={loop}
                  musicKey={musicKey}
                />
              )}
              <p className="muted small">
                Boşluk: oynat/duraklat · Delete: seçili notayı sil · Esc: seçimi kaldır · A / B: döngü başı / sonu ·
                L: döngüyü kaldır · M: metronom
              </p>
            </div>
            <NoteEditor
              note={selected !== null ? notes[selected] ?? null : null}
              tuning={tuning}
              musicKey={musicKey}
              onChange={updateNote}
              onDelete={deleteSelected}
            />
          </div>

          <div className="card stats-bar">
            <span>
              Tespit edilen nota: <strong>{notes.length}</strong>
            </span>
            <span title="Kullanıcı kayıtlarında referans nota olmadığından doğruluk yerine modelin ortalama güveni gösterilir.">
              Ortalama güven: <strong>{meanConfidence !== null ? `%${Math.round(meanConfidence * 100)}` : "—"}</strong>
            </span>
            <label
              className="row compact"
              title="Tahmini tempo. Yanlışsa düzeltin: vuruş ızgarası, metronom, MIDI ve MusicXML ritmi bu tempoya göre."
            >
              BPM:
              <input
                className="tempo-input"
                type="number"
                min={MIN_TEMPO}
                max={MAX_TEMPO}
                step={0.1}
                value={tempo !== null ? Math.round(tempo * 10) / 10 : ""}
                onChange={(e) => {
                  const value = e.target.valueAsNumber;
                  if (Number.isNaN(value) || value < MIN_TEMPO || value > MAX_TEMPO) return;
                  editNotation(() => setTempo(value));
                }}
              />
            </label>
            <label className="row compact" title="Ölçüdeki vuruş sayısı (dörtlük vuruş)">
              Ölçü:
              <select
                value={beatsPerMeasure}
                onChange={(e) => editNotation(() => setBeatsPerMeasure(Number(e.target.value)))}
              >
                {METERS.map((beats) => (
                  <option key={beats} value={beats}>
                    {beats}/4
                  </option>
                ))}
              </select>
            </label>
            <label className="row compact" title="Armür, nota ve akor isimleri bu tona göre yazılır">
              Ton:
              <select
                className="notation-select"
                value={keyChoice ?? ""}
                onChange={(e) => editNotation(() => setKeyChoice(e.target.value || null))}
              >
                <option value="">Otomatik{analysis?.estimated_key ? ` (${analysis.estimated_key.name})` : ""}</option>
                {KEY_NAMES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <span className="row compact">
              Ölçü başı:{" "}
              <strong>{downbeatChoice !== null ? formatTime(downbeatChoice) : "otomatik"}</strong>
              <button
                onClick={markDownbeat}
                title="Seçili notanın başlangıcını (seçim yoksa oynatma konumunu) ölçünün 1. vuruşu yapar"
              >
                {selected !== null ? "Seçili nota 1. vuruş" : "Buradan başlat"}
              </button>
              {downbeatChoice !== null && (
                <button onClick={() => editNotation(() => setDownbeatChoice(null))}>Otomatik</button>
              )}
            </span>
            {analysisError && <span className="error small">Analiz yapılamadı: {analysisError}</span>}
          </div>
        </>
      )}
    </div>
  );
}
