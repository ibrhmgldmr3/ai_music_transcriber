"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import type { Note, Project, Transcription } from "@music-transcriber/shared-types";
import GuitarTab from "@/components/GuitarTab";
import NoteEditor from "@/components/NoteEditor";
import PianoRoll from "@/components/PianoRoll";
import PlaybackControls from "@/components/PlaybackControls";
import Waveform from "@/components/Waveform";
import {
  STATUS_LABELS,
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
import { STANDARD_TUNING, defaultPosition } from "@/lib/music";
import { useSettings } from "@/lib/settings";
import { usePlayback } from "@/lib/usePlayback";

// Same bounds as NotesUpdate.tempo in apps/api/app/schemas/project.py.
const MIN_TEMPO = 20;
const MAX_TEMPO = 400;

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
  const [selected, setSelected] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [songMode, setSongMode] = useState(false);

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
          setTranscription(t);
          setNotes(t.notes);
          setTempo(t.tempo);
          setSelected(null);
          setDirty(false);
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
  }, [id, reloadKey]);

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

  const save = async () => {
    if (!id) return;
    setSaving(true);
    setError(null);
    try {
      const t = await saveNotes(id, notes, tempo);
      setTranscription(t);
      setNotes(t.notes);
      setTempo(t.tempo);
      setDirty(false);
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
      setNotes([]);
      setSelected(null);
      setDirty(false);
      setReloadKey((k) => k + 1);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  // Space: play/pause · Delete/Backspace: remove selected note · Esc: deselect.
  const { toggle } = playback;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (e.code === "Space") {
        e.preventDefault();
        toggle();
      } else if ((e.key === "Delete" || e.key === "Backspace") && selected !== null) {
        e.preventDefault();
        deleteSelected();
      } else if (e.key === "Escape") {
        setSelected(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, deleteSelected, selected]);

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
      />
      <PlaybackControls
        playing={playback.playing}
        currentTime={playback.currentTime}
        duration={playback.duration || duration}
        rate={playback.rate}
        onToggle={toggle}
        onSeek={playback.seek}
        onRateChange={playback.setRate}
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
                />
              )}
              <p className="muted small">Boşluk: oynat/duraklat · Delete: seçili notayı sil · Esc: seçimi kaldır</p>
            </div>
            <NoteEditor
              note={selected !== null ? notes[selected] ?? null : null}
              tuning={tuning}
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
              title="Tahmini tempo. Yanlışsa düzeltin: MIDI ve MusicXML ritmi bu tempoya göre yazılır."
            >
              BPM:
              <input
                className="tempo-input"
                type="number"
                min={MIN_TEMPO}
                max={MAX_TEMPO}
                step={1}
                value={tempo !== null ? Math.round(tempo) : ""}
                onChange={(e) => {
                  const value = e.target.valueAsNumber;
                  if (Number.isNaN(value) || value < MIN_TEMPO || value > MAX_TEMPO) return;
                  setTempo(value);
                  setDirty(true);
                }}
              />
            </label>
          </div>
        </>
      )}
    </div>
  );
}
