"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Analysis, ModelInfo, Note, Project, Source, Transcription } from "@music-transcriber/shared-types";
import GuitarTab from "@/components/GuitarTab";
import NotationView from "@/components/NotationView";
import NoteEditor from "@/components/NoteEditor";
import PianoRoll from "@/components/PianoRoll";
import PlaybackControls, { type ListenMode } from "@/components/PlaybackControls";
import TuningPicker from "@/components/TuningPicker";
import Waveform from "@/components/Waveform";
import {
  SOURCE_LABELS,
  STATUS_LABELS,
  analyzeNotes,
  audioUrl,
  errorMessage,
  getModelInfo,
  getProject,
  getTranscription,
  isOutdated,
  midiUrl,
  modelLabel,
  musicXmlUrl,
  retranscribe,
  saveNotes,
  stageLabel,
  tabUrl,
} from "@/lib/api";
import { type Drag, applyDrag, copyNotes, deleteNotes, pasteNotes } from "@/lib/editing";
import { KEY_NAMES, STANDARD_TUNING, beatGrid, defaultPosition, formatTime } from "@/lib/music";
import { useSettings } from "@/lib/settings";
import { useHistory } from "@/lib/useHistory";
import { renderNotes, useNotePlayer } from "@/lib/synth";
import { usePlayback } from "@/lib/usePlayback";
import { encodeWav } from "@/lib/wav";

// Same bounds as apps/api/app/schemas/project.py.
const MIN_TEMPO = 20;
const MAX_TEMPO = 400;
const METERS = [2, 3, 4, 5, 6, 7];
const ANALYSIS_DELAY_MS = 300;
// One shared empty list, so "no notes yet" never looks like an unsaved edit.
const NO_NOTES: Note[] = [];

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
  // Notes with undo/redo; `savedNotes` is what the server has, so undoing back to it
  // clears the unsaved state.
  const history = useHistory<Note[]>(NO_NOTES);
  const notes = history.value;
  const [savedNotes, setSavedNotes] = useState<Note[]>(NO_NOTES);
  const [tempo, setTempo] = useState<number | null>(null);
  // Notation chosen by the user; null key / downbeat mean "estimate from the notes".
  const [beatsPerMeasure, setBeatsPerMeasure] = useState(4);
  const [keyChoice, setKeyChoice] = useState<string | null>(null);
  const [downbeatChoice, setDownbeatChoice] = useState<number | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysisPending, setAnalysisPending] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  // Selected note indices in click order; the last one is shown in the note editor.
  const [selection, setSelection] = useState<number[]>([]);
  const [notationDirty, setNotationDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [songMode, setSongMode] = useState(false);
  // Guitar or voice recording, for the next transcription like the tuning below.
  const [source, setSource] = useState<Source>("guitar");
  const [rendering, setRendering] = useState(false);
  // Tuning and capo for the next transcription; they take effect with "Yeniden çözümle".
  const [tuningName, setTuningName] = useState("standard");
  const [capo, setCapo] = useState(0);
  const [listen, setListen] = useState<ListenMode>("audio");
  const [metronome, setMetronome] = useState(false);
  const [models, setModels] = useState<ModelInfo | null>(null);
  const [showScore, setShowScore] = useState(false);

  useEffect(() => {
    getModelInfo()
      .then(setModels)
      .catch(() => setModels(null));
  }, []);

  const { reset: resetHistory } = history;
  const applyTranscription = useCallback(
    (t: Transcription) => {
      setTranscription(t);
      resetHistory(t.notes);
      setSavedNotes(t.notes);
      setTempo(t.tempo);
      setBeatsPerMeasure(t.beats_per_measure);
      setKeyChoice(t.key);
      setDownbeatChoice(t.downbeat);
      setNotationDirty(false);
    },
    [resetHistory],
  );
  const dirty = notes !== savedNotes || notationDirty;
  const primary = selection.length ? selection[selection.length - 1] : null;
  const selectionSet = useMemo(() => new Set(selection), [selection]);

  // Undo/redo can remove notes that were selected.
  useEffect(() => {
    setSelection((current) => (current.every((i) => i < notes.length) ? current : current.filter((i) => i < notes.length)));
  }, [notes.length]);

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
        setSource(p.source);
        setTuningName(p.tuning_name);
        setCapo(p.capo);
        if (p.status === "completed") {
          const t = await getTranscription(id);
          if (cancelled) return;
          applyTranscription(t);
          setSelection([]);
        } else if (p.status === "pending" || p.status === "processing") {
          timer = setTimeout(load, 1000);
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
  const scoreRequest = useMemo(
    () =>
      transcription
        ? {
            notes,
            tempo,
            beats_per_measure: beatsPerMeasure,
            key: keyChoice,
            downbeat: downbeatChoice,
            tuning,
            capo: transcription.capo,
            title: project?.name ?? "Transcription",
          }
        : null,
    [notes, tempo, beatsPerMeasure, keyChoice, downbeatChoice, tuning, transcription, project?.name],
  );

  const { setMuted } = playback;
  useEffect(() => setMuted(listen === "notes"), [listen, setMuted]);
  useNotePlayer({ audio: playback.audio, playing: playback.playing, notes, synth: listen !== "audio", clicks });

  const { set: setNotes, undo, redo } = history;
  const updateNote = useCallback(
    (note: Note) => setNotes(notes.map((n, i) => (i === primary ? note : n))),
    [notes, primary, setNotes],
  );

  const deleteSelected = useCallback(() => {
    if (selection.length === 0) return;
    setNotes(deleteNotes(notes, selection));
    setSelection([]);
  }, [notes, selection, setNotes]);

  const select = useCallback((index: number | null, additive: boolean) => {
    if (index === null) setSelection([]);
    else if (additive)
      setSelection((current) => (current.includes(index) ? current.filter((i) => i !== index) : [...current, index]));
    else setSelection([index]);
  }, []);

  const selectMany = useCallback((indices: number[], additive: boolean) => {
    setSelection((current) => (additive ? [...current, ...indices.filter((i) => !current.includes(i))] : indices));
  }, []);

  // A drag is applied to the notes as they were when it started, one undo step at the end.
  // It reads the latest selection: pressing an unselected note selects it in the same
  // gesture, after the component handling the pointer captured this callback.
  const latest = useRef({ notes, selection, tuning });
  latest.current = { notes, selection, tuning };
  const dragBase = useRef<Note[] | null>(null);
  const drag = useCallback(
    (change: Drag, done: boolean) => {
      const current = latest.current;
      const base = dragBase.current ?? current.notes;
      dragBase.current = done ? null : base;
      setNotes(applyDrag(base, current.selection, change, current.tuning), !done);
    },
    [setNotes],
  );

  const clipboard = useRef<Note[]>([]);

  const addNote = () => {
    const start = playback.currentTime;
    const pitch = primary !== null ? notes[primary]?.pitch ?? 64 : 64;
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
    setNotes([...notes, note]);
    setSelection([notes.length]);
  };

  const editNotation = (change: () => void) => {
    change();
    setNotationDirty(true);
  };

  /** The selected note's onset (else the playhead) becomes beat 1 of a bar. */
  const markDownbeat = () => {
    const note = primary !== null ? notes[primary] : undefined;
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
    const lost = dirty ? "Kaydedilmemiş değişiklikler" : project?.edited ? "Kaydettiğiniz düzenlemeler" : null;
    if (lost && !confirm(`${lost} kaybolacak. Devam edilsin mi?`)) return;
    try {
      setProject(await retranscribe(id, { source, separateGuitar: songMode, tuning: tuningName, capo }));
      setTranscription(null);
      setAnalysis(null);
      resetHistory(NO_NOTES);
      setSavedNotes(NO_NOTES);
      setNotationDirty(false);
      setSelection([]);
      setReloadKey((k) => k + 1);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  // The notes (saved or not) played on the synthesized guitar, as a WAV download.
  const downloadGuitar = async () => {
    setRendering(true);
    setError(null);
    try {
      const rendered = await renderNotes(notes);
      const url = URL.createObjectURL(encodeWav([rendered.getChannelData(0)], rendered.sampleRate));
      const link = document.createElement("a");
      link.href = url;
      link.download = `${project?.name ?? "transcription"} (gitar).wav`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setRendering(false);
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

  // Space: play/pause · Delete: remove notes · Esc: deselect · A/B/L: loop · M: metronome ·
  // arrows: nudge (Shift: more) · Ctrl+Z/Y: undo/redo · Ctrl+C/V: copy/paste at the playhead.
  useEffect(() => {
    const nudge = (dt: number, steps: number) => {
      if (selection.length) setNotes(applyDrag(notes, selection, { kind: "move", dt, steps }, tuning));
    };
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      const key = e.key.toLowerCase();
      if (e.ctrlKey || e.metaKey) {
        const handled = ["z", "y", "c", "v", "a"].includes(key);
        if (key === "z" && e.shiftKey) redo();
        else if (key === "z") undo();
        else if (key === "y") redo();
        else if (key === "c") clipboard.current = copyNotes(notes, selection);
        else if (key === "v" && clipboard.current.length) {
          const pasted = pasteNotes(notes, clipboard.current, audio?.currentTime ?? 0);
          setNotes(pasted.notes);
          setSelection(pasted.selection);
        } else if (key === "a") setSelection(notes.map((_, i) => i));
        if (handled) e.preventDefault();
        return;
      }
      if (e.altKey) return;
      const arrows: Record<string, [number, number]> = {
        ArrowUp: [0, e.shiftKey ? 12 : 1],
        ArrowDown: [0, e.shiftKey ? -12 : -1],
        ArrowLeft: [e.shiftKey ? -0.1 : -0.01, 0],
        ArrowRight: [e.shiftKey ? 0.1 : 0.01, 0],
      };
      if (e.key in arrows && selection.length) {
        e.preventDefault();
        nudge(...arrows[e.key]);
      } else if (e.code === "Space") {
        e.preventDefault();
        toggle();
      } else if ((e.key === "Delete" || e.key === "Backspace") && selection.length) {
        e.preventDefault();
        deleteSelected();
      } else if (e.key === "Escape") {
        setSelection([]);
      } else if (e.shiftKey) {
        return;
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
  }, [toggle, deleteSelected, selection, notes, setNotes, undo, redo, audio, tuning, loopFromHere, loopUntilHere, setLoop]);

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
          {project?.source === "voice" && <span className="badge">Ses</span>}
          {project?.separate_guitar && <span className="badge">Şarkı modu</span>}
        </div>
        <div className="row">
          <button className="primary play-button" onClick={toggle}>
            {playback.playing ? "❚❚ Duraklat" : "▶ Oynat"}
          </button>
          <button onClick={addNote} disabled={!transcription}>
            Nota ekle
          </button>
          <button onClick={undo} disabled={!history.canUndo} title="Geri al (Ctrl+Z)">
            ↶ Geri al
          </button>
          <button onClick={redo} disabled={!history.canRedo} title="Yinele (Ctrl+Y)">
            ↷ Yinele
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
          <button
            onClick={() => void downloadGuitar()}
            disabled={!notes.length || rendering}
            title="Notaları sentezlenmiş gitar sesiyle çalıp WAV olarak indirir (kaydedilmemiş düzenlemeler dahil)"
          >
            {rendering ? "Hazırlanıyor…" : "Gitar sesi indir"}
          </button>
          <label className="row compact small" title="Kayıtta ne var?">
            Kaynak
            <select value={source} onChange={(e) => setSource(e.target.value as Source)} disabled={busy}>
              {(Object.keys(SOURCE_LABELS) as Source[]).map((key) => (
                <option key={key} value={key}>
                  {SOURCE_LABELS[key]}
                </option>
              ))}
            </select>
          </label>
          <TuningPicker
            tuning={tuningName}
            capo={capo}
            disabled={busy}
            onChange={(nextTuning, nextCapo) => {
              setTuningName(nextTuning);
              setCapo(nextCapo);
            }}
          />
          <label className="row compact small" title="Başka enstrümanlar da çalan kayıtlar için">
            <input type="checkbox" checked={songMode} onChange={(e) => setSongMode(e.target.checked)} />
            {source === "voice" ? "Vokali ayır" : "Gitarı ayır"}
          </label>
          <button onClick={() => void rerun()} disabled={busy || !project}>
            Yeniden çözümle
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {busy && (
        <div className="card stack tight">
          <div className="row between">
            <span>{project ? stageLabel(project) : "Sırada bekliyor"}…</span>
            <span className="muted">{project?.progress != null ? `%${Math.round(project.progress * 100)}` : ""}</span>
          </div>
          <div className="progress" aria-label="Çözümleme ilerlemesi">
            <div style={{ width: `${Math.round((project?.progress ?? 0) * 100)}%` }} />
          </div>
          <p className="muted small">Sayfa otomatik olarak güncellenecek.</p>
        </div>
      )}
      {project?.status === "failed" && <div className="card error">Çözümleme başarısız: {project.error}</div>}
      {dirty && <p className="muted small">Kaydedilmemiş değişiklikler var. Dışa aktarmadan önce kaydedin.</p>}
      {project && (project.tuning_name !== tuningName || project.capo !== capo) && (
        <p className="muted small">Akort veya capo değişti; tel ve perdeler &quot;Yeniden çözümle&quot; ile güncellenir.</p>
      )}
      {project && project.source !== source && (
        <p className="muted small">Kaynak değişti; &quot;Yeniden çözümle&quot; ile uygulanır.</p>
      )}
      {transcription && transcription.transpose !== 0 && (
        <p className="muted small">
          Melodi gitarın aralığına sığsın diye {Math.abs(transcription.transpose / 12)} oktav{" "}
          {transcription.transpose < 0 ? "aşağı" : "yukarı"} taşındı; kayıtla birlikte dinlerken notalar bu kadar farklı
          duyulur.
        </p>
      )}
      {project && models && isOutdated(project, models) && (
        <div className="card notice row between">
          <span>
            Bu kayıt {project.model_version ? `eski bir modelle (${modelLabel(project.model_version)})` : "eski bir modelle"}{" "}
            çözüldü. Güncel model: {modelLabel(models.version)}.
            {project.edited && " Yeniden çözümlerseniz kaydettiğiniz düzenlemeler kaybolur."}
          </span>
          <button onClick={() => void rerun()} disabled={busy}>
            Güncel modelle yeniden çözümle
          </button>
        </div>
      )}

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
                  capo={transcription.capo}
                  duration={duration}
                  currentTime={playback.currentTime}
                  pixelsPerSecond={settings.pixelsPerSecond}
                  selection={selectionSet}
                  onSelect={select}
                  onSelectMany={selectMany}
                  onDrag={drag}
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
                  selection={selectionSet}
                  onSelect={select}
                  onSelectMany={selectMany}
                  onDrag={drag}
                  onSeek={playback.seek}
                  follow={settings.followPlayhead}
                  grid={grid}
                  loop={loop}
                  musicKey={musicKey}
                />
              )}
              <p className="muted small">
                Tıkla: seç · Shift/Ctrl+tıkla: seçime ekle · Boş alanda sürükle: alan seç · Notayı sürükle: taşı
                (TAB&apos;da yukarı/aşağı: başka tel) · Nota ucunu sürükle: uzat/kısalt · Oklar: perde / zaman
                (Shift: oktav / 0.1 sn) · Ctrl+Z / Ctrl+Y: geri al / yinele · Ctrl+C / Ctrl+V: kopyala / oynatma
                konumuna yapıştır · Ctrl+A: tümünü seç · Delete: sil · Boşluk: oynat · A / B / L: döngü · M: metronom
              </p>
            </div>
            <NoteEditor
              note={primary !== null ? notes[primary] ?? null : null}
              selectedCount={selection.length}
              tuning={tuning}
              capo={transcription.capo}
              musicKey={musicKey}
              onChange={updateNote}
              onDelete={deleteSelected}
            />
          </div>

          <section className="stack tight">
            <div className="row between">
              <h3 className="section-title">Nota ve TAB (MusicXML önizlemesi)</h3>
              <button onClick={() => setShowScore((on) => !on)}>{showScore ? "Gizle" : "Notayı göster"}</button>
            </div>
            {showScore && scoreRequest && <NotationView request={scoreRequest} />}
          </section>

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
                {primary !== null ? "Seçili nota 1. vuruş" : "Buradan başlat"}
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
