"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  Analysis,
  ModelInfo,
  Note,
  PitchCurve,
  Project,
  SongChord,
  Source,
  Transcription,
  Voicing,
} from "@music-transcriber/shared-types";
import ChordChart from "@/components/ChordChart";
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
  chordSheetUrl,
  currentVersion,
  errorMessage,
  getModelInfo,
  getPitchCurve,
  getProject,
  getTranscription,
  getVoicings,
  isOutdated,
  midiUrl,
  modelLabel,
  musicXmlUrl,
  retranscribe,
  saveNotes,
  stageLabel,
  tabUrl,
} from "@/lib/api";
import {
  type Drag,
  applyDrag,
  copyNotes,
  deleteNotes,
  pasteNotes,
  quantizeNotes,
  snapToScale,
} from "@/lib/editing";
import { KEY_NAMES, STANDARD_TUNING, defaultPosition, formatTime, gridLines } from "@/lib/music";
import { songBars, strums } from "@/lib/chords";
import RhythmPattern from "@/components/RhythmPattern";
import { sungPitch } from "@/lib/pitchCurve";
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
  // Whether the bar grid follows the beats tracked in the recording (else one tempo).
  const [beatGrid, setBeatGrid] = useState(true);
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
  const [pitchCurve, setPitchCurve] = useState<PitchCurve | null>(null);
  // Song projects: the chords (edited like the notes, saved with them) and their shapes.
  const [chords, setChords] = useState<SongChord[] | null>(null);
  const [savedChords, setSavedChords] = useState<SongChord[] | null>(null);
  const [voicings, setVoicings] = useState<ReadonlyMap<string, Voicing>>(new Map());
  // Rhythm grid of "Ritmi oturt": 2 = eighths, 4 = sixteenths.
  const [division, setDivision] = useState(4);
  const [toolMessage, setToolMessage] = useState<string | null>(null);
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
      setBeatGrid(t.beat_grid);
      setNotationDirty(false);
      setChords(t.chords);
      setSavedChords(t.chords);
    },
    [resetHistory],
  );
  const dirty = notes !== savedNotes || notationDirty || chords !== savedChords;
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
        setCapo(p.capo_auto ? -1 : p.capo);
        if (p.status === "completed") {
          const [t, curve] = await Promise.all([
            getTranscription(id),
            p.source === "voice" ? getPitchCurve(id) : Promise.resolve(null),
          ]);
          if (cancelled) return;
          applyTranscription(t);
          setPitchCurve(curve);
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
        {
          notes,
          tempo,
          beats_per_measure: beatsPerMeasure,
          key: keyChoice,
          downbeat: downbeatChoice,
          project_id: transcription.project_id,
          beat_grid: beatGrid,
          duration: Math.max(project?.duration ?? 0, playback.duration || 0),
        },
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
  }, [transcription, notes, tempo, beatsPerMeasure, keyChoice, downbeatChoice, beatGrid, project?.duration, playback.duration]);

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
  const grid = useMemo(() => (analysis ? gridLines(analysis) : null), [analysis]);
  const tracked = (transcription?.beats.length ?? 0) >= 2;
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
            project_id: transcription.project_id,
            beat_grid: beatGrid,
          }
        : null,
    [notes, tempo, beatsPerMeasure, keyChoice, downbeatChoice, tuning, transcription, project?.name, beatGrid],
  );

  // Shapes and names of the song's chords, from the capo, spelled for the key.
  const chordLabels = useMemo(() => [...new Set((chords ?? []).map((c) => c.label))].sort().join("|"), [chords]);
  useEffect(() => {
    if (!transcription || !chordLabels) return;
    const controller = new AbortController();
    getVoicings(
      {
        labels: chordLabels.split("|"),
        capo: transcription.capo,
        tuning_name: transcription.tuning_name,
        key: musicKey?.name ?? null,
      },
      controller.signal,
    )
      .then((list) => setVoicings(new Map(list.map((v) => [v.label, v]))))
      .catch(() => undefined); // chords show their labels until the next try
    return () => controller.abort();
  }, [chordLabels, transcription, musicKey?.name]);
  // The chord chart's bars are the grid's, so they follow a corrected meter or bar line.
  const bars = useMemo(
    () =>
      chords && transcription
        ? songBars(
            chords,
            analysis ? analysis.bars.map((b) => b.time) : transcription.downbeats,
            transcription.beats,
            beatsPerMeasure,
          )
        : [],
    [chords, transcription, analysis, beatsPerMeasure],
  );
  const tabChords = useMemo(
    () => chords?.map((c) => ({ start: c.start, end: c.end, label: voicings.get(c.label)?.shape ?? c.label })),
    [chords, voicings],
  );
  const chordStrums = useMemo(
    () =>
      chords && transcription
        ? strums(
            chords,
            analysis?.beats ?? transcription.beats,
            voicings,
            tuning,
            analysis?.rhythm ?? null,
            analysis?.bars.map((b) => b.time),
          )
        : undefined,
    [chords, transcription, analysis, voicings, tuning],
  );
  const relabel = useCallback(
    (index: number, label: string) =>
      setChords((current) => current && current.map((c, i) => (i === index ? { ...c, label } : c))),
    [],
  );

  const { setMuted } = playback;
  useEffect(() => setMuted(listen === "notes"), [listen, setMuted]);
  useNotePlayer({
    audio: playback.audio,
    playing: playback.playing,
    notes,
    synth: listen !== "audio",
    clicks,
    strums: listen !== "audio" ? chordStrums : undefined,
  });

  const { set: setNotes, undo, redo } = history;

  const report = (message: string) => {
    setToolMessage(message);
    window.setTimeout(() => setToolMessage((current) => (current === message ? null : current)), 4000);
  };
  // Tidy-up for sung (or loosely played) melodies; they work on the selection, or on
  // every note when nothing is selected, and are undone like any edit.
  const quantize = () => {
    if (!analysis) return;
    const next = quantizeNotes(notes, selection, analysis.beats, division);
    if (next !== notes) setNotes(next);
    report(next === notes ? "Notalar zaten ızgarada" : `Ritim 1/${division * 4} ızgaraya oturtuldu`);
  };
  const snapScale = () => {
    if (!musicKey) return;
    const sung = pitchCurve ? (n: Note) => sungPitch(pitchCurve, n.start, n.end) : undefined;
    const result = snapToScale(notes, selection, musicKey, tuning, sung);
    if (result.moved) setNotes(result.notes);
    report(result.moved ? `${result.moved} nota ${musicKey.name} gamına taşındı` : `${musicKey.name} gamı dışında nota yok`);
  };
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
          beat_grid: beatGrid,
          ...(chords ? { chords } : {}),
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
      setPitchCurve(null);
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
          {project?.source === "song" && <span className="badge">Şarkı</span>}
          {project?.separate_guitar && project.source !== "song" && (
            <span className="badge">{project.source === "voice" ? "Vokal ayrıldı" : "Gitar ayrıldı"}</span>
          )}
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
            ...(project?.source === "song"
              ? [{ label: "Akor şeması indir", href: chordSheetUrl(id), title: "Ölçü ölçü akorlar ve parmak yerleri (metin)" }]
              : []),
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
            allowAuto={source === "song"}
            disabled={busy}
            onChange={(nextTuning, nextCapo) => {
              setTuningName(nextTuning);
              setCapo(nextCapo);
            }}
          />
          {source !== "song" && (
            <label className="row compact small" title="Başka enstrümanlar da çalan kayıtlar için">
              <input type="checkbox" checked={songMode} onChange={(e) => setSongMode(e.target.checked)} />
              {source === "voice" ? "Vokali ayır" : "Gitarı ayır"}
            </label>
          )}
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
      {project && (project.tuning_name !== tuningName || (project.capo_auto ? -1 : project.capo) !== capo) && (
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
            çözüldü. Güncel model: {modelLabel(currentVersion(project.source, models))}.
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
          {chords && (
            <ChordChart
              chords={chords}
              bars={bars}
              keys={keyChoice ? [] : analysis?.keys ?? transcription.keys ?? []}
              voicings={voicings}
              capo={transcription.capo}
              currentTime={playback.currentTime}
              onSeek={playback.seek}
              onRelabel={relabel}
            />
          )}
          <div className="editor-layout">
            <div className="stack">
              {analysis?.rhythm && <RhythmPattern rhythm={analysis.rhythm} />}
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
                  chords={tabChords ?? analysis?.chords}
                  chordsStale={!tabChords && analysisPending}
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
                  pitchCurve={pitchCurve}
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
              title={
                tracked && beatGrid
                  ? "Kayıttaki vuruşların ortanca temposu. Değiştirirseniz ızgara bu sabit tempoya geçer."
                  : "Tahmini tempo. Yanlışsa düzeltin: vuruş ızgarası, metronom, MIDI ve MusicXML ritmi bu tempoya göre."
              }
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
                  editNotation(() => {
                    setTempo(value);
                    setBeatGrid(false); // a typed tempo is one tempo for the whole recording
                  });
                }}
              />
            </label>
            {tracked && (
              <label
                className="row compact"
                title="Vuruş takibi: ızgara, metronom ve dışa aktarma kayıttaki vuruşları izler (metronomsuz, hızlanıp yavaşlayan çalışlarda). Sabit tempo: tek bir BPM."
              >
                Izgara:
                <select
                  value={beatGrid ? "tracked" : "fixed"}
                  onChange={(e) => editNotation(() => setBeatGrid(e.target.value === "tracked"))}
                >
                  <option value="tracked">Vuruş takibi</option>
                  <option value="fixed">Sabit tempo</option>
                </select>
              </label>
            )}
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
            <span
              className="row compact"
              title="Seçili notaların (seçim yoksa hepsinin) başını ve sonunu vuruş ızgarasına oturtur"
            >
              <button onClick={quantize} disabled={!analysis || !notes.length}>
                Ritmi oturt
              </button>
              <select value={division} onChange={(e) => setDivision(Number(e.target.value))} aria-label="Izgara">
                <option value={2}>1/8</option>
                <option value={4}>1/16</option>
              </select>
            </span>
            <button
              onClick={snapScale}
              disabled={!musicKey || !notes.length}
              title="Tonun gamı dışında kalan seçili notaları (seçim yoksa hepsini) yarım ton yandaki gam notasına taşır; ses projelerinde söylenen perdeye göre yön seçilir"
            >
              Gama oturt
            </button>
            {toolMessage && <span className="muted small">{toolMessage}</span>}
            {analysisError && <span className="error small">Analiz yapılamadı: {analysisError}</span>}
          </div>
        </>
      )}
    </div>
  );
}
