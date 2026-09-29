"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, type DragEvent, type FormEvent, type KeyboardEvent } from "react";
import type { Source } from "@music-transcriber/shared-types";
import TuningPicker from "@/components/TuningPicker";
import { SOURCE_LABELS, errorMessage, uploadAudio } from "@/lib/api";
import { formatTime } from "@/lib/music";
import { MicRecorder } from "@/lib/recorder";

const ACCEPTED = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"];
const MAX_MB = 50;
// A 16-bit mono WAV at 48 kHz fills MAX_MB in about 9 minutes.
const MAX_RECORDING_SECONDS = 8 * 60;

function extension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

export default function AudioUploader() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [source, setSource] = useState<Source>("guitar");
  const voice = source === "voice";
  const [separateGuitar, setSeparateGuitar] = useState(false);
  const [tuning, setTuning] = useState("standard");
  const [capo, setCapo] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const uploading = progress !== null;
  const recorderRef = useRef<MicRecorder | null>(null);
  const [recording, setRecording] = useState(false);
  const [recorded, setRecorded] = useState(0);
  const [level, setLevel] = useState(0);

  const stopRecording = async () => {
    const recorder = recorderRef.current;
    if (!recorder) return;
    recorderRef.current = null;
    setRecording(false);
    const stamp = new Date().toISOString().slice(0, 16).replace(/[-:]/g, "").replace("T", "-");
    pick(await recorder.stop(`kayit-${stamp}.wav`));
  };

  const startRecording = async () => {
    setError(null);
    try {
      recorderRef.current = await MicRecorder.start((rms) => setLevel(rms));
      setRecording(true);
      setRecorded(0);
    } catch (err) {
      const denied = err instanceof DOMException && err.name === "NotAllowedError";
      setError(denied ? "Mikrofon izni verilmedi." : `Mikrofon açılamadı: ${errorMessage(err)}`);
    }
  };

  // Timer, and an automatic stop before the recording outgrows the upload limit.
  useEffect(() => {
    if (!recording) return;
    const timer = setInterval(() => {
      const seconds = recorderRef.current?.seconds ?? 0;
      setRecorded(seconds);
      if (seconds >= MAX_RECORDING_SECONDS) void stopRecording();
    }, 250);
    return () => clearInterval(timer);
    // stopRecording reads the recorder from its ref, so it needn't be a dependency.
  }, [recording]);

  // Release the microphone when leaving the page mid-recording.
  useEffect(() => () => void recorderRef.current?.stop("kayit.wav"), []);

  const pick = (candidate: File | undefined) => {
    if (!candidate) return;
    const ext = extension(candidate.name);
    if (!ACCEPTED.includes(ext)) {
      setError(`Desteklenmeyen dosya türü: ${ext || "?"} (${ACCEPTED.join(", ")})`);
      return;
    }
    if (candidate.size > MAX_MB * 1024 * 1024) {
      setError(`Dosya ${MAX_MB} MB sınırını aşıyor.`);
      return;
    }
    setError(null);
    setFile(candidate);
    if (!name) setName(candidate.name.replace(/\.[^.]+$/, ""));
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files[0]);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const project = await uploadAudio(
        file,
        { name: name.trim() || undefined, source, separateGuitar, tuning, capo },
        setProgress,
      );
      router.push(`/editor?id=${project.id}`);
    } catch (err) {
      setError(errorMessage(err));
      setProgress(null);
    }
  };

  return (
    <form className="card stack" onSubmit={onSubmit}>
      <fieldset className="row compact source-picker">
        <legend className="small">Ne kaydettiniz?</legend>
        {(Object.keys(SOURCE_LABELS) as Source[]).map((key) => (
          <label key={key} className="row compact">
            <input
              type="radio"
              name="source"
              value={key}
              checked={source === key}
              onChange={() => setSource(key)}
            />
            {SOURCE_LABELS[key]}
          </label>
        ))}
      </fieldset>
      {voice && (
        <p className="muted small">
          Tek bir melodiyi söyleyin, mırıldanın ya da ıslıkla çalın: notalar, seçtiğiniz akortta gitar TAB&apos;ına
          yerleştirilir. Editörde gitar sesiyle dinleyip &quot;Gitar sesi indir&quot; ile WAV olarak alabilirsiniz.
        </p>
      )}
      <div
        className={`dropzone${dragging ? " active" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={onKeyDown}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {file ? (
          <>
            <strong>{file.name}</strong>
            <span className="muted">{(file.size / 1024 / 1024).toFixed(1)} MB · değiştirmek için tıklayın</span>
          </>
        ) : (
          <>
            <strong>{voice ? "Ses kaydını" : "Gitar kaydını"} buraya sürükleyin</strong>
            <span className="muted">veya seçmek için tıklayın · {ACCEPTED.join(" ")} · en fazla {MAX_MB} MB</span>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={["audio/*", ...ACCEPTED].join(",")}
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
      </div>

      <div className="row">
        {recording ? (
          <>
            <button type="button" className="danger" onClick={() => void stopRecording()}>
              ■ Kaydı durdur
            </button>
            <span className="time">{formatTime(recorded)}</span>
            <div className="level-meter" aria-label="Giriş seviyesi">
              <div style={{ width: `${Math.min(100, Math.round(level * 300))}%` }} />
            </div>
          </>
        ) : (
          <button type="button" onClick={() => void startRecording()} disabled={uploading}>
            🎙 Mikrofonla kaydet
          </button>
        )}
        <span className="muted small">
          En fazla {MAX_RECORDING_SECONDS / 60} dakika.{" "}
          {voice
            ? "Sessiz bir odada, notaları net ve ayrık söyleyin; seviye çubuğu sonuna vurmasın."
            : "Gitarı mikrofona yakın tutun, seviye çubuğu sonuna vurmasın."}
        </span>
      </div>

      <label className="field">
        Proje adı
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="My Guitar Recording" />
      </label>

      <div className="row">
        <TuningPicker
          tuning={tuning}
          capo={capo}
          onChange={(nextTuning, nextCapo) => {
            setTuning(nextTuning);
            setCapo(nextCapo);
          }}
        />
      </div>

      <label className="row compact">
        <input
          type="checkbox"
          checked={separateGuitar}
          onChange={(e) => setSeparateGuitar(e.target.checked)}
        />
        {voice
          ? "Müzik eşliğinde söylenmiş: önce vokali diğer enstrümanlardan ayır"
          : "Şarkı / grup kaydı: önce gitarı diğer enstrümanlardan ayır"}
      </label>
      <p className="muted small">
        {voice
          ? "Arkada müzik çalıyorsa açın. Ses tek başınaysa gerekmez; çözümleme biraz uzar."
          : "Kayıtta davul, bas veya vokal varsa açın. Yalnız gitar kayıtlarında gerekmez; çözümleme biraz uzar."}
      </p>

      {uploading && (
        <div className="progress" aria-label="Yükleme ilerlemesi">
          <div style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
        </div>
      )}
      {error && <p className="error">{error}</p>}

      <div>
        <button type="submit" className="primary" disabled={!file || uploading || recording}>
          {uploading ? `Yükleniyor… %${Math.round((progress ?? 0) * 100)}` : "Yükle ve çözümle"}
        </button>
      </div>
    </form>
  );
}
