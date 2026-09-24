"use client";

import { useRouter } from "next/navigation";
import { useRef, useState, type DragEvent, type FormEvent, type KeyboardEvent } from "react";
import { errorMessage, uploadAudio } from "@/lib/api";

const ACCEPTED = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"];
const MAX_MB = 50;

function extension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

export default function AudioUploader() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [separateGuitar, setSeparateGuitar] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const uploading = progress !== null;

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
        { name: name.trim() || undefined, separateGuitar },
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
            <strong>Gitar kaydını buraya sürükleyin</strong>
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

      <label className="field">
        Proje adı
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="My Guitar Recording" />
      </label>

      <label className="row compact">
        <input
          type="checkbox"
          checked={separateGuitar}
          onChange={(e) => setSeparateGuitar(e.target.checked)}
        />
        Şarkı / grup kaydı: önce gitarı diğer enstrümanlardan ayır
      </label>
      <p className="muted small">
        Kayıtta davul, bas veya vokal varsa açın. Yalnız gitar kayıtlarında gerekmez; çözümleme biraz uzar.
      </p>

      {uploading && (
        <div className="progress" aria-label="Yükleme ilerlemesi">
          <div style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
        </div>
      )}
      {error && <p className="error">{error}</p>}

      <div>
        <button type="submit" className="primary" disabled={!file || uploading}>
          {uploading ? `Yükleniyor… %${Math.round((progress ?? 0) * 100)}` : "Yükle ve çözümle"}
        </button>
      </div>
    </form>
  );
}
