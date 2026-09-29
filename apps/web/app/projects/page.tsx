"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import type { ModelInfo, Project } from "@music-transcriber/shared-types";
import {
  STATUS_LABELS,
  deleteProject,
  errorMessage,
  getModelInfo,
  isOutdated,
  listProjects,
  modelLabel,
  retranscribe,
} from "@/lib/api";
import { formatTime } from "@/lib/music";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [models, setModels] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setProjects(await listProjects());
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    getModelInfo()
      .then(setModels)
      .catch(() => setModels(null)); // the list works without it
  }, [load]);

  // Refresh while any transcription is still running.
  const busy = projects.some((p) => p.status === "pending" || p.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => void load(), 1500);
    return () => clearInterval(timer);
  }, [busy, load]);

  const outdated = projects.filter((p) => isOutdated(p, models));
  // Saved edits would be lost, so those projects are only redone one by one, from the editor.
  const redoable = outdated.filter((p) => !p.edited);

  const remove = async (project: Project) => {
    if (!confirm(`"${project.name}" silinsin mi? Bu işlem geri alınamaz.`)) return;
    try {
      await deleteProject(project.id);
      await load();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const redoOutdated = async () => {
    if (!models || redoable.length === 0) return;
    const skipped = outdated.length - redoable.length;
    const message =
      `${redoable.length} proje ${modelLabel(models.version)} ile yeniden çözümlensin mi?` +
      (skipped ? `\nDüzenlenmiş ${skipped} proje atlanacak; onları editörden tek tek yenileyebilirsiniz.` : "");
    if (!confirm(message)) return;
    setUpdating(true);
    try {
      for (const project of redoable) await retranscribe(project.id);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setUpdating(false);
      await load();
    }
  };

  return (
    <div className="stack">
      <div className="row between">
        <h1>Projeler</h1>
        <div className="row">
          {redoable.length > 0 && (
            <button
              onClick={() => void redoOutdated()}
              disabled={updating}
              title="Eski modelle çözülmüş ve düzenlenmemiş projeleri güncel modelle yeniden çözümler"
            >
              {updating ? "Başlatılıyor…" : `Eski modelle çözülenleri yenile (${redoable.length})`}
            </button>
          )}
          <Link href="/upload" className="button primary">
            Yeni kayıt
          </Link>
        </div>
      </div>
      {models && <p className="muted small">Kullanılan model: {modelLabel(models.version)}</p>}
      {error && <p className="error">{error}</p>}
      {loading ? (
        <p className="muted">Yükleniyor…</p>
      ) : projects.length === 0 ? (
        <div className="card">
          <p className="muted">
            Henüz proje yok. <Link href="/upload">İlk kaydınızı yükleyin.</Link>
          </p>
        </div>
      ) : (
        <div className="card">
          <table>
            <thead>
              <tr>
                <th>Ad</th>
                <th>Durum</th>
                <th>Süre</th>
                <th>Oluşturulma</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {projects.map((p) => (
                <tr key={p.id}>
                  <td>
                    <Link href={`/editor?id=${p.id}`}>{p.name}</Link>
                    <div className="muted small">{p.filename}</div>
                  </td>
                  <td>
                    <div className="row compact">
                      <span className={`badge ${p.status}`} title={p.error ?? undefined}>
                        {STATUS_LABELS[p.status]}
                        {p.status === "processing" && p.progress != null && ` %${Math.round(p.progress * 100)}`}
                      </span>
                      {isOutdated(p, models) && (
                        <span
                          className="badge outdated"
                          title={
                            p.model_version
                              ? `${modelLabel(p.model_version)} ile çözüldü`
                              : "Hangi modelle çözüldüğü bilinmiyor"
                          }
                        >
                          Eski model
                        </span>
                      )}
                      {p.source === "voice" && (
                        <span className="badge" title="Söylenen / mırıldanan melodiden">
                          Ses
                        </span>
                      )}
                      {p.edited && <span className="badge">Düzenlendi</span>}
                    </div>
                  </td>
                  <td>{p.duration !== null ? formatTime(p.duration) : "—"}</td>
                  <td>{new Date(p.created_at).toLocaleString("tr-TR")}</td>
                  <td style={{ textAlign: "right" }}>
                    <button className="danger" onClick={() => void remove(p)}>
                      Sil
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
