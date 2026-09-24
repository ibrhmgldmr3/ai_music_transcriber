"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import type { Project } from "@music-transcriber/shared-types";
import { STATUS_LABELS, deleteProject, errorMessage, listProjects } from "@/lib/api";
import { formatTime } from "@/lib/music";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
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
  }, [load]);

  // Refresh while any transcription is still running.
  const busy = projects.some((p) => p.status === "pending" || p.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => void load(), 3000);
    return () => clearInterval(timer);
  }, [busy, load]);

  const remove = async (project: Project) => {
    if (!confirm(`"${project.name}" silinsin mi? Bu işlem geri alınamaz.`)) return;
    try {
      await deleteProject(project.id);
      await load();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  return (
    <div className="stack">
      <div className="row between">
        <h1>Projeler</h1>
        <Link href="/upload" className="button primary">
          Yeni kayıt
        </Link>
      </div>
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
                    <span className={`badge ${p.status}`} title={p.error ?? undefined}>
                      {STATUS_LABELS[p.status]}
                    </span>
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
