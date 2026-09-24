import type {
  Note,
  NotesUpdate,
  Project,
  ProjectStatus,
  Transcription,
} from "@music-transcriber/shared-types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export const STATUS_LABELS: Record<ProjectStatus, string> = {
  pending: "Sırada",
  processing: "İşleniyor",
  completed: "Tamamlandı",
  failed: "Başarısız",
};

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function extractDetail(body: unknown): string | null {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  return null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api${path}`, init);
  } catch {
    throw new ApiError(0, `API'ye ulaşılamadı (${API_URL}).`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, extractDetail(body) ?? res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// Project ids come from the URL (?id=...), so they are encoded before use in a path.
const projectPath = (id: string) => `/projects/${encodeURIComponent(id)}`;

export const audioUrl = (id: string) => `${API_URL}/api${projectPath(id)}/audio`;
export const midiUrl = (id: string) => `${API_URL}/api${projectPath(id)}/midi`;
export const tabUrl = (id: string) => `${API_URL}/api${projectPath(id)}/tab`;
export const musicXmlUrl = (id: string) => `${API_URL}/api${projectPath(id)}/musicxml`;

export const listProjects = () => request<Project[]>("/projects");
export const getProject = (id: string) => request<Project>(projectPath(id));
export const deleteProject = (id: string) => request<void>(projectPath(id), { method: "DELETE" });
export const retranscribe = (id: string, separateGuitar?: boolean) => {
  const query = separateGuitar === undefined ? "" : `?separate_guitar=${separateGuitar}`;
  return request<Project>(`${projectPath(id)}/retranscribe${query}`, { method: "POST" });
};
export const getTranscription = (id: string) =>
  request<Transcription>(`${projectPath(id)}/transcription`);
export const saveNotes = (id: string, notes: Note[], tempo?: number | null) => {
  const body: NotesUpdate = { notes, tempo: tempo ?? null };
  return request<Transcription>(`${projectPath(id)}/notes`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
};

/** Upload with progress reporting (fetch cannot report upload progress). */
export function uploadAudio(
  file: File,
  options: { name?: string; separateGuitar?: boolean } = {},
  onProgress?: (fraction: number) => void,
): Promise<Project> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    if (options.name) form.append("name", options.name);
    form.append("separate_guitar", String(Boolean(options.separateGuitar)));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/projects`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as Project);
      else reject(new ApiError(xhr.status, extractDetail(xhr.response) ?? xhr.statusText));
    };
    xhr.onerror = () => reject(new ApiError(0, `API'ye ulaşılamadı (${API_URL}).`));
    xhr.send(form);
  });
}
