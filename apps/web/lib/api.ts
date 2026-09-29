import type {
  Analysis,
  AnalysisRequest,
  ModelInfo,
  Note,
  NotesUpdate,
  Project,
  ProjectStatus,
  RenderRequest,
  Source,
  Transcription,
} from "@music-transcriber/shared-types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export const SOURCE_LABELS: Record<Source, string> = {
  guitar: "Gitar",
  voice: "Ses (şarkı, mırıldanma, ıslık)",
};

export const STATUS_LABELS: Record<ProjectStatus, string> = {
  pending: "Sırada",
  processing: "İşleniyor",
  completed: "Tamamlandı",
  failed: "Başarısız",
};

export const STAGE_LABELS: Record<NonNullable<Project["stage"]>, string> = {
  loading: "Ses yükleniyor",
  separating: "Gitar diğer enstrümanlardan ayrılıyor",
  transcribing: "Notalar çözümleniyor",
  finishing: "Tel/perde ve tempo hesaplanıyor",
};

/** What a queued or running transcription is doing. */
export function stageLabel(project: Project): string {
  if (!project.stage) return "Sırada bekliyor";
  if (project.stage === "separating" && project.source === "voice") return "Vokal diğer enstrümanlardan ayrılıyor";
  return STAGE_LABELS[project.stage];
}

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
  } catch (err) {
    if (init?.signal?.aborted) throw err;
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
export const getModelInfo = () => request<ModelInfo>("/models");

/** "guitar_v8@1a2b3c4d+guitar_v7@5e6f7a8b" -> "guitar_v8 + guitar_v7". */
export const modelLabel = (version: string) =>
  version
    .split("+")
    .map((part) => part.split("@")[0])
    .join(" + ");

/** A finished transcription made by other models (or voice method) than the current ones. */
export const isOutdated = (project: Project, models: ModelInfo | null) =>
  models !== null &&
  project.status === "completed" &&
  project.model_version !== (project.source === "voice" ? models.voice_version : models.version);
export const getProject = (id: string) => request<Project>(projectPath(id));
export const deleteProject = (id: string) => request<void>(projectPath(id), { method: "DELETE" });
/** Transcribe again; the options, when given, change the project's settings first. */
export const retranscribe = (
  id: string,
  options: { source?: Source; separateGuitar?: boolean; tuning?: string; capo?: number } = {},
) => {
  const query = new URLSearchParams();
  if (options.source !== undefined) query.set("source", options.source);
  if (options.separateGuitar !== undefined) query.set("separate_guitar", String(options.separateGuitar));
  if (options.tuning !== undefined) query.set("tuning", options.tuning);
  if (options.capo !== undefined) query.set("capo", String(options.capo));
  const suffix = query.toString() ? `?${query}` : "";
  return request<Project>(`${projectPath(id)}/retranscribe${suffix}`, { method: "POST" });
};
export const getTranscription = (id: string) =>
  request<Transcription>(`${projectPath(id)}/transcription`);
export const saveNotes = (id: string, notes: Note[], notation: Omit<NotesUpdate, "notes"> = {}) => {
  const body: NotesUpdate = { notes, ...notation };
  return request<Transcription>(`${projectPath(id)}/notes`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
};
/** MusicXML of the given (possibly unsaved) notes. */
export async function renderMusicXml(body: RenderRequest, signal?: AbortSignal): Promise<string> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/render/musicxml`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new ApiError(0, `API'ye ulaşılamadı (${API_URL}).`);
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(res.status, extractDetail(detail) ?? res.statusText);
  }
  return res.text();
}

export const analyzeNotes = (body: AnalysisRequest, signal?: AbortSignal) =>
  request<Analysis>("/analysis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

/** Upload with progress reporting (fetch cannot report upload progress). */
export function uploadAudio(
  file: File,
  options: { name?: string; source?: Source; separateGuitar?: boolean; tuning?: string; capo?: number } = {},
  onProgress?: (fraction: number) => void,
): Promise<Project> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    if (options.name) form.append("name", options.name);
    if (options.source) form.append("source", options.source);
    form.append("separate_guitar", String(Boolean(options.separateGuitar)));
    if (options.tuning) form.append("tuning", options.tuning);
    if (options.capo) form.append("capo", String(options.capo));

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
