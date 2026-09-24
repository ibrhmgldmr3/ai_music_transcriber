// Keep in sync with apps/api/app/schemas/project.py.

export type ProjectStatus = "pending" | "processing" | "completed" | "failed";

export interface Note {
  /** MIDI pitch (0-127). */
  pitch: number;
  /** Onset in seconds. */
  start: number;
  /** Offset in seconds. */
  end: number;
  /** MIDI velocity (1-127). */
  velocity: number;
  /** Guitar string, 0 = lowest string (low E in standard tuning). */
  string: number | null;
  /** Fret number, 0 = open string. */
  fret: number | null;
  /** Model confidence in [0, 1]; null for hand-entered notes. */
  confidence: number | null;
}

export interface Project {
  id: string;
  name: string;
  filename: string;
  status: ProjectStatus;
  /** Song mode: the guitar is isolated from a band mix before transcription. */
  separate_guitar: boolean;
  error: string | null;
  duration: number | null;
  created_at: string;
  updated_at: string;
}

export interface Transcription {
  project_id: string;
  tempo: number | null;
  /** Open-string MIDI pitches, lowest string first. */
  tuning: number[];
  /** Mean note confidence (there is no ground truth for user uploads). */
  mean_confidence: number | null;
  notes: Note[];
}

export interface NotesUpdate {
  notes: Note[];
  /** Corrected tempo in BPM (20-400); exports quantize rhythm to it. */
  tempo?: number | null;
}
