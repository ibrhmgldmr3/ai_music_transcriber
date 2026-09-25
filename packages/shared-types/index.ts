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
  /** Quarter-note beats per bar (2-7). */
  beats_per_measure: number;
  /** Key chosen by the user ("A minor"); null = estimated from the notes. */
  key: string | null;
  /** A bar line chosen by the user (seconds); null = estimated from the notes. */
  downbeat: number | null;
}

/** Omitted fields keep their stored value; key/downbeat set to null go back to the estimate. */
export interface NotesUpdate {
  notes: Note[];
  /** Corrected tempo in BPM (20-400); exports quantize rhythm to it. */
  tempo?: number | null;
  beats_per_measure?: number;
  key?: string | null;
  downbeat?: number | null;
}

export interface AnalysisRequest {
  notes: Note[];
  tempo: number | null;
  beats_per_measure: number;
  key: string | null;
  downbeat: number | null;
}

export interface MusicKey {
  /** "Bb major" */
  name: string;
  /** Pitch class of the tonic, 0 = C. */
  tonic: number;
  mode: "major" | "minor";
  /** Key signature: sharps > 0, flats < 0. */
  fifths: number;
}

export interface ChordSymbol {
  start: number;
  end: number;
  /** "F#m7", "D/F#" */
  label: string;
}

/** Key, bar grid and chord symbols of a set of notes (POST /api/analysis). */
export interface Analysis {
  tempo: number;
  beats_per_measure: number;
  /** A bar line in [0, bar length) seconds; bar lines repeat every bar. */
  downbeat: number;
  /** The chosen key, else the estimate. */
  key: MusicKey | null;
  estimated_key: MusicKey | null;
  chords: ChordSymbol[];
}
