// Keep in sync with apps/api/app/schemas/project.py.

export type ProjectStatus = "pending" | "processing" | "completed" | "failed";

/** What was recorded: a guitar, or a voice (sung, hummed or whistled melody) set for guitar. */
export type Source = "guitar" | "voice";

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
  source: Source;
  /** Song mode: the guitar (voice: the vocals) is isolated from a band mix before transcription. */
  separate_guitar: boolean;
  /** The guitar's tuning (a key of TUNINGS in apps/web/lib/music.ts) and capo fret. */
  tuning_name: string;
  capo: number;
  /** Models that transcribed it (compare with ModelInfo.version, or voice_version for
   * voice projects); null if unknown. */
  model_version: string | null;
  /** The user saved edits since the transcription; re-transcribing discards them. */
  edited: boolean;
  /** While transcribing: done fraction (0-1) and stage; null otherwise. */
  progress: number | null;
  stage: "loading" | "separating" | "transcribing" | "finishing" | null;
  error: string | null;
  duration: number | null;
  created_at: string;
  updated_at: string;
}

/** The models new transcriptions use (GET /api/models). */
export interface ModelInfo {
  version: string;
  /** Version of the voice method (voice projects). */
  voice_version: string;
  /** Checkpoint folders, e.g. "guitar_v8". */
  notes_model: string;
  tab_model: string | null;
}

export interface Transcription {
  project_id: string;
  tempo: number | null;
  /** Open-string MIDI pitches the frets count from (capo included), lowest string first. */
  tuning: number[];
  tuning_name: string;
  capo: number;
  /** Semitones the notes were moved from the recording (voice mode fits the melody
   * into the guitar's range by octaves). */
  transpose: number;
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

/** Notes and notation to engrave as MusicXML (POST /api/render/musicxml). */
export interface RenderRequest extends AnalysisRequest {
  /** Open strings the frets count from (capo included). */
  tuning: number[];
  capo: number;
  title: string;
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
