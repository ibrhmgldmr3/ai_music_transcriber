// Keep in sync with apps/api/app/schemas/project.py.

export type ProjectStatus = "pending" | "processing" | "completed" | "failed";

/**
 * What was recorded: a guitar, a voice (sung, hummed or whistled melody) or a whole song
 * (its chords and sung melody); the last two are set for guitar.
 */
export type Source = "guitar" | "voice" | "song";

/** A chord of a song project; `label` in Harte syntax ("A:min7", "F#", "D:maj/3" = D/F#). */
export interface SongChord {
  start: number;
  end: number;
  label: string;
}

/** POST /api/chords/voicings: chord labels to name and finger. */
export interface VoicingRequest {
  labels: string[];
  capo: number;
  tuning_name: string;
  key: string | null;
}

export interface Voicing {
  label: string;
  /** The chord as it sounds, e.g. "Cm". */
  name: string;
  /** What to play from the capo, e.g. "Am" with a capo at 3. */
  shape: string;
  /** Per string, lowest first: -1 muted, 0 open; null: no playable shape. */
  frets: number[] | null;
  difficulty: number | null;
}

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
  /** Song projects: the transcription chooses the capo that makes the chords easiest. */
  capo_auto: boolean;
  /** Models that transcribed it (compare with ModelInfo.version, or voice_version /
   * song_version for voice / song projects); null if unknown. */
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
  /** Version of the song method, including the voice method for its melody (song projects). */
  song_version: string;
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
  capo_auto: boolean;
  /** Song projects: the chords; null for guitar and voice projects. */
  chords: SongChord[] | null;
  /** Song projects that modulate: their keys over time; null otherwise. */
  keys: KeySpan[] | null;
  /** Beats and bar lines (seconds) tracked in the recording (guitar and song projects). */
  beats: number[];
  downbeats: number[];
  /** The bar grid follows the tracked beats (else the one tempo above). */
  beat_grid: boolean;
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

/** The sung pitch of a voice project (GET /api/projects/{id}/pitch), on the notes' scale. */
export interface PitchCurve {
  /** Values per second, the first at 0 s. */
  frame_rate: number;
  /** MIDI pitch (fractional); null where nothing is sung. */
  values: (number | null)[];
}

/** Omitted fields keep their stored value; key/downbeat set to null go back to the estimate. */
export interface NotesUpdate {
  notes: Note[];
  /** Corrected tempo in BPM (20-400); exports quantize rhythm to it. */
  tempo?: number | null;
  beats_per_measure?: number;
  key?: string | null;
  downbeat?: number | null;
  /** Song projects: the edited chords. */
  chords?: SongChord[];
  /** Whether the bar grid follows the tracked beats. */
  beat_grid?: boolean;
}

export interface AnalysisRequest {
  notes: Note[];
  tempo: number | null;
  beats_per_measure: number;
  key: string | null;
  downbeat: number | null;
  /** The notes' project: its tracked beats (with `beat_grid`), recognized chords, keys and strums join in. */
  project_id?: string;
  beat_grid?: boolean;
  /** Seconds the grid should cover (the recording). */
  duration?: number;
}

/** A song's key over a stretch of time. */
export interface KeySpan {
  start: number;
  end: number;
  /** "A minor" */
  key: string;
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

/** The strumming pattern: which eighths or sixteenths of a bar are struck. */
export interface Rhythm {
  /** 2: eighths, 4: sixteenths. */
  per_beat: number;
  /** The song's bar: struck slots. */
  pattern: boolean[];
  /** "D-DU-UDU": D down, U up (the pendulum rule), - not struck. */
  text: string;
  /** Every bar's struck slots, starting at `bar_times`. */
  bars: boolean[][];
  bar_times: number[];
}

/** Key, bar grid, chord symbols and strumming pattern of a set of notes (POST /api/analysis). */
export interface Analysis {
  /** BPM; with tracked beats, of the median beat. */
  tempo: number;
  beats_per_measure: number;
  /** The first bar line at or after 0 s. */
  downbeat: number;
  /** The chosen key, else the estimate (a modulating song's main key). */
  key: MusicKey | null;
  estimated_key: MusicKey | null;
  chords: ChordSymbol[];
  /** The grid follows the recording's tracked beats rather than one tempo. */
  tracked: boolean;
  /** Every beat (bar lines included), seconds. */
  beats: number[];
  /** Bar lines, numbered like the MusicXML measures. */
  bars: { time: number; number: number }[];
  /** A song's keys over time. */
  keys: KeySpan[];
  rhythm: Rhythm | null;
}
