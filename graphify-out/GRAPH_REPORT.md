# Graph Report - ai_music_transcriber  (2026-10-25)

## Corpus Check
- 162 files · ~103,329 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: (none) 8, .example 1, .css 1)

## Summary
- 2183 nodes · 5337 edges · 121 communities (102 shown, 19 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 376 edges (avg confidence: 0.93)
- Token cost: 149,048 input · 0 output

## Community Hubs (Navigation)
- Voice Pitch Tracking
- Evaluation Metrics
- Editor & Projects Pages
- Audio Augmentation
- Analysis API Endpoint
- Dataset Loading (GuitarSet)
- Dataset Download Scripts
- Training Callbacks
- Guitar Chord Shapes & Capo
- Tab & Piano Roll Views
- Pitch Range & Track Splitting
- Project REST Routes
- BTC Chord Transformer
- Note Editor & Music Utils
- Predictor & EGDB Alignment
- Key & Chord Analysis
- MIDI Export
- MusicXML Quantization
- EGDB Dataset
- CNN/CRNN Models & Heads
- Bend & Vibrato Detection
- Note Model & Tab Assignment
- API App & Security Setup
- Web Package Dependencies
- Technique Pitch Curves
- API Security Tests
- Web Synth & Techniques
- Upload Storage & Validation
- Transcription Service
- CI Workflow
- Chord Chart Components
- GPU Docker Compose
- ML YAML Config Loader
- Song Mode Chords
- Tab Optimizer Search
- Database & Project Model
- Web App Shell & Upload
- Song Melody & Tempo
- MusicXML Measures & Tests
- Web Note Editing Ops
- Config Overrides & Player Splits
- GuitarSet Preprocessing
- Strum Detection
- Predictor Transcribe Pipeline
- Visualization Helpers
- Transcription Dataset
- Musical Key & Spelling
- Timing & Quantization
- Technique Benchmark
- API Test Fixtures
- Guitar YAML Config
- AAM Dataset
- Guitar-TECHS Dataset
- Beat Grid
- Shared TS Types
- Web TS Config
- Beat Tracker & Chord Bench
- Chord Voicings & Render API
- Strumming Rhythm
- Augmentation Variants Config
- Model Meta Endpoint
- ASGI Security Middleware
- Training Config
- CNN Building Blocks
- MIDI Import & Tests
- Background Job Progress
- Beat-Level Chord Fusion
- Analysis Tests
- API Requirements
- Base Audio/Paths Config
- IDMT Technique Dataset
- README Song Features & API
- Fake Predictor Tests
- Chord Recognizer Inference
- README Chord Pipeline
- Web Settings
- Model Config
- Mixed Guitar Config
- Mixed Tab Config
- Chord Inference Module
- README Model & Pitch Trackers
- README EGDB & Tab Head
- API Settings
- Compose Service Deps
- Compose API Environment
- Feature Config
- Source Separation
- Compose Env Anchor
- Augmentation Config
- Compose Volumes
- Compose Web Build
- Compose Postgres
- Inference Config
- Shared Types Package
- Compose Healthcheck
- Compose Redis Service
- Mixed Low Config
- Tab Head Config
- README Datasets & Splits
- Compose Web Service
- Label Config
- Loss Weights Config
- Room Config
- Room+EGDB Config
- Transcription Loss
- Song API Tests
- String MIDI Tests
- Web Fixture Parity Test
- Next.js Config
- Demucs-Missing Test
- Pyproject Root

## God Nodes (most connected - your core abstractions)
1. `Note` - 173 edges
2. `Editor()` - 47 edges
3. `Key` - 47 edges
4. `Predictor` - 36 edges
5. `notes_to_musicxml()` - 36 edges
6. `transcribe_song()` - 33 edges
7. `wav_bytes()` - 33 edges
8. `upload()` - 31 edges
9. `ChordSymbol` - 29 edges
10. `load_audio()` - 26 edges

## Surprising Connections (you probably didn't know these)
- `_key_out()` --uses--> `Key`  [INFERRED]
  apps/api/app/api/analysis.py → packages/music-core/analysis.py
- `project_context()` --uses--> `Key`  [INFERRED]
  apps/api/app/api/analysis.py → packages/music-core/analysis.py
- `project_context()` --uses--> `KeySpan`  [INFERRED]
  apps/api/app/api/analysis.py → packages/music-core/analysis.py
- `song_chords()` --uses--> `Chord`  [INFERRED]
  apps/api/app/api/analysis.py → packages/music-core/analysis.py
- `song_chords()` --uses--> `ChordSymbol`  [INFERRED]
  apps/api/app/api/analysis.py → packages/music-core/guitar_chords.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Song -> guitar pipeline (separation, beats, chords, slash chords, melody, keys, capo, rhythm)** — readme_song_mode, readme_demucs, readme_beat_this, readme_btc_chord_recognizer, readme_slash_chords, readme_voice_mode, readme_key_segmentation, readme_capo_suggestion, readme_rhythm_pattern [EXTRACTED 1.00]
- **Datasets and alignment behind default V10 + V7 model pair** — readme_guitarset, readme_egdb, readme_guitar_techs, readme_egdb_alignment, readme_simulated_rooms, readme_v10_model, readme_v7_tab_model [EXTRACTED 1.00]
- **Playing techniques rendered across MIDI, MusicXML, ASCII TAB and synth** — readme_playing_techniques, readme_midi_export, readme_musicxml_export, readme_ascii_tab_export, readme_karplus_strong_synth [EXTRACTED 1.00]

## Communities (121 total, 19 thin omitted)

### Community 0 - "Voice Pitch Tracking"
Cohesion: 0.06
Nodes (39): _absorb_short(), _arrival(), _dip(), estimate_tuning(), fit_to_range(), _merge_equal(), _note(), _pieces() (+31 more)

### Community 1 - "Evaluation Metrics"
Cohesion: 0.08
Nodes (26): evaluate_files(), _prefixed(), _counts(), frame_metrics(), mean_confidence(), note_metrics(), _prf(), tab_metrics() (+18 more)

### Community 2 - "Editor & Projects Pages"
Cohesion: 0.13
Nodes (40): Editor(), EditorPage(), METERS, NO_NOTES, ProjectsPage(), NotationView(), Osmd, analyzeNotes() (+32 more)

### Community 3 - "Audio Augmentation"
Cohesion: 0.09
Nodes (24): add_noise(), add_reverb(), apply_variant(), codec(), distortion(), echo(), _lag(), phone_quality() (+16 more)

### Community 4 - "Analysis API Endpoint"
Cohesion: 0.11
Nodes (22): analysis_out(), analyze_notes(), _context(), _key_out(), project_context(), AnalysisOut, AnalysisRequest, BarOut (+14 more)

### Community 5 - "Dataset Loading (GuitarSet)"
Cohesion: 0.11
Nodes (7): find_tracks(), compute_cqt(), compute_features(), compute_mel(), log_compress(), midi_to_hz(), benchmark()

### Community 6 - "Dataset Download Scripts"
Cohesion: 0.08
Nodes (12): main(), RemoteFile, download(), extract(), main(), _get(), list_folder(), main() (+4 more)

### Community 7 - "Training Callbacks"
Cohesion: 0.09
Nodes (6): Callback, CSVLogger, EarlyStopping, _improved(), ModelCheckpoint, Trainer

### Community 8 - "Guitar Chord Shapes & Capo"
Cohesion: 0.11
Nodes (21): chord_sheet(), shape(), ChordSymbol, _difficulty(), _note(), _searched_voicing(), _standard_voicing(), _string_options() (+13 more)

### Community 9 - "Tab & Piano Roll Views"
Cohesion: 0.11
Nodes (30): GuitarTab(), GuitarTabProps, labelBox(), Marquee, Marquee, PianoRoll(), PianoRollProps, pitchRange() (+22 more)

### Community 10 - "Pitch Range & Track Splitting"
Cohesion: 0.09
Nodes (18): extend_pitch_range(), split_into_parts(), array_to_notes(), load_guitarset_contours(), load_guitarset_jams(), load_string_midi(), notes_to_array(), notes_to_targets() (+10 more)

### Community 11 - "Project REST Routes"
Cohesion: 0.12
Nodes (19): _attachment(), _capo_setting(), _chosen_key(), create_project(), delete_project(), _enqueue(), export_chord_sheet(), export_midi() (+11 more)

### Community 12 - "BTC Chord Transformer"
Cohesion: 0.10
Nodes (11): _bias_mask(), BiDirectionalSelfAttention, BiDirectionalSelfAttentionLayers, BTC, Conv, LayerNorm, MultiHeadAttention, PositionwiseFeedForward (+3 more)

### Community 13 - "Note Editor & Music Utils"
Cohesion: 0.11
Nodes (28): BENDS, NoteEditor(), NoteEditorProps, TuningPickerProps, BLACK_KEYS, CHROMATIC, CIRCLE, clamp() (+20 more)

### Community 14 - "Predictor & EGDB Alignment"
Cohesion: 0.15
Nodes (14): read_split(), main(), main(), Predictor, main(), onset_f1(), main(), main() (+6 more)

### Community 15 - "Key & Chord Analysis"
Cohesion: 0.10
Nodes (17): Analysis, analyze(), beat_phase(), _chord_templates(), _decode_chords(), estimate_key(), KeySpan, _pc() (+9 more)

### Community 16 - "MIDI Export"
Cohesion: 0.12
Nodes (16): main(), _bend(), _bend_range(), _has_technique(), notes_to_midi(), _pitch_curve(), curve(), _pitch_events() (+8 more)

### Community 17 - "MusicXML Quantization"
Cohesion: 0.10
Nodes (18): _decompose(), _Event, first_bar_line(), _Links, to_units(), _one_per_string(), _pieces(), _quantize() (+10 more)

### Community 18 - "EGDB Dataset"
Cohesion: 0.10
Nodes (13): align_notes(), aligned_clips(), augmentable(), EGDBTrack, find_aligned_tracks(), find_tracks(), load_aligned_track(), load_track() (+5 more)

### Community 19 - "CNN/CRNN Models & Heads"
Cohesion: 0.11
Nodes (4): build_heads(), PitchHead, TabHead, resolve_device()

### Community 20 - "Bend & Vibrato Detection"
Cohesion: 0.17
Nodes (18): Bend, detect_techniques(), find_bend(), has_vibrato(), _smooth(), curve(), note_curve(), test_a_note_named_after_its_bend_is_set_back_to_the_fretted_note() (+10 more)

### Community 21 - "Note Model & Tab Assignment"
Cohesion: 0.14
Nodes (20): Note, assign_tab(), fret_to_pitch(), is_valid_position(), _onset_groups(), _tab_label(), tab_to_ascii(), test_ascii_tab_lines_never_break_inside_a_bend() (+12 more)

### Community 22 - "API App & Security Setup"
Cohesion: 0.09
Nodes (8): health(), lifespan(), init_db(), BodySizeLimitMiddleware, OriginCheckMiddleware, SecurityHeadersMiddleware, validation_error_handler(), fail_interrupted_jobs()

### Community 23 - "Web Package Dependencies"
Cohesion: 0.08
Nodes (28): dependencies, @music-transcriber/shared-types@file:../../packages/shared-types, next@^15.5.26, opensheetmusicdisplay@^2.1.3, react@^19.3.0, react-dom@^19.3.0, devDependencies, @types/node@^20.11.0 (+20 more)

### Community 24 - "Technique Pitch Curves"
Cohesion: 0.10
Nodes (10): _continuous_path(), glides(), _in_chord(), _longest_run(), _one_string(), PitchCurve, PitchTracker, _refret() (+2 more)

### Community 25 - "API Security Tests"
Cohesion: 0.12
Nodes (26): test_allowed_origin_and_no_origin_pass(), test_audio_varies_on_origin(), test_corrected_tempo_is_saved_and_used_by_exports(), test_cross_site_requests_are_blocked(), test_detect_audio_format(), test_disallowed_extension_is_rejected(), test_empty_upload_is_rejected(), test_existing_database_gets_new_columns() (+18 more)

### Community 26 - "Web Synth & Techniques"
Cohesion: 0.14
Nodes (21): Strum, applyPitch(), click(), noteLevel(), NotePlayerOptions, pluck(), renderNotes(), useNotePlayer() (+13 more)

### Community 27 - "Upload Storage & Validation"
Cohesion: 0.14
Nodes (10): get_audio(), clean_text(), delete_file(), detect_audio_format(), InvalidUpload, media_type(), sanitize_filename(), save_upload() (+2 more)

### Community 28 - "Transcription Service"
Cohesion: 0.13
Nodes (14): _check_audio(), get_beat_tracker(), get_chord_recognizer(), get_predictor(), get_separator(), job_is_stale(), _load_predictor(), _load_separator() (+6 more)

### Community 29 - "CI Workflow"
Cohesion: 0.09
Nodes (25): concurrency, cancel-in-progress, group, run, jobs, python, web, name (+17 more)

### Community 30 - "Chord Chart Components"
Cohesion: 0.16
Nodes (18): ChordChart(), ChordChartProps, ChordDiagram(), ChordDiagramProps, RhythmPattern(), models, Bar, BASS_INTERVALS (+10 more)

### Community 31 - "GPU Docker Compose"
Cohesion: 0.11
Nodes (24): build, deploy, environment, TORCH_INDEX_URL, args, context, dockerfile, resources (+16 more)

### Community 32 - "ML YAML Config Loader"
Cohesion: 0.19
Nodes (12): apply_overrides(), _deep_merge(), feature_bins(), load_config(), _load_with_parents(), num_pitches(), tab_shape(), build_model() (+4 more)

### Community 33 - "Song Mode Chords"
Cohesion: 0.13
Nodes (10): beat_chords(), ChordSegment, _chord_tones(), _level(), meter(), song_key(), SongResult, _symbols() (+2 more)

### Community 34 - "Tab Optimizer Search"
Cohesion: 0.14
Nodes (13): _candidates(), search(), _evidence_cost(), _greedy_state(), _hand_center(), _local_cost(), _mean_string(), pitch_to_positions() (+5 more)

### Community 35 - "Database & Project Model"
Cohesion: 0.13
Nodes (6): Base, get_db(), Project, ProjectStatus, UTCDateTime, _utcnow()

### Community 36 - "Web App Shell & Upload"
Cohesion: 0.12
Nodes (10): metadata, STEPS, UploadPage(), ACCEPTED, AudioUploader(), extension(), TuningPicker(), MicRecorder (+2 more)

### Community 37 - "Song Melody & Tempo"
Cohesion: 0.14
Nodes (16): fill_instrumental(), downbeat_phase(), median_tempo(), _absolute(), slowing_beats(), strummed(), test_analysis_follows_tracked_beats(), test_fixed_grid_puts_bar_lines_through_the_downbeat() (+8 more)

### Community 38 - "MusicXML Measures & Tests"
Cohesion: 0.21
Nodes (18): notes_to_musicxml(), bar_time(), measure_key(), measure_tempo(), assert_full_measures(), parse(), staff_notes(), test_bar_lines_start_at_the_downbeat() (+10 more)

### Community 39 - "Web Note Editing Ops"
Cohesion: 0.19
Nodes (19): strums(), applyDrag(), copyNotes(), deleteNotes(), edited(), MAJOR, MIN_DURATION, MINOR (+11 more)

### Community 40 - "Config Overrides & Player Splits"
Cohesion: 0.14
Nodes (11): parse_overrides(), GuitarSetTrack, _player_id(), player_splits(), accuracy(), load_notes(), main(), _score() (+3 more)

### Community 41 - "GuitarSet Preprocessing"
Cohesion: 0.18
Nodes (9): load_track(), preprocess_audio(), preprocess_track(), subset_per_player(), main(), offline_variants(), prepare_extra(), prepare_guitarset() (+1 more)

### Community 42 - "Strum Detection"
Cohesion: 0.14
Nodes (9): beat_slots(), beats_of(), Strum, strums(), audio_strums(), _at16(), f1(), main() (+1 more)

### Community 43 - "Predictor Transcribe Pipeline"
Cohesion: 0.13
Nodes (3): _estimate_tempo(), _load(), TranscriptionResult

### Community 44 - "Visualization Helpers"
Cohesion: 0.14
Nodes (7): plot_pianoroll(), plot_tab(), plot_training_history(), save_figure(), midi_to_name(), name_to_midi(), test_midi_names_round_trip()

### Community 45 - "Transcription Dataset"
Cohesion: 0.14
Nodes (4): TranscriptionDataset, FeatureAugmenter, main(), set_seed()

### Community 46 - "Musical Key & Spelling"
Cohesion: 0.16
Nodes (8): Key, pitch_class_name(), spell(), _tonic_pc(), test_every_pitch_class_is_spelled_in_every_key(), test_key_names_round_trip(), test_key_signature(), test_spelling_follows_the_key()

### Community 47 - "Timing & Quantization"
Cohesion: 0.13
Nodes (12): hz_to_midi(), beats_to_seconds(), frames_to_time(), quantize_notes(), quantize_time(), seconds_per_beat(), seconds_to_beats(), time_to_frames() (+4 more)

### Community 48 - "Technique Benchmark"
Cohesion: 0.22
Nodes (12): TechniqueSettings, curve_labels(), main(), match(), own_note_labels(), previous_on_string(), report(), run_guitarset() (+4 more)

### Community 49 - "API Test Fixtures"
Cohesion: 0.13
Nodes (8): drifting_beats(), fake_separator(), _set_status(), test_guitar_projects_follow_the_tracked_beats(), test_retranscribe_can_switch_between_guitar_and_voice(), test_retranscribe_can_switch_song_mode(), test_song_mode_separates_the_guitar_first(), test_stuck_job_can_be_restarted_only_after_timeout()

### Community 50 - "Guitar YAML Config"
Cohesion: 0.10
Nodes (19): dataset, audio_type, name, test_player, val_player, extends, labels, max_midi (+11 more)

### Community 51 - "AAM Dataset"
Cohesion: 0.15
Nodes (7): AAMSong, _bass_line(), harte(), _rows(), songs(), parse_key(), references()

### Community 52 - "Guitar-TECHS Dataset"
Cohesion: 0.19
Nodes (11): _audio_path(), _best_lag(), _envelope(), estimate_offset(), find_tracks(), GuitarTechsTrack, label_offset(), load_track() (+3 more)

### Community 54 - "Shared TS Types"
Cohesion: 0.11
Nodes (19): Analysis, AnalysisRequest, ChordSymbol, KeySpan, ModelInfo, MusicKey, Note, NotesUpdate (+11 more)

### Community 55 - "Web TS Config"
Cohesion: 0.11
Nodes (18): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+10 more)

### Community 56 - "Beat Tracker & Chord Bench"
Cohesion: 0.17
Nodes (9): beat_bass(), BeatTracker, frame_chords(), aam_songs(), analyze(), guitarset_songs(), _harte(), main() (+1 more)

### Community 57 - "Chord Voicings & Render API"
Cohesion: 0.12
Nodes (6): chord_voicings(), render_musicxml(), song_chords(), _transcription_out(), update_notes(), open_strings()

### Community 58 - "Strumming Rhythm"
Cohesion: 0.16
Nodes (8): _add_missing_columns(), BarRhythm, _consensus(), Rhythm, strum_rhythm(), strums_from_notes(), test_sixteenth_patterns_are_written_in_sixteenths(), test_strums_and_their_pattern()

### Community 59 - "Augmentation Variants Config"
Cohesion: 0.12
Nodes (17): variants, decay_range, wet_range, distance_range, rt60_range, codec, distortion, echo (+9 more)

### Community 60 - "Model Meta Endpoint"
Cohesion: 0.17
Nodes (6): current_models(), _fingerprint(), model_version(), song_version(), voice_version(), test_model_version_follows_the_checkpoint_file()

### Community 61 - "ASGI Security Middleware"
Cohesion: 0.20
Nodes (6): limited_receive(), _header(), RequestTooLarge, send_with_headers(), _send_json(), _vary_on_origin()

### Community 62 - "Training Config"
Cohesion: 0.12
Nodes (16): training, amp, batch_size, early_stopping_patience, epochs, grad_clip, lr, monitor (+8 more)

### Community 63 - "CNN Building Blocks"
Cohesion: 0.15
Nodes (4): CNN, ConvBlock, ConvStack, CRNN

### Community 64 - "MIDI Import & Tests"
Cohesion: 0.27
Nodes (10): midi_to_notes(), notes_to_midi_bytes(), _messages(), test_a_bend_is_a_pitch_bend_curve_on_its_strings_channel(), test_a_slide_glides_into_the_next_note_without_a_gap(), test_hammer_ons_are_softer_and_read_back(), test_midi_round_trip(), test_notes_without_techniques_stay_on_one_channel() (+2 more)

### Community 65 - "Background Job Progress"
Cohesion: 0.16
Nodes (5): _progress_reporter(), report(), run_transcription(), _update(), transcribe_project()

### Community 66 - "Beat-Level Chord Fusion"
Cohesion: 0.21
Nodes (3): _BeatGrid, Chord, _intervals()

### Community 67 - "Analysis Tests"
Cohesion: 0.29
Nodes (12): labels(), progression(), strum(), test_arpeggios_are_chords_but_a_melody_is_not(), test_beat_phase_and_downbeat_with_a_pickup(), test_chord_names_are_spelled_in_the_key(), test_chord_progression(), test_downbeat_follows_chord_changes() (+4 more)

### Community 68 - "API Requirements"
Cohesion: 0.23
Nodes (12): API / worker requirements.txt, celery[redis]>=5.3, fastapi>=0.141, psycopg[binary]>=3.1, pydantic>=2.6, pydantic-settings>=2.2, python-multipart>=0.0.31, sqlalchemy>=2.0 (+4 more)

### Community 69 - "Base Audio/Paths Config"
Cohesion: 0.15
Nodes (12): audio, hop_length, normalize, sample_rate, offline_augmentation, enabled, paths, checkpoint_dir (+4 more)

### Community 70 - "IDMT Technique Dataset"
Cohesion: 0.24
Nodes (6): find_takes(), IdmtNote, IdmtTake, load_annotation(), test_annotation_strings_count_from_the_low_e(), test_lick_11_annotations_find_their_audio()

### Community 71 - "README Song Features & API"
Cohesion: 0.18
Nodes (10): REST API endpoints (/api/projects, /api/analysis, /api/chords/voicings, /api/render/musicxml, /api/models), ASCII TAB export, Beat This! beat/downbeat tracker (Foscarin et al., ISMIR 2024), GitHub Actions CI (ruff, pytest, tsc, Vitest, Next.js build), IDMT-SMT-Guitar dataset (Kehling et al., 2014), Karplus-Strong string synthesis in browser, MIDI export (tempo map, per-string channels, pitch bend), Web music editor (waveform, TAB, piano roll, loop, metronome, drag editing, undo) (+2 more)

### Community 72 - "Fake Predictor Tests"
Cohesion: 0.17
Nodes (8): client(), _deferred(), fake_model(), FakePredictor, test_deleted_project_during_transcription_is_skipped(), transcribe(), test_progress_is_visible_while_transcribing(), transcribe()

### Community 74 - "README Chord Pipeline"
Cohesion: 0.23
Nodes (9): AAM dataset (Ostermann et al., 2023), BTC Transformer chord recognizer (Park et al., ISMIR 2019; ml/models/btc.py), Capo suggestion (min duration-weighted chord-shape difficulty, capo 0-7), Chord shape/voicing search on strings (four-fret window), Demucs htdemucs_6s source separation, MUSDB18 stems (band-mix evaluation), Key, meter and chord analysis (packages/music-core/analysis.py), Song -> guitar mode (chords + melody, ml/inference/song.py) (+1 more)

### Community 75 - "Web Settings"
Cohesion: 0.31
Nodes (8): SettingsPage(), API_URL, DEFAULT_SETTINGS, EditorSettings, readSettings(), sanitize(), useSettings(), writeSettings()

### Community 76 - "Model Config"
Cohesion: 0.20
Nodes (10): model, bidirectional, conv_channels, dropout, embed_dim, heads, name, rnn_hidden (+2 more)

### Community 77 - "Mixed Guitar Config"
Cohesion: 0.20
Nodes (9): extends, paths, checkpoint_dir, processed_dir, splits_dir, training, num_workers, preload (+1 more)

### Community 78 - "Mixed Tab Config"
Cohesion: 0.20
Nodes (9): extends, paths, checkpoint_dir, processed_dir, splits_dir, training, num_workers, preload (+1 more)

### Community 79 - "Chord Inference Module"
Cohesion: 0.24
Nodes (4): add_bass(), _btc_checkpoint(), _root(), SongAnalysis

### Community 80 - "README Model & Pitch Trackers"
Cohesion: 0.20
Nodes (9): AMT model: CNN -> BiLSTM -> onset/frame/offset heads, CREPE (torchcrepe) pitch tracker, mir_eval metrics, Music Transcriber (end-to-end guitar AMT project), PESTO pitch tracker, Per-note pitch contour (6-harmonic salience, linear-cost Viterbi), pYIN pitch tracker (librosa), VocalSet + Annotated-VocalSet datasets (+1 more)

### Community 81 - "README EGDB & Tab Head"
Cohesion: 0.27
Nodes (7): EGDB electric guitar dataset, Simulated rooms and codecs augmentation (pyroomacoustics, Opus/AAC/MP3), Tab head model (learns string/fret directly), Alternative tunings and capo support, V10 notes model (default; rooms + aligned EGDB), V7 tab model (GuitarSet + EGDB + Guitar-TECHS), V9 notes model (range extended down to C2)

### Community 83 - "Compose Service Deps"
Cohesion: 0.25
Nodes (9): depends_on, postgres, redis, condition, condition, worker, command, depends_on (+1 more)

### Community 84 - "Compose API Environment"
Cohesion: 0.39
Nodes (9): environment, CORS_ORIGINS, DATABASE_URL, MODEL_CHECKPOINT, MODEL_TAB_CHECKPOINT, REDIS_URL, STORAGE_DIR, USE_CELERY (+1 more)

### Community 85 - "Feature Config"
Cohesion: 0.22
Nodes (9): features, bins_per_octave, fmin, fmin_midi, log_offset, n_bins, n_fft, n_mels (+1 more)

### Community 87 - "Compose Env Anchor"
Cohesion: 0.25
Nodes (8): x-api-env, CORS_ORIGINS, DATABASE_URL, MODEL_CHECKPOINT, MODEL_TAB_CHECKPOINT, REDIS_URL, STORAGE_DIR, USE_CELERY

### Community 88 - "Augmentation Config"
Cohesion: 0.25
Nodes (8): augmentation, enabled, freq_mask_width, freq_masks, gain_db, noise_std, time_mask_width, time_masks

### Community 89 - "Compose Volumes"
Cohesion: 0.29
Nodes (6): name, volumes, model-cache, postgres-data, storage, x-api-volumes

### Community 90 - "Compose Web Build"
Cohesion: 0.38
Nodes (7): build, NEXT_PUBLIC_API_URL, args, context, dockerfile, build, build

### Community 91 - "Compose Postgres"
Cohesion: 0.29
Nodes (7): POSTGRES_DB, POSTGRES_PASSWORD, POSTGRES_USER, environment, image, volumes, postgres

### Community 92 - "Inference Config"
Cohesion: 0.29
Nodes (7): inference, chunk_frames, context_frames, frame_threshold, min_note_duration, offset_threshold, onset_threshold

### Community 93 - "Shared Types Package"
Cohesion: 0.29
Nodes (6): description, main, name, private, types, version

### Community 94 - "Compose Healthcheck"
Cohesion: 0.53
Nodes (6): healthcheck, interval, retries, test, timeout, healthcheck

### Community 95 - "Compose Redis Service"
Cohesion: 0.33
Nodes (6): ports, volumes, image, services, api, redis

### Community 96 - "Mixed Low Config"
Cohesion: 0.33
Nodes (5): extends, labels, min_midi, paths, checkpoint_dir

### Community 97 - "Tab Head Config"
Cohesion: 0.33
Nodes (5): extends, model, heads, paths, checkpoint_dir

### Community 98 - "README Datasets & Splits"
Cohesion: 0.33
Nodes (4): Guitar-TECHS dataset, GuitarSet dataset, Offline augmentation (pitch shift, tempo, noise, reverb, overdrive, distortion, echo, phone, room), V8 notes model (GuitarSet + Guitar-TECHS)

### Community 99 - "Compose Web Service"
Cohesion: 0.40
Nodes (5): condition, api, web, depends_on, ports

### Community 100 - "Label Config"
Cohesion: 0.40
Nodes (5): labels, max_midi, min_midi, offset_frames, onset_frames

### Community 101 - "Loss Weights Config"
Cohesion: 0.40
Nodes (5): frame, offset, onset, tab, loss_weights

### Community 102 - "Room Config"
Cohesion: 0.40
Nodes (4): extends, paths, checkpoint_dir, splits_dir

### Community 103 - "Room+EGDB Config"
Cohesion: 0.40
Nodes (4): extends, paths, checkpoint_dir, splits_dir

### Community 105 - "Song API Tests"
Cohesion: 0.40
Nodes (4): fake_song(), transcribe(), test_songs_get_chords_bars_and_an_automatic_capo(), test_songs_get_key_changes_slash_chords_and_a_strumming_pattern()

### Community 107 - "Web Fixture Parity Test"
Cohesion: 0.60
Nodes (3): build(), _name(), test_fixture_matches_python()

## Knowledge Gaps
- **18 isolated node(s):** `music-transcriber`, `Music Transcriber (end-to-end guitar AMT project)`, `Offline augmentation (pitch shift, tempo, noise, reverb, overdrive, distortion, echo, phone, room)`, `Guitar-TECHS dataset`, `Simulated rooms and codecs augmentation (pyroomacoustics, Opus/AAC/MP3)` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 791 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Note` connect `Note Model & Tab Assignment` to `Voice Pitch Tracking`, `Evaluation Metrics`, `Audio Augmentation`, `Analysis API Endpoint`, `Dataset Loading (GuitarSet)`, `Pitch Range & Track Splitting`, `Project REST Routes`, `Predictor & EGDB Alignment`, `Key & Chord Analysis`, `MIDI Export`, `MusicXML Quantization`, `EGDB Dataset`, `Bend & Vibrato Detection`, `Technique Pitch Curves`, `Song Mode Chords`, `Song Melody & Tempo`, `MusicXML Measures & Tests`, `Config Overrides & Player Splits`, `GuitarSet Preprocessing`, `Strum Detection`, `Predictor Transcribe Pipeline`, `Visualization Helpers`, `Timing & Quantization`, `Technique Benchmark`, `Guitar-TECHS Dataset`, `Chord Voicings & Render API`, `Strumming Rhythm`, `MIDI Import & Tests`, `Beat-Level Chord Fusion`, `Analysis Tests`, `IDMT Technique Dataset`, `Fake Predictor Tests`, `Song API Tests`, `Web Fixture Parity Test`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 159 inferred relationships involving `Note` (e.g. with `analysis_out()` and `analyze_notes()`) actually correct?**
  _`Note` has 159 INFERRED edges - model-reasoned connections that need verification._
- **What connects `music-transcriber`, `Music Transcriber (end-to-end guitar AMT project)`, `Offline augmentation (pitch shift, tempo, noise, reverb, overdrive, distortion, echo, phone, room)` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Voice Pitch Tracking` be split into smaller, more focused modules?**
  _Cohesion score 0.061175666438824335 - nodes in this community are weakly interconnected._
- **Why does `Settings` connect `API Settings` to `Upload Storage & Validation`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Are the 34 inferred relationships involving `Key` (e.g. with `analyze_notes()` and `chord_voicings()`) actually correct?**
  _`Key` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Should `Evaluation Metrics` be split into smaller, more focused modules?**
  _Cohesion score 0.08295625942684766 - nodes in this community are weakly interconnected._