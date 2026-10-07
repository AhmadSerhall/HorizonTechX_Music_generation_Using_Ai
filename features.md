# NeuraTune — Current Features

NeuraTune is the AI Music Generation Studio developed for HorizonTechX Internship Task 3. This inventory covers the current application and the research pipelines retained in the project.

## V4 Production generation

- Streamlit uses the frozen **original unweighted, EOS-aware V4 Production** checkpoint: `models/classical_lstm_v4_production.keras`.
- Its training metadata is `models/classical_training_metadata_v4_production.json`; processed inputs and mappings come from `data/processed/classical_v4_sequences.npz` and `data/processed/classical_v4_metadata.json`.
- Model resources are cached with `st.cache_resource` across interactions.
- The Generate button reuses `src/generate_v4.py`, starting from a real 50-onset sequence from the processed dataset.
- Each generated musical moment is an onset group containing up to eight aligned pitch/duration slots.
- A historical LSTM encoder provides context to a conditional GRU pitch decoder. START begins decoding; EOS terminates the onset.
- Pitch slots are generated autoregressively with associated durations. A separate delta head predicts spacing between onset groups.
- Learned timing uses a `0.25` quarter-length grid. Simultaneous notes are reconstructed at the same offset with their individual durations.
- User-controlled seeds support reproducible selection and sampling with the same model, settings, and runtime.
- New MIDI files use timestamped names such as `outputs/neuraltune_v4_<timestamp>.mid`.
- The exported V4 MIDI is the authoritative composition. The app no longer parses and rewrites it to enforce a minimum duration.
- Audio rendering reads the saved MIDI without modifying its pitches, onsets, durations, or file contents.
- Missing production artifacts and generation failures receive user-facing errors. New generation does not silently fall back to another model version.

## Studio interface and controls

- Responsive dark interface with NeuraTune branding, teal/sky-blue accents, and a compact two-column desktop layout.
- **Composition Length:** 50–500 generated musical moments, default 200. Internally these are onset groups, not individual notes.
- **Creativity:** manual pitch-temperature slider from `0.3` to `1.5`.
- Preset buttons update the same slider and visually indicate an exact preset match:

| Preset | Pitch temperature |
| --- | ---: |
| Focused | 0.8 |
| Balanced | 1.0 |
| Experimental | 1.1 |

- Internal production settings remain pitch top-k `10`, delta temperature `0.8`, and duration temperature `0.8` for all presets.
- **Tempo:** 60–180 BPM, default 100; changes MIDI playback tempo without changing model predictions.
- **Random Seed:** user-selectable, default 42.
- Presets do not change composition length, seed, or tempo.
- A loading overlay appears while the network composes.
- The composition card shows the display name, musical-moment count, creativity, tempo, unique pitches, pitch range, and seed.
- New V4 compositions retain model-predicted durations. The old Minimum Duration control has been removed from the production UI.
- The decorative sequence graphic is labeled as decorative rather than presented as an audio waveform.
- The selected composition can be downloaded as MIDI.

## Composition Library

- Persistent local history in `outputs/history.json`, with MIDI files in `outputs/`; no database is required.
- Retains up to ten recent valid compositions across refreshes and restarts.
- Newly generated tracks are automatically selected in the library and loaded into the player and analysis area.
- A fresh session restores the newest valid saved composition.
- Entries store a unique ID, display name, MIDI path, optional preview path, timestamp, generation settings, evaluation metrics, favorite state, and model version.
- New entries identify their model as `V4 Production`; default names follow the `Neural Composition 001` pattern.
- Selecting an entry loads its saved MIDI without regenerating it.
- Rename updates the display name without renaming the physical MIDI file.
- Favorite/unfavorite state persists, with a compact **All / Favorites** filter.
- Delete requires confirmation and removes the entry and associated MIDI/recorded preview when their paths are safely inside `outputs/`.
- Deleting the selected entry selects another valid composition or shows the empty state.
- Older V1/V2 entries remain readable and playable, including legacy minimum-duration settings. These settings are not applied to new V4 output.
- Missing MIDI entries are skipped; missing or corrupt history is handled safely.

## Local audio rendering and caching

- Playback uses a WAV rendered from the selected saved MIDI, without a second generation call.
- Preferred renderer: **FluidSynth with a configured local SoundFont**.
- FluidSynth previews are validated as stereo, 44,100 Hz, 16-bit PCM WAV files.
- Executable detection supports `FLUIDSYNTH_PATH`, lookup on `PATH`, and the Windows name `fluidsynth.exe`.
- SoundFont configuration prefers `NEURATUNE_SOUNDFONT`; `SOUNDFONT_PATH` remains supported when the preferred variable is unset.
- Existing `.sf2` and `.sf3` files pass path validation; actual format support depends on the installed FluidSynth build.
- FluidSynth is invoked through a safe subprocess argument list, with rendering options before the SoundFont/MIDI positional arguments.
- If FluidSynth, a SoundFont, or a valid rendered WAV is unavailable, the built-in sine-wave synthesizer supplies a mono, 22,050 Hz, 16-bit PCM preview.
- The frontend reports **High-quality SoundFont preview** or **Basic local MIDI preview** according to the renderer used.
- Previews are cached in `outputs/previews/` using the MIDI content hash and renderer configuration. FluidSynth cache identity also includes executable and SoundFont identity.
- A cached basic preview does not take priority over an available valid SoundFont preview.
- Rendering stays local. No external audio service, automatic SoundFont download, or bundled instrument samples are required.

Optional Windows configuration, set before launching Streamlit:

```powershell
$env:FLUIDSYNTH_PATH = "C:\path\to\fluidsynth.exe"
$env:NEURATUNE_SOUNDFONT = "C:\path\to\legally-obtained-soundfont.sf2"
streamlit run app.py
```

FluidSynth is optional. The SoundFont changes the instrument sound while the saved MIDI remains the same composition.

## Animated Piano Visualizer

- Starting playback from the compact composition player opens a centered Piano Visualizer modal.
- The visualization uses actual MIDI pitches, offsets, and durations extracted with `music21`, expanding chord pitches individually.
- Falling note bars use the dark teal/cyan theme and a short look-ahead for upcoming notes.
- A dynamic pitch range displays white and black piano keys with active-key highlighting.
- Bars align with the corresponding piano keys, and their lengths reflect note duration.
- Overlapping sustained/retriggered notes on the same pitch use separate narrow visual lanes so a long bar does not hide a shorter note. This changes presentation only.
- One **Play / Pause** button toggles with playback state.
- **Restart** returns playback and animation to the beginning; progress and elapsed/total time are displayed.
- A top-right **Exit** button closes the modal and stops/resets playback.
- **Minimize** closes the large view while preserving playback state.
- A circular floating player appears at the bottom right when minimized, with a progress ring and a larger centered play/pause state icon.
- The floating player has a pop-in entrance animation and a subtle pulse while playing.
- Clicking it reopens the visualizer at the current position without requiring a pause/play cycle.
- The embedded HTML5 audio player and animation share the audio playback clock. JavaScript uses `requestAnimationFrame`, without Python calls per animation frame.
- If audio is unavailable, the MIDI timeline can still animate with a visualization-only status.
- The visualizer opens from the player rather than occupying an analysis tab.

## Analysis and training insights

| Tab | Content |
| --- | --- |
| Overview | Selected-MIDI metrics including note/chord counts, pitch range/diversity, and duration information |
| Pitch Analysis | Pitch-class distribution chart |
| Rhythm | Note-duration distribution chart |
| Training Insights | Saved dataset/preprocessing information, architecture, parameter count, metrics, and available training history |
| About the Model | Active V4 representation, LSTM encoder, conditional GRU decoder, START/EOS, aligned durations, and learned timing |

- Analysis follows the selected saved MIDI, including older library compositions.
- Matplotlib charts match the theme and expand by clicking the chart, without a separate expand button.
- Training Insights reads V4 Production training metadata and original V4 preprocessing metadata.
- Unavailable metadata values are identified as unavailable; epoch loss curves appear only when training history was saved.
- Production insights describe the active model rather than presenting V2 metrics as V4 results.

## Dataset and model development features

- Recursive discovery and validation of `.mid`/`.midi` files under genre directories in `data/raw/`.
- MAESTRO classical piano MIDI is used for current production work; raw dataset files are obtained separately.
- `music21` extraction, numeric NPZ storage, JSON metadata, categorical mappings, and Keras model artifacts.
- Later preprocessing versions preserve timing and source-piece identity, with sequences that remain within one composition.
- V4 preprocessing groups simultaneous notes, preserves pitch-duration alignment, removes exact duplicates, and explicitly pads unused slots.
- Groups larger than eight notes are measured and reduced deterministically, retaining the lowest/highest pitches and distributed interior notes.
- The original V4 dataset contains 50 pieces, 120,824 onset groups, and 118,324 sequences of 50 onset groups.
- V4 uses pieces 0–39 for training and 40–49 for validation: 88,814 training sequences and 29,510 validation sequences.
- Production embeddings are pitch/duration/delta `32/16/16`, with an LSTM(256) encoder, Dropout(0.3), and a conditional GRU(256) pitch decoder.
- Training uses teacher forcing and explicit EOS termination loss. Duration loss covers real pitch positions; production rhythm losses are unweighted.
- Training pipelines save metrics and support early stopping with best-weight restoration; V4 also uses learning-rate reduction.

| Version | Implemented pipeline | Current role |
| --- | --- | --- |
| V1 | Note/chord-plus-duration tokens, vocabulary mappings, LSTM training, temperature generation, MIDI evaluation | Retained baseline and CLI |
| V2 | Separate pitch/delta/duration attributes and heads, piece-level split, timing-aware reconstruction, top-k and optional inference safeguards | Retained CLI; historical compositions remain playable |
| V3 | Onset-grouped preprocessing and LSTM training with aligned pitch/duration slots | Experimental; no V3 generation integration |
| V4 | Dataset diagnostics, onset-grouped preprocessing, conditional EOS-aware training/generation, rhythm and onset diagnostics | Frozen unweighted production checkpoint powers Streamlit; other V4 checkpoints remain experiments |
| V4E | Separate compact rhythm-bucket preprocessing, training, and generation | Experimental; not integrated into Streamlit |

## Diagnostic and evaluation tools

- `src/evaluate.py` analyzes MIDI note/chord counts, pitch range, unique pitches, pitch-class distribution, and duration statistics.
- `src/analyze_training_music.py` measures onset size, rhythm, register, highest-pitch melody proxy, simultaneous intervals, repetition, and per-piece statistics from original MIDI.
- V4 generation diagnostics report onset-size distribution, EOS/cap behavior, timing distributions, register, repetition, melodic movement, and close-interval harmony.
- Statistical diagnostics support comparison with training data; they do not by themselves establish perceived musical quality.

## Common commands

Run from the project root with dependencies from `requirements.txt` installed.

```powershell
# Launch the studio with its frozen V4 Production resources
streamlit run app.py

# Validate a local classical MIDI subset
python -m src.dataset --genre classical --limit 50

# Generate with the production checkpoint explicitly selected
python -m src.generate_v4 --genre classical --length 200 --temperature 1.0 --top-k 10 --delta-temperature 0.8 --duration-temperature 0.8 --seed 42 --tempo 100 --model-path models/classical_lstm_v4_production.keras --training-metadata-path models/classical_training_metadata_v4_production.json

# Evaluate an existing MIDI; replace the example filename
python -m src.evaluate outputs/your-composition.mid
```

Version-specific preprocessing, training, and V1/V2 generation CLIs remain available. Normal studio use loads existing artifacts without preprocessing or retraining. The V4 CLI default checkpoint differs from the app's frozen production checkpoint, so the explicit paths above select the production model.

## Current limitations

- Generated quality remains variable, including repetition, unusual harmony/rhythm, and limited long-term structure.
- The basic synthesizer is a functional preview rather than a realistic sampled piano; SoundFont quality depends on the configured library.
- Browser audio permissions can prevent playback; the visualizer supports a visualization-only fallback.
- The library currently retains ten recent valid entries rather than an unlimited catalog.
- There is no model-switching UI, database, authentication, or external generation API.
