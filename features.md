# NeuraTune Features

## AI music pipeline

- Discovers `.mid` and `.midi` files recursively in `data/raw/`, with separate `classical/` and `jazz/` folders.
- Validates MIDI files with `music21` and reports unreadable files without stopping the dataset check.
- Extracts note and chord events from validated MIDI files.
- Stores notes as `note:<pitch>|duration:<duration>` and chords as sorted pitches with a duration.
- Quantizes event durations to quarter-beat steps, with a minimum duration of `0.25`.
- Builds event vocabulary mappings and fixed-length input/next-event training sequences.
- Saves lightweight preprocessing metadata as JSON and numeric sequences as compressed NumPy `.npz` files.
- Trains an integer-ID next-event TensorFlow/Keras model: `Embedding → LSTM → Dropout → Softmax`.
- Uses an embedding layer and sparse categorical cross-entropy, avoiding memory-heavy one-hot encoded inputs.
- Saves trained models in the modern `.keras` format along with training metadata.

## MIDI generation and evaluation

- Loads the existing trained model and processed data without preprocessing or retraining.
- Selects a valid sequence seed and generates events autoregressively.
- Supports reproducible generation through a random seed.
- Uses temperature sampling for conservative, balanced, or more experimental output.
- Reconstructs notes and chords as `music21` objects and exports playable MIDI files to `outputs/`.
- Supports optional tempo and playback-only minimum-duration controls; neither changes the model vocabulary or predictions.
- Evaluates generated MIDI for notes, chords, pitch range, unique pitches, total and average duration, pitch-class distribution, and duration distribution.

## NeuraTune Streamlit studio

- Modern responsive dark music-studio interface with a compact two-column desktop layout.
- Generation controls for event length, creativity/temperature, random seed, tempo, and minimum duration.
- Cached model loading so widget interactions do not reload the Keras model.
- A composition card with generated-event, creativity, tempo, pitch, pitch-range, seed, and duration details.
- MIDI download for every generated composition.
- Local browser audio preview with native play/pause controls.
  - A cached WAV preview is rendered locally.
  - A built-in basic synthesizer is used when FluidSynth is unavailable.
  - Optional FluidSynth plus a user-provided General MIDI soundfont can provide a higher quality preview.
- Persistent recent-composition history in `outputs/history.json`.
- Restores the newest valid composition after refresh and lets users select earlier generated MIDI files without regenerating them.
- Loading overlay while the neural network is composing.
- Overview, pitch-class, rhythm, and model-explanation analysis tabs.
- Pitch and rhythm charts have no visible toolbar button: click a chart to open it in browser fullscreen mode; click again or exit fullscreen to return.
- Responsive behavior stacks the studio controls and composition card on narrower screens.

## Commands

```powershell
# Validate manually added MIDI files
python -m src.dataset --genre classical --limit 50

# Preprocess a small dataset subset
python -m src.preprocess --genre classical --limit 50 --sequence-length 50

# Train with a development-sized subset
python -m src.train --genre classical --epochs 1 --batch-size 64 --max-sequences 5000

# Generate MIDI from the trained model
python -m src.generate --genre classical --length 200 --temperature 1.0 --seed 42 --min-duration 0.5 --tempo 100

# Evaluate a generated MIDI file
python -m src.evaluate <generated-midi-path>

# Run the NeuraTune studio
streamlit run app.py
```
