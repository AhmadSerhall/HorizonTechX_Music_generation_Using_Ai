# NeuraTune — AI Music Generation Studio 🎵

NeuraTune is an AI-powered music generation studio developed as **Task 3 of the HorizonTechX Artificial Intelligence Internship**.

It explores how recurrent neural networks can learn patterns from MIDI music and generate new musical sequences. The project covers the complete pipeline from MIDI dataset validation and preprocessing to LSTM training, autoregressive generation, MIDI reconstruction, audio preview, visualization, and interactive analysis through Streamlit.

## ✨ Features

- AI music generation with TensorFlow/Keras LSTMs
- MIDI dataset discovery and validation
- Three experimental representations: V1, V2, and V3
- Temperature-based sampling
- Top-k sampling
- Reproducible generation with seeds
- Configurable generation length and tempo
- Minimum playback-duration control
- Polyphonic MIDI generation
- MIDI download
- Basic local WAV audio preview
- Optional FluidSynth/SoundFont audio rendering
- Animated Synthesia-style piano visualizer
- Play/Pause, Restart, progress, fullscreen, and audio synchronization
- Pitch and rhythm analysis
- Training insights and model information
- Persistent composition history/library
- Rename, delete, favorite, and filter generated compositions
- Presets for different creativity levels
- Responsive dark music-studio interface

## 🧠 AI Approach

The project uses **Long Short-Term Memory (LSTM)** recurrent neural networks because music is sequential: future musical events depend on preceding context.

### V1 — Event Tokens

V1 represented events as tokens such as:

```text
note:C4|duration:0.5
chord:C4.E4.G4|duration:1.0
```

Architecture:

```text
Integer Event IDs
      ↓
Embedding(64)
      ↓
LSTM(64)
      ↓
Dropout(0.2)
      ↓
Dense(vocabulary_size, softmax)
```

Key characteristics:

- 50-event sequences
- 31,668-token vocabulary
- 149,953 training sequences
- ~4.1M parameters

V1 established the end-to-end pipeline but exposed limitations: a very large sparse vocabulary, many rare tokens, and loss of general timing/rest/polyphony information.

### V2 — Explicit Musical Attributes

V2 represents each event as:

```text
[pitch, delta_time, duration]
```

It preserves:

- MIDI pitch
- time since the previous onset
- note duration
- simultaneous notes
- piece boundaries

Architecture:

```text
Pitch Embedding(32)
       +
Delta Embedding(16)
       +
Duration Embedding(16)
       ↓
Concatenation
       ↓
LSTM(256)
       ↓
Dropout(0.3)
       ↓
Pitch / Delta / Duration heads
```

Parameters: **392,902**

V2 supports temperature, top-k, duration sampling controls, onset-note limits, seed, and tempo. **V2 is the model currently integrated into the Streamlit application.**

### V3 — Onset-Grouped Experimental Model

V3 groups simultaneous notes into one onset timestep.

Each timestep contains up to:

- 8 pitch slots
- 8 aligned duration slots
- 1 delta-time value

Inputs:

```text
Pitch:    (batch, 50, 8)
Duration: (batch, 50, 8)
Delta:    (batch, 50)
```

Outputs:

```text
Pitch:    (batch, 8, 76)
Duration: (batch, 8, 28)
Delta:    (batch, 25)
```

Architecture:

```text
Pitch Embedding(32)
Duration Embedding(16)
Delta Embedding(16)
        ↓
Per-slot feature combination
        ↓
Onset-group representation
        ↓
LSTM(256)
        ↓
Dropout(0.3)
        ↓
Pitch / Duration / Delta outputs
```

Parameters: **896,297**

Current V3 training artifact:

- 3 source pieces
- 5,101 training sequences
- 960 validation sequences
- piece-level split
- PAD-aware losses
- best validation loss: **5.7387 at epoch 2**
- early stopping after epoch 5 with best weights restored

V3 remains an experimental model and is **not integrated into the Streamlit UI**.

## 📚 Dataset

The project uses the **MAESTRO v3.0.0 MIDI dataset from Google Magenta**.

The full MIDI collection used during development contains 1,276 MIDI performances.

Raw MIDI files are intentionally not included in the repository. Place them under:

```text
data/raw/classical/
```

The repository should keep raw dataset files out of Git.

## 🎼 MIDI Preprocessing

The project uses **music21** for MIDI parsing and processing.

The preprocessing pipeline supports:

- MIDI discovery
- validation
- note extraction
- chord extraction
- pitch/duration extraction
- timing quantization
- sequence creation
- metadata generation
- piece-level dataset splitting

V1, V2, and V3 use different representations to investigate how representation affects generated music.

## 📊 Evaluation

Generated music was evaluated using both model metrics and musical characteristics, including:

- pitch diversity
- pitch range
- note density
- timing distribution
- duration distribution
- simultaneous notes/polyphony
- repeated events
- pitch movement
- harmonic/semitone clashes
- generated duration

A key lesson was that classification accuracy and loss alone do not determine whether generated music sounds musically convincing, so listening and structural MIDI analysis were also used.

## 🖥️ Streamlit Application

The final NeuraTune app uses **V2** for generation.

The application flow is:

```text
Streamlit UI
     ↓
V2 LSTM
     ↓
V2 Autoregressive Generator
     ↓
Generated MIDI
     ↓
Audio Preview
     ↓
Piano Visualizer
     ↓
Pitch/Rhythm Analysis
     ↓
History / Library
     ↓
MIDI Download
```

The existing UI controls include:

- generation length
- creativity/temperature
- seed
- tempo
- minimum duration

The application caches the trained V2 model so it is not repeatedly loaded for every interaction.

## 🎧 Audio Preview

NeuraTune includes a lightweight local WAV renderer, so basic audio preview works without FluidSynth.

Optional FluidSynth configuration:

```powershell
$env:FLUIDSYNTH_PATH = "C:\path\to\fluidsynth.exe"
$env:SOUNDFONT_PATH = "C:\path\to\your\soundfont.sf2"
```

If FluidSynth is unavailable, the application falls back to the built-in renderer.

## 🎹 Piano Visualizer

The application includes a real-time MIDI piano visualization with:

- falling note blocks
- animated keyboard
- active-key highlighting
- MIDI timing
- play/pause
- restart
- progress/time display
- fullscreen mode
- audio synchronization
- dynamic pitch visualization

## 📈 Analysis

### Overview
Composition-level metrics such as note count, unique pitches, pitch range, durations, tempo, and polyphony.

### Pitch Analysis
Pitch distribution, pitch classes, and register/range information.

### Rhythm Analysis
Duration and timing distributions.

### Training Insights
Dataset size, sequences, vocabulary/model characteristics, architecture, parameters, and training/validation metrics.

### About the Model
Explanation of the current neural music-generation approach.

## 📁 Project Structure

```text
Horizon-TechX-Music-generation-Using-Ai/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── classical/
│   └── processed/
│       ├── classical_preprocessed.json
│       ├── classical_sequences.npz
│       ├── classical_v2_metadata.json
│       ├── classical_v2_sequences.npz
│       ├── classical_v3_metadata.json
│       └── classical_v3_sequences.npz
│
├── models/
│   ├── classical_lstm.keras
│   ├── classical_training_metadata.json
│   ├── classical_lstm_v2.keras
│   ├── classical_training_metadata_v2.json
│   ├── classical_lstm_v3.keras
│   └── classical_training_metadata_v3.json
│
├── outputs/
│
└── src/
    ├── dataset.py
    ├── preprocess.py
    ├── model.py
    ├── train.py
    ├── generate.py
    ├── evaluate.py
    ├── preprocess_v2.py
    ├── model_v2.py
    ├── train_v2.py
    ├── generate_v2.py
    ├── preprocess_v3.py
    ├── model_v3.py
    ├── train_v3.py
    └── audio_preview.py
```

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- music21
- NumPy
- Matplotlib
- Streamlit
- MIDI
- LSTM / RNN
- HTML / CSS / JavaScript for the piano visualizer

## 🚀 Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## ▶️ Run NeuraTune

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open the local URL provided by Streamlit, normally:

```text
http://localhost:8501
```

## 🧪 Dataset Validation

```powershell
python -m src.dataset --genre classical --limit 50
```

## 🔄 Preprocessing

V1:

```powershell
python -m src.preprocess --genre classical --limit 50 --sequence-length 50
```

V2:

```powershell
python -m src.preprocess_v2 --genre classical --limit 50 --sequence-length 50
```

V3:

```powershell
python -m src.preprocess_v3 --genre classical --limit 50 --sequence-length 50
```

## 🏋️ Training

V1:

```powershell
python -m src.train --genre classical --epochs 5 --batch-size 64
```

V2:

```powershell
python -m src.train_v2 --genre classical --epochs 15 --batch-size 128
```

V3:

```powershell
python -m src.train_v3 --genre classical --epochs 15 --batch-size 128
```

## 🎵 V2 Generation

Example:

```powershell
python -m src.generate_v2 --genre classical --length 200 --temperature 0.8 --top-k 10 --seed 42 --tempo 100
```

Additional V2 generation controls include duration sampling and maximum notes per onset. Generated MIDI files are saved in `outputs/`.

## ⚠️ Limitations

This is an educational/internship project rather than a production-grade music-generation system.

Current limitations include:

- relatively limited training data compared with the complexity of music
- primarily classical piano MIDI
- limited long-term musical structure
- possible repetition or unstable transitions
- no explicit deep understanding of music theory
- V2 predicts musical components through separate output heads
- generated compositions can contain unusual harmonic/rhythmic combinations
- audio quality depends on the renderer
- V3 does not yet have a complete generation pipeline integrated into the UI

## 🔮 Future Improvements

Potential extensions include:

- training on a larger portion of MAESTRO
- adding jazz and other genres
- Transformer-based music generation
- longer-context models
- better joint pitch/duration prediction
- richer harmony and melody modeling
- velocity and dynamics
- rest modeling
- instrument-aware generation
- improved MIDI-to-audio synthesis
- full V3 generation and evaluation
- larger-scale training

## 🧠 Key Development Lessons

1. **Representation matters.** V1's huge token vocabulary made learning difficult.
2. **Timing matters.** Explicit onset timing significantly improves the representation of rhythm.
3. **Polyphony is difficult.** Simultaneous notes require dedicated structure.
4. **Sampling matters.** Temperature and top-k affect diversity and stability.
5. **Generative quality needs perceptual evaluation.** Loss and accuracy are useful, but listening and MIDI analysis are essential.

## 🏁 Project Status

**HorizonTechX Internship — Task 3: Completed**

NeuraTune demonstrates an end-to-end AI music-generation workflow:

```text
MIDI Dataset
    ↓
Preprocessing
    ↓
LSTM Training
    ↓
V2 Autoregressive Generation
    ↓
MIDI Reconstruction
    ↓
Audio Preview
    ↓
Animated Piano Visualization
    ↓
Musical Analysis
    ↓
Composition Library
    ↓
MIDI Download
```

## 🙏 Dataset Attribution

This project uses the **MAESTRO v3.0.0 MIDI dataset from Google Magenta** for educational/research purposes. Obtain and use the dataset according to its official license and terms.

The raw dataset is intentionally excluded from this repository.

## 👨‍💻 Internship Project

Developed as part of the **Artificial Intelligence Internship at HorizonTechX**.

**Project:** NeuraTune — AI Music Generation Studio  
**Task:** Music Generation Using AI  
**Core technologies:** Python, TensorFlow/Keras, LSTM, music21, NumPy, Streamlit, MIDI
