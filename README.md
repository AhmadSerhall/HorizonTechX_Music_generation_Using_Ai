# HorizonTechX AI Music Generation

## Task objective

Train an LSTM-based model on MIDI music, generate new musical sequences, and export them as MIDI files.

## Planned stack

- Python
- music21
- NumPy
- TensorFlow
- Streamlit
- Matplotlib

## Planned pipeline

`MIDI Dataset → Preprocessing → Sequences → LSTM → Music Generation → MIDI Output`

## MIDI dataset setup

Add your MIDI files manually; this project does not download a dataset automatically.

- Put classical files in `data/raw/classical/`.
- Put jazz files in `data/raw/jazz/`.
- Use `.mid` or `.midi` files. Subfolders are supported.

Later training can use one genre at a time. Validate the dataset with:

```powershell
python -m src.dataset
```

## MIDI preprocessing

Create event sequences and a JSON artifact with:

```powershell
python -m src.preprocess --genre classical --limit 50 --sequence-length 50
```

Each note is stored as `note:C4|duration:1.0`, and each chord stores sorted component
pitches, such as `chord:C4.E4.G4|duration:1.0`. Durations are quantized to the
nearest 0.25 beat, with a minimum of 0.25. The lightweight
`data/processed/<genre>_preprocessed.json` file contains the events, vocabulary
mappings, and metadata. The `X` input sequences and `y` targets are stored separately
in `data/processed/<genre>_sequences.npz`. `id_to_event` is a list, so an integer
event ID is used as its index.

## LSTM training

Train a next-event model from the processed artifacts without parsing MIDI again:

```powershell
python -m src.train --genre classical --epochs 1 --batch-size 64 --max-sequences 5000
```

The model uses integer event IDs with an Embedding layer, then an LSTM, Dropout, and
a softmax output layer. Training saves `models/<genre>_lstm.keras` and
`models/<genre>_training_metadata.json`.

## Music generation

Generate a MIDI file from the trained model without retraining:

```powershell
python -m src.generate --genre classical --length 200 --temperature 1.0 --seed 42
```

The temperature controls sampling randomness: `0.5` is more conservative, `1.0` is
balanced, and `1.5` is more random. Generated MIDI files are saved in `outputs/`.

For refined playback with a minimum reconstructed duration and tempo:

```powershell
python -m src.generate --genre classical --length 200 --temperature 1.0 --seed 42 --min-duration 0.5 --tempo 100
```

Evaluate a generated file with:

```powershell
python -m src.evaluate <generated-midi-path>
```

This repository currently contains Phases 1 through 7 only.

## Streamlit studio

Run the NeuraTune interface with:

```powershell
streamlit run app.py
```

The studio provides generation controls for length, temperature, seed, tempo, and
playback-only minimum duration. It offers MIDI download plus a pitch and rhythm
analysis dashboard for each generated composition.
