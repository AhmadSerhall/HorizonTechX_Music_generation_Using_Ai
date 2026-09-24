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

This repository currently contains Phases 1 and 2 only.
