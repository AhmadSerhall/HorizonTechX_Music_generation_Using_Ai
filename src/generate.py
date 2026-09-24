"""Generate musical events with a trained LSTM and export them as MIDI."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from music21 import chord, note, stream, tempo
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_generation_artifacts(
    genre: str,
) -> tuple[keras.Model, np.ndarray, list[str], int]:
    """Load the trained model, vocabulary, and encoded training sequences."""
    model_path = MODELS_DIR / f"{genre}_lstm.keras"
    json_path = PROCESSED_DATA_DIR / f"{genre}_preprocessed.json"
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_sequences.npz"

    if not model_path.exists() or not json_path.exists() or not sequence_path.exists():
        raise FileNotFoundError(
            f"Generation artifacts for '{genre}' were not found. Complete preprocessing and training first."
        )

    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    id_to_event = artifact["id_to_event"]
    sequence_length = int(artifact["metadata"]["sequence_length"])

    with np.load(sequence_path, allow_pickle=False) as sequence_data:
        input_sequences = np.asarray(sequence_data["X"], dtype=np.int32)

    model = keras.models.load_model(model_path, compile=False)
    if input_sequences.ndim != 2 or input_sequences.shape[1] != sequence_length:
        raise ValueError("Stored X sequences do not match the preprocessing metadata.")
    if model.output_shape[-1] != len(id_to_event):
        raise ValueError("Model output size does not match the stored vocabulary.")

    return model, input_sequences, id_to_event, sequence_length


def select_seed_sequence(input_sequences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Select one valid encoded sequence as the starting context."""
    if len(input_sequences) == 0:
        raise ValueError("No input sequences are available for generation.")

    seed_index = rng.integers(len(input_sequences))
    return input_sequences[seed_index].copy()


def sample_with_temperature(
    probabilities: np.ndarray, temperature: float, rng: np.random.Generator
) -> int:
    """Sample one event ID from probabilities adjusted by temperature."""
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    safe_probabilities = np.nan_to_num(
        np.asarray(probabilities, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0
    )
    safe_probabilities = np.clip(safe_probabilities, 1e-12, 1.0)
    scaled_logits = np.log(safe_probabilities) / temperature
    scaled_logits -= np.max(scaled_logits)
    adjusted_probabilities = np.exp(scaled_logits)
    adjusted_probabilities /= adjusted_probabilities.sum()
    return int(rng.choice(len(adjusted_probabilities), p=adjusted_probabilities))


def generate_event_ids(
    model: keras.Model,
    seed_sequence: np.ndarray,
    length: int,
    temperature: float,
    rng: np.random.Generator,
) -> list[int]:
    """Generate event IDs by repeatedly predicting and extending the seed sequence."""
    current_sequence = seed_sequence.copy()
    generated_ids: list[int] = []

    for _ in range(length):
        probabilities = model.predict(current_sequence[np.newaxis, :], verbose=0)[0]
        next_event_id = sample_with_temperature(probabilities, temperature, rng)
        generated_ids.append(next_event_id)
        current_sequence = np.append(current_sequence[1:], next_event_id).astype(np.int32)

    return generated_ids


def event_token_to_music21(
    event_token: str, min_duration: float | None = None
) -> note.Note | chord.Chord | None:
    """Convert one event token into a music21 note or chord, if it is well formed."""
    try:
        event_part, separator, duration_text = event_token.partition("|duration:")
        if not separator:
            return None

        duration = float(duration_text)
        if not math.isfinite(duration) or duration <= 0:
            return None

        event_type, pitch_text = event_part.split(":", maxsplit=1)
        if event_type == "note" and pitch_text:
            musical_event = note.Note(pitch_text)
        elif event_type == "chord" and pitch_text:
            pitches = pitch_text.split(".")
            musical_event = chord.Chord(pitches)
        else:
            return None

        if min_duration is not None:
            duration = max(duration, min_duration)
        musical_event.duration.quarterLength = duration
        return musical_event
    except Exception:
        return None


def export_midi(
    event_tokens: list[str],
    output_path: Path,
    min_duration: float | None = None,
    tempo_bpm: float = 100,
) -> int:
    """Arrange valid events sequentially and return the number of malformed tokens skipped."""
    generated_stream = stream.Stream()
    generated_stream.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    skipped_events = 0

    for event_token in event_tokens:
        musical_event = event_token_to_music21(event_token, min_duration)
        if musical_event is None:
            skipped_events += 1
            continue
        generated_stream.append(musical_event)

    generated_stream.write("midi", fp=str(output_path))
    return skipped_events


def main() -> None:
    """Generate musical events from the trained model and export a MIDI file."""
    parser = argparse.ArgumentParser(description="Generate MIDI music with a trained LSTM.")
    parser.add_argument("--genre", default="classical", help="Genre model to use.")
    parser.add_argument("--length", type=int, default=200, help="Number of events to generate.")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature above zero."
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Minimum duration used only when reconstructing the MIDI file.",
    )
    parser.add_argument("--tempo", type=float, default=100, help="Output MIDI tempo in BPM.")
    args = parser.parse_args()

    if args.length < 1:
        parser.error("--length must be a positive integer.")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero.")
    if args.min_duration is not None and args.min_duration <= 0:
        parser.error("--min-duration must be greater than zero.")
    if args.tempo <= 0:
        parser.error("--tempo must be greater than zero.")

    rng = np.random.default_rng(args.seed)
    model, input_sequences, id_to_event, sequence_length = load_generation_artifacts(
        args.genre
    )
    seed_sequence = select_seed_sequence(input_sequences, rng)
    generated_ids = generate_event_ids(
        model, seed_sequence, args.length, args.temperature, rng
    )
    generated_tokens = [id_to_event[event_id] for event_id in generated_ids]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temperature_label = str(args.temperature).replace(".", "p")
    tempo_label = str(args.tempo).replace(".", "p")
    duration_label = (
        f"_mindur{str(args.min_duration).replace('.', 'p')}"
        if args.min_duration is not None
        else ""
    )
    seed_label = f"_seed{args.seed}" if args.seed is not None else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / (
        f"{args.genre}_generated_{args.length}_t{temperature_label}_tempo{tempo_label}"
        f"{duration_label}{seed_label}_{timestamp}.mid"
    )
    skipped_events = export_midi(
        generated_tokens, output_path, args.min_duration, args.tempo
    )

    print("Music generation summary")
    print(f"Model used: {MODELS_DIR / f'{args.genre}_lstm.keras'}")
    print(f"Seed sequence length: {sequence_length}")
    print(f"Generated events: {len(generated_tokens)}")
    print(f"Temperature: {args.temperature}")
    print(
        "Minimum reconstruction duration: "
        f"{args.min_duration if args.min_duration is not None else 'original token duration'}"
    )
    print(f"Tempo: {args.tempo} BPM")
    print(f"Random seed: {args.seed if args.seed is not None else 'not provided'}")
    print(f"Malformed events skipped: {skipped_events}")
    print(f"Output MIDI: {output_path}")


if __name__ == "__main__":
    main()
