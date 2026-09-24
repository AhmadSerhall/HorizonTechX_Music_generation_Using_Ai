"""Train the LSTM model using previously processed music-event sequences."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.model import DROPOUT_RATE, EMBEDDING_DIMENSION, LSTM_UNITS, build_lstm_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VALIDATION_FRACTION = 0.1


def load_processed_data(genre: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load event IDs and vocabulary information produced by preprocessing."""
    json_path = PROCESSED_DATA_DIR / f"{genre}_preprocessed.json"
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_sequences.npz"

    if not json_path.exists() or not sequence_path.exists():
        raise FileNotFoundError(
            f"Processed artifacts for '{genre}' were not found. Run src.preprocess first."
        )

    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    vocabulary_size = len(artifact["id_to_event"])
    sequence_length = int(artifact["metadata"]["sequence_length"])

    with np.load(sequence_path, allow_pickle=False) as sequence_data:
        input_sequences = np.asarray(sequence_data["X"], dtype=np.int32)
        next_events = np.asarray(sequence_data["y"], dtype=np.int32)

    if input_sequences.ndim != 2 or input_sequences.shape[1] != sequence_length:
        raise ValueError("X does not match the sequence length stored in metadata.")
    if len(input_sequences) != len(next_events):
        raise ValueError("X and y contain different numbers of training examples.")

    return input_sequences, next_events, vocabulary_size, sequence_length


def split_train_validation(
    input_sequences: np.ndarray, next_events: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reserve the final 10 percent of examples for validation."""
    if len(input_sequences) < 2:
        raise ValueError("At least two sequences are needed for a train/validation split.")

    validation_size = max(1, int(len(input_sequences) * VALIDATION_FRACTION))
    split_index = len(input_sequences) - validation_size
    return (
        input_sequences[:split_index],
        input_sequences[split_index:],
        next_events[:split_index],
        next_events[split_index:],
    )


def save_training_metadata(
    genre: str,
    vocabulary_size: int,
    sequence_length: int,
    training_examples: int,
    validation_examples: int,
    batch_size: int,
    epochs: int,
    parameter_count: int,
    history: dict[str, list[float]],
) -> Path:
    """Save the settings and final metrics for a completed training run."""
    metadata_path = MODELS_DIR / f"{genre}_training_metadata.json"
    metadata = {
        "vocabulary_size": vocabulary_size,
        "sequence_length": sequence_length,
        "training_examples": training_examples,
        "validation_examples": validation_examples,
        "batch_size": batch_size,
        "epochs": epochs,
        "model": {
            "embedding_dimension": EMBEDDING_DIMENSION,
            "lstm_units": LSTM_UNITS,
            "dropout_rate": DROPOUT_RATE,
        },
        "parameter_count": parameter_count,
        "final_metrics": {name: values[-1] for name, values in history.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    """Train and save a next-event LSTM model."""
    parser = argparse.ArgumentParser(description="Train an LSTM on processed music data.")
    parser.add_argument("--genre", default="classical", help="Genre artifact to train on.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument(
        "--max-sequences", type=int, help="Maximum number of sequences to use for training."
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be a positive integer.")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.max_sequences is not None and args.max_sequences < 2:
        parser.error("--max-sequences must be at least 2 when provided.")

    input_sequences, next_events, vocabulary_size, sequence_length = load_processed_data(
        args.genre
    )
    if args.max_sequences is not None:
        input_sequences = input_sequences[: args.max_sequences]
        next_events = next_events[: args.max_sequences]

    train_x, validation_x, train_y, validation_y = split_train_validation(
        input_sequences, next_events
    )
    model = build_lstm_model(vocabulary_size, sequence_length)
    parameter_count = model.count_params()

    print("LSTM training summary")
    print(f"X shape: {input_sequences.shape}")
    print(f"y shape: {next_events.shape}")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Parameter count: {parameter_count:,}")

    history = model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{args.genre}_lstm.keras"
    model.save(model_path)
    metadata_path = save_training_metadata(
        args.genre,
        vocabulary_size,
        sequence_length,
        len(train_x),
        len(validation_x),
        args.batch_size,
        args.epochs,
        parameter_count,
        history.history,
    )

    print("Training results")
    print(f"Training loss: {history.history['loss'][-1]:.4f}")
    print(f"Training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Training metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
