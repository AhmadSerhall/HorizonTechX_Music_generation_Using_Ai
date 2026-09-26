"""Train the separate multi-head LSTM for NeuraTune V2 preprocessing artifacts."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.model_v2 import (
    DELTA_EMBEDDING_DIMENSION,
    DROPOUT_RATE,
    DURATION_EMBEDDING_DIMENSION,
    LSTM_UNITS,
    PITCH_EMBEDDING_DIMENSION,
    build_v2_lstm_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VALIDATION_FRACTION = 0.1
RANDOM_SEED = 42


def load_v2_data(
    genre: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load V2 arrays and metadata without changing preprocessing artifacts."""
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_v2_sequences.npz"
    metadata_path = PROCESSED_DATA_DIR / f"{genre}_v2_metadata.json"
    if not sequence_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"V2 artifacts for '{genre}' were not found. Run src.preprocess_v2 first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(sequence_path, allow_pickle=False) as data:
        input_sequences = np.asarray(data["X"], dtype=np.int32)
        targets = np.asarray(data["y"], dtype=np.int32)
        events = np.asarray(data["events"], dtype=np.int32)
        sequence_piece_ids = np.asarray(data["sequence_piece_ids"], dtype=np.int32)

    sequence_length = int(metadata["sequence_length"])
    if input_sequences.ndim != 3 or input_sequences.shape[1:] != (sequence_length, 3):
        raise ValueError("V2 X must have shape (sequences, sequence_length, 3).")
    if targets.shape != (len(input_sequences), 3):
        raise ValueError("V2 y must contain one three-attribute target per sequence.")
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError("V2 events must have three attributes per row.")
    if sequence_piece_ids.shape != (len(input_sequences),):
        raise ValueError("V2 sequence_piece_ids must identify every training sequence.")
    return input_sequences, targets, events, sequence_piece_ids, metadata


def build_class_mappings(events: np.ndarray) -> dict[str, np.ndarray]:
    """Build sorted raw-value-to-class-index mappings for each V2 attribute."""
    return {
        "pitch": np.unique(events[:, 0]),
        "delta_steps": np.unique(events[:, 1]),
        "duration_steps": np.unique(events[:, 2]),
    }


def encode_attribute(values: np.ndarray, class_values: np.ndarray) -> np.ndarray:
    """Map raw MIDI/timing values to contiguous softmax class indices."""
    indices = np.searchsorted(class_values, values)
    if np.any(indices >= len(class_values)) or not np.array_equal(class_values[indices], values):
        raise ValueError("An input value was not found in its class mapping.")
    return np.asarray(indices, dtype=np.int32)


def encode_v2_data(
    input_sequences: np.ndarray, targets: np.ndarray, mappings: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create model input and target dictionaries using mapping-derived indices."""
    inputs = {
        "pitch_input": encode_attribute(input_sequences[:, :, 0], mappings["pitch"]),
        "delta_input": encode_attribute(
            input_sequences[:, :, 1], mappings["delta_steps"]
        ),
        "duration_input": encode_attribute(
            input_sequences[:, :, 2], mappings["duration_steps"]
        ),
    }
    encoded_targets = {
        "pitch_output": encode_attribute(targets[:, 0], mappings["pitch"]),
        "delta_output": encode_attribute(targets[:, 1], mappings["delta_steps"]),
        "duration_output": encode_attribute(
            targets[:, 2], mappings["duration_steps"]
        ),
    }
    return inputs, encoded_targets


def piece_level_split(sequence_piece_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Reserve the final deterministic set of source pieces for validation."""
    piece_ids = np.unique(sequence_piece_ids)
    if len(piece_ids) < 2:
        validation_size = max(1, int(len(sequence_piece_ids) * VALIDATION_FRACTION))
        split_index = len(sequence_piece_ids) - validation_size
        return (
            np.arange(split_index, dtype=np.int32),
            np.arange(split_index, len(sequence_piece_ids), dtype=np.int32),
            {
                "strategy": "deterministic final sequence segment",
                "limitation": "Fewer than two source pieces were available for a piece-level split.",
            },
        )

    validation_piece_count = max(1, math.ceil(len(piece_ids) * VALIDATION_FRACTION))
    validation_piece_ids = piece_ids[-validation_piece_count:]
    validation_mask = np.isin(sequence_piece_ids, validation_piece_ids)
    return (
        np.flatnonzero(~validation_mask).astype(np.int32),
        np.flatnonzero(validation_mask).astype(np.int32),
        {
            "strategy": "deterministic piece-level split",
            "training_piece_ids": [int(value) for value in piece_ids[:-validation_piece_count]],
            "validation_piece_ids": [int(value) for value in validation_piece_ids],
        },
    )


def limit_split_indices(
    train_indices: np.ndarray, validation_indices: np.ndarray, max_sequences: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Limit a smoke-test set while preserving source-piece separation."""
    if max_sequences is None or len(train_indices) + len(validation_indices) <= max_sequences:
        return train_indices, validation_indices

    validation_limit = min(len(validation_indices), max(1, int(max_sequences * VALIDATION_FRACTION)))
    training_limit = min(len(train_indices), max_sequences - validation_limit)
    remaining = max_sequences - training_limit - validation_limit
    if remaining:
        additional_training = min(remaining, len(train_indices) - training_limit)
        training_limit += additional_training
        remaining -= additional_training
    if remaining:
        validation_limit += min(remaining, len(validation_indices) - validation_limit)
    return (
        deterministic_subset(train_indices, training_limit),
        deterministic_subset(validation_indices, validation_limit),
    )


def deterministic_subset(indices: np.ndarray, limit: int) -> np.ndarray:
    """Select a reproducible, evenly distributed subset without mixing split sides."""
    if limit >= len(indices):
        return indices
    positions = np.linspace(0, len(indices) - 1, num=limit, dtype=np.int64)
    return indices[positions]


def mapping_metadata(class_values: np.ndarray) -> dict[str, object]:
    """Store both mapping directions in a compact JSON-friendly form."""
    values = [int(value) for value in class_values]
    return {
        "index_to_value": values,
        "value_to_index": {str(value): index for index, value in enumerate(values)},
    }


def final_metrics(history: keras.callbacks.History) -> dict[str, float]:
    """Return the final recorded value for each metric in a training history."""
    return {name: float(values[-1]) for name, values in history.history.items() if values}


def save_training_metadata(
    genre: str,
    metadata: dict[str, object],
) -> Path:
    """Save V2 training details separately from V1 metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = MODELS_DIR / f"{genre}_training_metadata_v2.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    """Train the V2 multi-head LSTM from existing V2 preprocessing artifacts."""
    parser = argparse.ArgumentParser(description="Train NeuraTune's V2 multi-head LSTM.")
    parser.add_argument("--genre", default="classical", help="V2 artifact genre to train.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument(
        "--max-sequences", type=int, help="Maximum total train and validation sequences."
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be a positive integer.")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.max_sequences is not None and args.max_sequences < 2:
        parser.error("--max-sequences must be at least 2 when provided.")

    keras.utils.set_random_seed(RANDOM_SEED)
    input_sequences, targets, events, sequence_piece_ids, preprocessing_metadata = load_v2_data(
        args.genre
    )
    mappings = build_class_mappings(events)
    train_indices, validation_indices, split_metadata = piece_level_split(sequence_piece_ids)
    train_indices, validation_indices = limit_split_indices(
        train_indices, validation_indices, args.max_sequences
    )
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("The selected data must contain at least one training and validation sequence.")

    train_inputs, train_targets = encode_v2_data(
        input_sequences[train_indices], targets[train_indices], mappings
    )
    validation_inputs, validation_targets = encode_v2_data(
        input_sequences[validation_indices], targets[validation_indices], mappings
    )
    sequence_length = int(preprocessing_metadata["sequence_length"])
    class_counts = {name: len(values) for name, values in mappings.items()}
    model = build_v2_lstm_model(
        sequence_length,
        class_counts["pitch"],
        class_counts["delta_steps"],
        class_counts["duration_steps"],
    )
    parameter_count = model.count_params()

    print("V2 multi-head LSTM training summary")
    print(f"Pitch classes: {class_counts['pitch']}")
    print(f"Delta-time classes: {class_counts['delta_steps']}")
    print(f"Duration classes: {class_counts['duration_steps']}")
    print(f"LSTM units: {LSTM_UNITS}")
    print(f"Total parameters: {parameter_count:,}")
    print(f"Training sequences: {len(train_indices)}")
    print(f"Validation sequences: {len(validation_indices)}")
    print(f"Split strategy: {split_metadata['strategy']}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_inputs,
        train_targets,
        validation_data=(validation_inputs, validation_targets),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    metrics = final_metrics(history)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Training produced a non-finite loss or metric.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{args.genre}_lstm_v2.keras"
    model.save(model_path)
    statistics = preprocessing_metadata.get("statistics", {})
    training_metadata = {
        "architecture_version": "v2_multihead_lstm",
        "sequence_length": sequence_length,
        "source_midi_files": preprocessing_metadata.get("source_midi_files"),
        "total_note_events": statistics.get("total_note_events"),
        "available_sequences": len(input_sequences),
        "training_sequences_used": len(train_indices),
        "validation_sequences_used": len(validation_indices),
        "split": split_metadata,
        "class_mappings": {
            name: mapping_metadata(values) for name, values in mappings.items()
        },
        "model": {
            "pitch_embedding_dimension": PITCH_EMBEDDING_DIMENSION,
            "delta_embedding_dimension": DELTA_EMBEDDING_DIMENSION,
            "duration_embedding_dimension": DURATION_EMBEDDING_DIMENSION,
            "lstm_units": LSTM_UNITS,
            "dropout_rate": DROPOUT_RATE,
        },
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy per output head",
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.epoch),
        "preprocessing_grid_quarter_length": preprocessing_metadata.get(
            "grid_quarter_length"
        ),
        "parameter_count": parameter_count,
        "final_metrics": metrics,
    }
    metadata_path = save_training_metadata(args.genre, training_metadata)

    print("V2 training results")
    for metric_name in (
        "loss",
        "val_loss",
        "pitch_output_accuracy",
        "val_pitch_output_accuracy",
        "delta_output_accuracy",
        "val_delta_output_accuracy",
        "duration_output_accuracy",
        "val_duration_output_accuracy",
    ):
        if metric_name in metrics:
            print(f"{metric_name}: {metrics[metric_name]:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Training metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
