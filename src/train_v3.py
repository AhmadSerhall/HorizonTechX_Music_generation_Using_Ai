"""Train NeuraTune's separate onset-grouped V3 LSTM from V3 artifacts."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.model_v3 import (
    DELTA_EMBEDDING_DIMENSION,
    DROPOUT_RATE,
    DURATION_EMBEDDING_DIMENSION,
    LSTM_UNITS,
    PAD_CLASS_INDEX,
    PITCH_EMBEDDING_DIMENSION,
    build_v3_lstm_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42


def load_v3_data(
    genre: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """Load V3 arrays and metadata without changing any preprocessing artifact."""
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_v3_sequences.npz"
    metadata_path = PROCESSED_DATA_DIR / f"{genre}_v3_metadata.json"
    if not sequence_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"V3 artifacts for '{genre}' were not found.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(sequence_path, allow_pickle=False) as data:
        inputs = {
            "pitch_input": np.asarray(data["X_pitches"], dtype=np.int16),
            "duration_input": np.asarray(data["X_durations"], dtype=np.int16),
            "delta_input": np.asarray(data["X_deltas"], dtype=np.int16),
        }
        targets = {
            "pitch_output": np.asarray(data["y_pitches"], dtype=np.int16),
            "duration_output": np.asarray(data["y_durations"], dtype=np.int16),
            "delta_output": np.asarray(data["y_deltas"], dtype=np.int16),
        }
        sequence_piece_ids = np.asarray(data["sequence_piece_ids"], dtype=np.int16)

    sequence_length = int(metadata["sequence_length"])
    max_pitches = int(metadata["max_pitches_per_onset"])
    sequence_count = len(inputs["pitch_input"])
    if inputs["pitch_input"].shape != (sequence_count, sequence_length, max_pitches):
        raise ValueError("V3 pitch input shape does not match metadata.")
    if inputs["duration_input"].shape != inputs["pitch_input"].shape:
        raise ValueError("V3 pitch and duration input slots are not aligned.")
    if inputs["delta_input"].shape != (sequence_count, sequence_length):
        raise ValueError("V3 delta input shape does not match metadata.")
    if targets["pitch_output"].shape != (sequence_count, max_pitches):
        raise ValueError("V3 pitch target shape does not match metadata.")
    if targets["duration_output"].shape != targets["pitch_output"].shape:
        raise ValueError("V3 pitch and duration target slots are not aligned.")
    if targets["delta_output"].shape != (sequence_count,):
        raise ValueError("V3 delta target shape does not match metadata.")
    if sequence_piece_ids.shape != (sequence_count,):
        raise ValueError("V3 sequence_piece_ids must identify every training sequence.")
    return inputs, targets, sequence_piece_ids, metadata


def mapping_with_pad(values: list[int], pad_value: int) -> np.ndarray:
    """Create an index-to-value mapping with PAD at class index zero."""
    class_values = np.asarray([pad_value, *values], dtype=np.int16)
    if len(np.unique(class_values)) != len(class_values):
        raise ValueError("PAD value must be distinct from every real class value.")
    return class_values


def encode_values(values: np.ndarray, class_values: np.ndarray) -> np.ndarray:
    """Map raw stored values to contiguous categorical indices."""
    indices = np.searchsorted(class_values, values)
    if np.any(indices >= len(class_values)) or not np.array_equal(class_values[indices], values):
        raise ValueError("An artifact value is missing from the V3 metadata mapping.")
    return np.asarray(indices, dtype=np.int32)


def encode_v3_data(
    inputs: dict[str, np.ndarray], targets: dict[str, np.ndarray], metadata: dict[str, object]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Encode raw V3 values while retaining PAD as categorical class zero."""
    pad_value = int(metadata["padding"]["value"])
    mappings = {
        "pitch": mapping_with_pad(metadata["pitch_values"], pad_value),
        "duration": mapping_with_pad(metadata["duration_step_values"], pad_value),
        "delta": np.asarray(metadata["delta_step_values"], dtype=np.int16),
    }
    encoded_inputs = {
        "pitch_input": encode_values(inputs["pitch_input"], mappings["pitch"]),
        "duration_input": encode_values(inputs["duration_input"], mappings["duration"]),
        "delta_input": encode_values(inputs["delta_input"], mappings["delta"]),
    }
    encoded_targets = {
        "pitch_output": encode_values(targets["pitch_output"], mappings["pitch"]),
        "duration_output": encode_values(targets["duration_output"], mappings["duration"]),
        "delta_output": encode_values(targets["delta_output"], mappings["delta"]),
    }
    return encoded_inputs, encoded_targets, mappings


def piece_level_split(sequence_piece_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Use the final deterministic group of source pieces for validation."""
    piece_ids = np.unique(sequence_piece_ids)
    if len(piece_ids) < 2:
        raise ValueError("At least two source pieces are needed for a piece-level split.")
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


def deterministic_subset(indices: np.ndarray, limit: int) -> np.ndarray:
    """Choose a reproducible distributed subset from one side of a split."""
    if limit >= len(indices):
        return indices
    positions = np.linspace(0, len(indices) - 1, num=limit, dtype=np.int64)
    return indices[positions]


def limit_split_indices(
    train_indices: np.ndarray, validation_indices: np.ndarray, max_sequences: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Limit smoke-test sequences while preserving the piece-disjoint split."""
    if max_sequences is None or len(train_indices) + len(validation_indices) <= max_sequences:
        return train_indices, validation_indices
    validation_limit = min(len(validation_indices), max(1, int(max_sequences * VALIDATION_FRACTION)))
    training_limit = min(len(train_indices), max_sequences - validation_limit)
    return deterministic_subset(train_indices, training_limit), deterministic_subset(
        validation_indices, validation_limit
    )


def select_rows(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    """Select matching rows from every input or target array."""
    return {name: values[indices] for name, values in data.items()}


def json_mapping(class_values: np.ndarray) -> dict[str, object]:
    """Store both class-index directions for future V3 decoding."""
    values = [int(value) for value in class_values]
    return {
        "index_to_value": values,
        "value_to_index": {str(value): index for index, value in enumerate(values)},
        "pad_class_index": PAD_CLASS_INDEX if values[0] == -1 else None,
    }


def main() -> None:
    """Train the standalone V3 onset-group LSTM."""
    parser = argparse.ArgumentParser(description="Train NeuraTune's V3 onset-group LSTM.")
    parser.add_argument("--genre", default="classical", help="V3 artifact genre to train.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--max-sequences", type=int, help="Maximum total train and validation sequences.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer.")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.max_sequences is not None and args.max_sequences < 2:
        parser.error("--max-sequences must be at least 2 when provided.")

    keras.utils.set_random_seed(RANDOM_SEED)
    raw_inputs, raw_targets, sequence_piece_ids, preprocessing_metadata = load_v3_data(args.genre)
    encoded_inputs, encoded_targets, mappings = encode_v3_data(
        raw_inputs, raw_targets, preprocessing_metadata
    )
    train_indices, validation_indices, split_metadata = piece_level_split(sequence_piece_ids)
    train_indices, validation_indices = limit_split_indices(
        train_indices, validation_indices, args.max_sequences
    )
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("The selected data must contain training and validation sequences.")

    train_inputs = select_rows(encoded_inputs, train_indices)
    validation_inputs = select_rows(encoded_inputs, validation_indices)
    train_targets = select_rows(encoded_targets, train_indices)
    validation_targets = select_rows(encoded_targets, validation_indices)
    sequence_length = int(preprocessing_metadata["sequence_length"])
    max_pitches = int(preprocessing_metadata["max_pitches_per_onset"])
    model = build_v3_lstm_model(
        sequence_length,
        max_pitches,
        len(mappings["pitch"]),
        len(mappings["duration"]),
        len(mappings["delta"]),
    )
    parameter_count = model.count_params()

    print("V3 onset-group LSTM training summary")
    print(f"Source pieces: {len(np.unique(sequence_piece_ids))}")
    print(f"Training sequences: {len(train_indices)}")
    print(f"Validation sequences: {len(validation_indices)}")
    print(f"Pitch classes including PAD: {len(mappings['pitch'])}")
    print(f"Duration classes including PAD: {len(mappings['duration'])}")
    print(f"Delta classes: {len(mappings['delta'])}")
    print(f"LSTM units: {LSTM_UNITS}")
    print(f"Total parameters: {parameter_count:,}")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        train_inputs,
        train_targets,
        validation_data=(validation_inputs, validation_targets),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )
    final_metrics = {
        name: float(values[-1]) for name, values in history.history.items() if values
    }
    if not all(math.isfinite(value) for value in final_metrics.values()):
        raise ValueError("Training produced a non-finite loss or metric.")

    validation_losses = history.history.get("val_loss", [])
    best_epoch = int(np.argmin(validation_losses) + 1) if validation_losses else None
    best_validation_loss = float(min(validation_losses)) if validation_losses else None
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{args.genre}_lstm_v3.keras"
    model.save(model_path)
    metadata_path = MODELS_DIR / f"{args.genre}_training_metadata_v3.json"
    training_metadata = {
        "architecture_version": "v3_onset_group_lstm",
        "sequence_length": sequence_length,
        "max_pitches_per_onset": max_pitches,
        "source_midi_files": preprocessing_metadata.get("source_midi_files"),
        "training_sequences_used": len(train_indices),
        "validation_sequences_used": len(validation_indices),
        "split": split_metadata,
        "class_mappings": {name: json_mapping(values) for name, values in mappings.items()},
        "class_counts": {name: len(values) for name, values in mappings.items()},
        "model": {
            "pitch_embedding_dimension": PITCH_EMBEDDING_DIMENSION,
            "duration_embedding_dimension": DURATION_EMBEDDING_DIMENSION,
            "delta_embedding_dimension": DELTA_EMBEDDING_DIMENSION,
            "lstm_units": LSTM_UNITS,
            "dropout_rate": DROPOUT_RATE,
        },
        "optimizer": "Adam",
        "loss": "masked sparse categorical crossentropy for pitch/duration; sparse categorical crossentropy for delta",
        "preprocessing_grid_quarter_length": preprocessing_metadata.get("grid_quarter_length"),
        "padding_value": preprocessing_metadata.get("padding", {}).get("value"),
        "parameter_count": parameter_count,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.epoch),
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "early_stopping_activated": len(history.epoch) < args.epochs,
        "history": {name: [float(value) for value in values] for name, values in history.history.items()},
        "final_metrics": final_metrics,
    }
    metadata_path.write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")

    print("V3 training results")
    for name, value in final_metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Training metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
