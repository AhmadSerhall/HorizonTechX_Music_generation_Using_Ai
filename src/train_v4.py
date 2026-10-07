"""Train V4 with explicit EOS targets for onset-size termination."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .model_v4 import build_v4_conditional_model

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _load_data(genre: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = PROCESSED / f"{genre}_v4_sequences.npz"
    metadata_path = PROCESSED / f"{genre}_v4_metadata.json"
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing V4 artifacts: {npz_path} and/or {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    arrays = dict(np.load(npz_path, allow_pickle=False))
    required = {"X_pitches", "X_durations", "X_deltas", "y_pitches", "y_durations", "y_deltas", "sequence_piece_ids"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"V4 NPZ is missing arrays: {sorted(missing)}")
    return arrays, metadata


def _encode(values: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    value_to_index = {int(k): int(v) for k, v in mapping["value_to_index"].items()}
    flat = values.reshape(-1)
    try:
        encoded = np.asarray([value_to_index[int(value)] for value in flat], dtype=np.int32)
    except KeyError as error:
        raise ValueError(f"Value {error.args[0]} is absent from the saved V4 mapping") from error
    return encoded.reshape(values.shape)


def _prepare_data(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int]:
    mappings = metadata["mappings"]
    x = {
        "pitch_input": _encode(arrays["X_pitches"], mappings["pitch"]),
        "duration_input": _encode(arrays["X_durations"], mappings["duration_steps"]),
        "delta_input": _encode(arrays["X_deltas"], mappings["delta_steps"]),
    }
    y_pitch = _encode(arrays["y_pitches"], mappings["pitch"])
    y_duration = _encode(arrays["y_durations"], mappings["duration_steps"])
    # EOS is appended after the final real pitch. Positions after EOS remain PAD.
    eos_index = len(mappings["pitch"]["index_to_value"])
    for row in range(y_pitch.shape[0]):
        pad_positions = np.flatnonzero(y_pitch[row] == 0)
        end = int(pad_positions[0]) if len(pad_positions) else y_pitch.shape[1]
        if end < y_pitch.shape[1]:
            y_pitch[row, end] = eos_index
            y_pitch[row, end + 1 :] = 0
            y_duration[row, end:] = 0
    y = {
        "pitch_output": y_pitch,
        "duration_output": y_duration,
        "delta_output": _encode(arrays["y_deltas"], mappings["delta_steps"]),
    }
    start_index = eos_index + 1
    decoder_input = np.zeros_like(y_pitch, dtype=np.int32)
    decoder_input[:, 0] = start_index
    decoder_input[:, 1:] = y_pitch[:, :-1]
    x["target_pitch_input"] = decoder_input
    return x, y, eos_index, start_index


def _piece_split(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    piece_ids = np.asarray(arrays["sequence_piece_ids"], dtype=np.int32)
    available = sorted(int(piece) for piece in np.unique(piece_ids))
    split_at = min(max(1, int(round(len(available) * 0.8))), len(available) - 1)
    train_pieces, validation_pieces = available[:split_at], available[split_at:]
    return (
        np.flatnonzero(np.isin(piece_ids, train_pieces)),
        np.flatnonzero(np.isin(piece_ids, validation_pieces)),
        train_pieces,
        validation_pieces,
    )


class _LearningRateRecorder(keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.values.append(float(keras.backend.get_value(self.model.optimizer.learning_rate)))


def _subset(data: dict[str, np.ndarray], indices: np.ndarray, maximum: int | None) -> dict[str, np.ndarray]:
    if maximum is not None:
        indices = indices[:maximum]
    return {key: value[indices] for key, value in data.items()}


def _rhythm_class_weights(values: np.ndarray, class_count: int, index_to_value: list[int], minimum: float = 0.5, maximum: float = 5.0) -> tuple[list[float], dict[str, Any]]:
    """Compute square-root inverse-frequency weights from training values only."""
    values = np.asarray(values, dtype=np.int32).reshape(-1)
    counts = np.bincount(values, minlength=class_count).astype(np.float64)
    observed = counts > 0
    n_valid = int(values.size)
    observed_count = int(observed.sum())
    raw = np.zeros(class_count, dtype=np.float64)
    raw[observed] = np.sqrt(n_valid / (observed_count * counts[observed]))
    weighted_mean_raw = float(np.sum(counts * raw) / max(1, n_valid))
    raw /= max(weighted_mean_raw, 1e-12)
    final = np.ones(class_count, dtype=np.float64)
    final[observed] = np.clip(raw[observed], minimum, maximum)
    weighted_mean_final = float(np.sum(counts * final) / max(1, n_valid))
    frequency = [
        {"class_index": int(index), "value": int(index_to_value[index]), "count": int(counts[index]), "percentage": float(counts[index] * 100 / max(1, n_valid)), "weight": float(final[index])}
        for index in np.flatnonzero(observed)
    ]
    frequency.sort(key=lambda item: item["count"], reverse=True)
    return final.tolist(), {
        "valid_target_count": n_valid,
        "observed_class_count": observed_count,
        "top_15": frequency[:15],
        "all_observed": frequency,
        "formula": "sqrt(N / (K * count_c)), normalized by training frequency-weighted mean, then clipped",
        "clip_min": minimum,
        "clip_max": maximum,
        "raw_weight_min": float(raw[observed].min()) if observed_count else None,
        "raw_weight_max": float(raw[observed].max()) if observed_count else None,
        "final_weight_min": float(final[observed].min()) if observed_count else None,
        "final_weight_max": float(final[observed].max()) if observed_count else None,
        "final_frequency_weighted_mean": weighted_mean_final,
        "weights": final.tolist(),
    }


def _rhythm_metrics(true: np.ndarray, probabilities: np.ndarray, index_to_value: list[int], groups: bool = False) -> dict[str, Any]:
    true = np.asarray(true, dtype=np.int32).reshape(-1)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=-1)
    classes = sorted(int(value) for value in np.unique(true))
    recalls = {}
    for class_index in classes:
        mask = true == class_index
        recalls[str(index_to_value[class_index])] = {"class_index": class_index, "count": int(mask.sum()), "recall_percent": float(np.mean(predicted[mask] == class_index) * 100)}
    grouped = {}
    if groups:
        for label, condition in [("1", true == 1), ("2", true == 2), ("3", true == 3), ("4", true == 4), ("5", true == 5), ("6_plus", true >= 6)]:
            if np.any(condition):
                grouped[label] = {"count": int(condition.sum()), "recall_percent": float(np.mean(predicted[condition] == true[condition]) * 100)}
    true_counts = {str(index_to_value[index]): int(np.sum(true == index)) for index in classes}
    predicted_counts = {str(index_to_value[index]): int(np.sum(predicted == index)) for index in sorted(np.unique(predicted))}
    return {
        "accuracy_percent": float(np.mean(predicted == true) * 100),
        "balanced_accuracy_percent": float(np.mean([item["recall_percent"] for item in recalls.values()])) if recalls else 0.0,
        "true_distribution_percent": {key: value * 100 / len(true) for key, value in true_counts.items()},
        "predicted_distribution_percent": {key: value * 100 / len(true) for key, value in predicted_counts.items()},
        "per_class_recall": recalls,
        "group_recall": grouped,
    }


def _greedy_onset_sizes(model: keras.Model, inputs: dict[str, np.ndarray], eos_index: int, start_index: int, max_slots: int = 8, max_samples: int = 5000) -> dict[str, Any]:
    """Decode pitches autoregressively, without teacher-forced target pitches."""
    sample = {key: value[:max_samples] for key, value in inputs.items() if key != "target_pitch_input"}
    input_tensors = {tensor.name.split(":")[0]: tensor for tensor in model.inputs}
    context_model = keras.Model(
        inputs=[input_tensors["pitch_input"], input_tensors["duration_input"], input_tensors["delta_input"]],
        outputs=model.get_layer("encoder_dropout").output,
    )
    context = context_model([sample["pitch_input"], sample["duration_input"], sample["delta_input"]], training=False)
    embedding = model.get_layer("target_pitch_decoder_embedding")
    decoder = model.get_layer("conditional_pitch_decoder")
    pitch_head = model.get_layer("pitch_output")
    token = np.full((len(context),), start_index, dtype=np.int32)
    sizes = np.zeros(len(context), dtype=np.int32)
    finished = np.zeros(len(context), dtype=bool)
    for _ in range(max_slots):
        decoder_input = embedding(token[:, None])
        states = decoder(decoder_input, initial_state=context, training=False)
        context = states[:, -1, :]
        probabilities = pitch_head(states, training=False)[:, 0, :]
        predicted = np.argmax(probabilities.numpy(), axis=-1).astype(np.int32)
        active = ~finished
        eos = active & (predicted == eos_index)
        pad = active & (predicted == 0)
        sizes[active & ~eos & ~pad] += 1
        finished |= eos | pad
        token = predicted
    counts = {"zero_notes": int(np.sum(sizes == 0))}
    for number in range(1, 6):
        counts[f"{number}_note" if number == 1 else f"{number}_notes"] = int(np.sum(sizes == number))
    counts["6_plus_notes"] = int(np.sum(sizes >= 6))
    total = max(1, len(sizes))
    percentages = {key: round(value * 100.0 / total, 4) for key, value in counts.items()}
    cap_hits = int(np.sum((sizes == max_slots) & ~finished))
    return {
        "sample_count": len(sizes),
        "counts": counts,
        "percentages": percentages,
        "mean": float(np.mean(sizes)) if len(sizes) else 0.0,
        "median": float(np.median(sizes)) if len(sizes) else 0.0,
        "maximum": int(np.max(sizes)) if len(sizes) else 0,
        "eos_prediction_rate_percent": round(float(np.mean(finished)) * 100.0, 4) if len(sizes) else 0.0,
        "cap_hit_rate_percent": round(cap_hits * 100.0 / total, 4),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    arrays, metadata = _load_data(args.genre)
    raw_inputs, raw_targets, eos_index, start_index = _prepare_data(arrays, metadata)
    train_indices, validation_indices, train_pieces, validation_pieces = _piece_split(arrays)
    train_inputs = _subset(raw_inputs, train_indices, args.max_sequences)
    train_targets = _subset(raw_targets, train_indices, args.max_sequences)
    validation_inputs = _subset(raw_inputs, validation_indices, args.max_sequences)
    validation_targets = _subset(raw_targets, validation_indices, args.max_sequences)
    mappings = metadata["mappings"]
    # Production V4 deliberately does not use the later V4C.2 class weights.
    delta_weight_info = {"enabled": False, "reason": "Original unweighted EOS-aware V4 objective"}
    duration_weight_info = {"enabled": False, "reason": "Original unweighted EOS-aware V4 objective"}
    model = build_v4_conditional_model(
        sequence_length=int(metadata["sequence_length"]),
        max_pitches_per_onset=int(metadata["max_pitches_per_onset"]),
        pitch_class_count=len(mappings["pitch"]["index_to_value"]),
        duration_class_count=len(mappings["duration_steps"]["index_to_value"]),
        delta_class_count=len(mappings["delta_steps"]["index_to_value"]),
        eos_pitch_class_index=eos_index,
        start_pitch_class_index=start_index,
        delta_class_weights=None,
        duration_class_weights=None,
    )
    print(model.summary())
    print(f"Pieces: {len(train_pieces)} train / {len(validation_pieces)} validation | sequences: {len(train_inputs['pitch_input'])} train / {len(validation_inputs['pitch_input'])} validation")
    print(f"Classes: pitch={len(mappings['pitch']['index_to_value'])} + EOS, duration={len(mappings['duration_steps']['index_to_value'])}, delta={len(mappings['delta_steps']['index_to_value'])} | parameters={model.count_params()}")
    lr_recorder = _LearningRateRecorder()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        lr_recorder,
    ]
    history = model.fit(train_inputs, train_targets, validation_data=(validation_inputs, validation_targets), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    history_dict = {key: [float(item) for item in values] for key, values in history.history.items()}
    best_epoch = int(np.argmin(history_dict.get("val_loss", [float("inf")])) + 1)
    best_val_loss = float(min(history_dict.get("val_loss", [float("inf")])) )
    smoke = bool(args.smoke)
    model_path = MODELS / (f"{args.genre}_lstm_v4_production_smoke.keras" if smoke else f"{args.genre}_lstm_v4_production.keras")
    metadata_path = MODELS / (f"{args.genre}_training_metadata_v4_production_smoke.json" if smoke else f"{args.genre}_training_metadata_v4_production.json")
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    reloaded = keras.models.load_model(model_path)
    validation_metrics = {key: float(value) for key, value in reloaded.evaluate(validation_inputs, validation_targets, batch_size=args.batch_size, verbose=0, return_dict=True).items()}
    validation_predictions = reloaded.predict(validation_inputs, batch_size=args.batch_size, verbose=0)
    delta_validation = _rhythm_metrics(validation_targets["delta_output"], validation_predictions["delta_output"], mappings["delta_steps"]["index_to_value"], groups=True)
    duration_mask = (validation_targets["pitch_output"] != 0) & (validation_targets["pitch_output"] != eos_index)
    duration_validation = _rhythm_metrics(validation_targets["duration_output"][duration_mask], validation_predictions["duration_output"][duration_mask], mappings["duration_steps"]["index_to_value"], groups=True)
    onset_sizes = _greedy_onset_sizes(reloaded, validation_inputs, eos_index, start_index)
    lr_changes = [{"epoch": index + 1, "learning_rate": value} for index, value in enumerate(lr_recorder.values) if index == 0 or value != lr_recorder.values[index - 1]]
    result = {
        "version": "V4_conditional_onset_grouped_EOS",
        "architecture": {"historical_pitch_embedding": 32, "historical_duration_embedding": 16, "historical_delta_embedding": 16, "encoder": "LSTM(256) -> Dropout(0.3)", "conditional_pitch_decoder": "GRU(256), teacher-forced over 8 slots with EOS", "duration_conditioning": "decoder state concatenated with pitch distribution per slot", "optimizer": "Adam"},
        "parameter_count": int(model.count_params()),
        "input_shapes": {key: list(value.shape[1:]) for key, value in raw_inputs.items()},
        "output_shapes": {key: list(value.shape[1:]) for key, value in raw_targets.items()},
        "class_counts": {"pitch_storage_including_pad": len(mappings["pitch"]["index_to_value"]), "pitch_decoder_including_eos": eos_index + 1, "duration": len(mappings["duration_steps"]["index_to_value"]), "delta": len(mappings["delta_steps"]["index_to_value"])},
        "decoder_vocabulary": {"pad": 0, "eos": eos_index, "start": start_index, "real_pitch_indices": list(range(1, eos_index))},
        "sequence_length": int(metadata["sequence_length"]), "max_pitches_per_onset": int(metadata["max_pitches_per_onset"]), "mappings": mappings,
        "train_piece_ids": train_pieces, "validation_piece_ids": validation_pieces,
        "train_sequence_count": int(len(train_inputs["pitch_input"])), "validation_sequence_count": int(len(validation_inputs["pitch_input"])),
        "epochs_requested": int(args.epochs), "epochs_completed": len(history_dict.get("loss", [])), "best_epoch": best_epoch, "best_validation_loss": best_val_loss,
        "early_stopping_activated": len(history_dict.get("loss", [])) < args.epochs, "learning_rate_history": lr_recorder.values, "learning_rate_changes": lr_changes, "history": history_dict,
        "rhythm_weighting": {"delta": delta_weight_info, "duration": duration_weight_info},
        "restored_validation_metrics": validation_metrics, "balanced_validation_metrics": {"delta": delta_validation, "duration": duration_validation},
        "greedy_validation_onset_size": onset_sizes, "smoke_test": smoke, "model_path": str(model_path),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(result), handle, indent=2)
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Best epoch/loss: {best_epoch} / {best_val_loss:.4f}")
    print(f"Restored validation metrics: {validation_metrics}")
    print(f"Balanced delta validation: {delta_validation['balanced_accuracy_percent']:.3f}%")
    print(f"Balanced duration validation: {duration_validation['balanced_accuracy_percent']:.3f}%")
    print(f"Greedy autoregressive onset sizes: {onset_sizes}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuraTune V4 with EOS onset termination.")
    parser.add_argument("--genre", default="classical")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
