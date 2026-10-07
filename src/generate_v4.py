"""Autoregressive V4 onset-group generation and MIDI diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from music21 import note, stream, tempo
from tensorflow import keras

# Registers the custom V4 losses/metrics before loading the saved model.
from . import model_v4  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"


def _load_artifacts(genre: str, model_path: str | None = None, training_metadata_path: str | None = None):
    npz_path = PROCESSED / f"{genre}_v4_sequences.npz"
    pre_path = PROCESSED / f"{genre}_v4_metadata.json"
    train_path = Path(training_metadata_path) if training_metadata_path else MODELS / f"{genre}_training_metadata_v4.json"
    model_file = Path(model_path) if model_path else MODELS / f"{genre}_lstm_v4.keras"
    for path in (npz_path, pre_path, train_path, model_file):
        if not path.exists():
            raise FileNotFoundError(f"Missing V4 artifact: {path}")
    with pre_path.open(encoding="utf-8") as handle:
        preprocessing = json.load(handle)
    with train_path.open(encoding="utf-8") as handle:
        training = json.load(handle)
    arrays = dict(np.load(npz_path, allow_pickle=False))
    model = keras.models.load_model(model_file)
    mappings = training.get("mappings", preprocessing.get("mappings"))
    if not mappings or "decoder_vocabulary" not in training:
        raise ValueError("V4 training metadata has no decoder mappings/vocabulary")
    decoder_vocab = training["decoder_vocabulary"]
    if int(decoder_vocab["eos"]) != len(mappings["pitch"]["index_to_value"]):
        raise ValueError("EOS index is incompatible with the saved pitch mapping")
    if int(decoder_vocab["start"]) != int(decoder_vocab["eos"]) + 1:
        raise ValueError("START/EOS decoder indices are incompatible")
    return model, arrays, preprocessing, training, mappings, decoder_vocab


def _encode(values: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    value_to_index = {int(k): int(v) for k, v in mapping["value_to_index"].items()}
    return np.asarray([value_to_index[int(v)] for v in values.reshape(-1)], dtype=np.int32).reshape(values.shape)


def _sample(probabilities: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator, forbidden: set[int] | None = None) -> int:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero")
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    logits = np.log(np.maximum(values, 1e-12)) / float(temperature)
    if forbidden:
        for index in forbidden:
            if 0 <= index < len(logits):
                logits[index] = -np.inf
    finite = np.isfinite(logits)
    if not np.any(finite):
        raise ValueError("Sampling distribution has no allowed classes")
    allowed = np.flatnonzero(finite)
    if top_k > 0 and top_k < len(allowed):
        keep = allowed[np.argpartition(logits[allowed], -top_k)[-top_k:]]
        mask = np.zeros(len(logits), dtype=bool)
        mask[keep] = True
        logits[~mask] = -np.inf
    shifted = logits - np.max(logits[np.isfinite(logits)])
    probabilities = np.exp(np.where(np.isfinite(shifted), shifted, -np.inf))
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def _layer_inputs(model: keras.Model) -> list[Any]:
    tensors = {tensor.name.split(":")[0]: tensor for tensor in model.inputs}
    return [tensors["pitch_input"], tensors["duration_input"], tensors["delta_input"]]


def generate(args: argparse.Namespace, resources: tuple[Any, ...] | None = None) -> dict[str, Any]:
    """Generate using V4; optionally reuse already-loaded resources."""
    if resources is None:
        resources = _load_artifacts(args.genre, args.model_path, args.training_metadata_path)
    model, arrays, preprocessing, training, mappings, vocabulary = resources
    rng = np.random.default_rng(args.seed)
    sequence_length = int(training["sequence_length"])
    max_slots = int(training["max_pitches_per_onset"])
    grid = float(preprocessing["grid_quarter_length"])
    pitch_map, duration_map, delta_map = mappings["pitch"], mappings["duration_steps"], mappings["delta_steps"]
    eos_index, start_index = int(vocabulary["eos"]), int(vocabulary["start"])
    seed_index = int(rng.integers(0, len(arrays["X_pitches"])))
    context_pitch = _encode(arrays["X_pitches"][seed_index], pitch_map)
    context_duration = _encode(arrays["X_durations"][seed_index], duration_map)
    context_delta = _encode(arrays["X_deltas"][seed_index], delta_map)

    input_model = keras.Model(inputs=_layer_inputs(model), outputs=model.get_layer("encoder_dropout").output)
    embedding = model.get_layer("target_pitch_decoder_embedding")
    decoder = model.get_layer("conditional_pitch_decoder")
    pitch_head = model.get_layer("pitch_output")
    duration_head = model.get_layer("duration_output")
    delta_head = model.get_layer("delta_output")
    real_pitch_forbidden = {0, eos_index}
    duration_forbidden = {0}
    generated: list[dict[str, Any]] = []
    current_onset = 0.0
    first_eos_resamples = 0

    for _ in range(int(args.length)):
        context = input_model([context_pitch[None, :], context_duration[None, :], context_delta[None, :]], training=False)
        delta_probabilities = delta_head(context, training=False).numpy()[0]
        delta_class = _sample(delta_probabilities, args.delta_temperature, args.top_k, rng)
        delta_steps = int(delta_map["index_to_value"][delta_class])
        current_onset += delta_steps * grid
        decoder_state = context
        token = start_index
        pitches: list[int] = []
        durations: list[int] = []
        eos_terminated = False

        for slot in range(max_slots):
            decoder_input = embedding(np.asarray([[token]], dtype=np.int32))
            decoder_sequence = decoder(decoder_input, initial_state=decoder_state, training=False)
            decoder_state = decoder_sequence[:, -1, :]
            pitch_probabilities = pitch_head(decoder_sequence, training=False).numpy()[0, 0]
            # Allow EOS to be selected at slot zero so its explicit-resample
            # frequency is measured; an empty onset is never emitted.
            forbidden = {0}
            pitch_class = _sample(pitch_probabilities, args.temperature, args.top_k, rng, forbidden)
            if pitch_class == eos_index:
                if slot == 0:
                    first_eos_resamples += 1
                    # The first pitch cannot be EOS; sample again from real pitches.
                    pitch_class = _sample(pitch_probabilities, args.temperature, args.top_k, rng, real_pitch_forbidden)
                else:
                    eos_terminated = True
                    break
            pitch_value = int(pitch_map["index_to_value"][pitch_class])
            duration_features = keras.layers.Concatenate(axis=-1)([decoder_sequence, pitch_head(decoder_sequence, training=False)])
            duration_probabilities = duration_head(duration_features, training=False).numpy()[0, 0]
            duration_class = _sample(duration_probabilities, args.duration_temperature, args.top_k, rng, duration_forbidden)
            duration_steps = max(1, int(duration_map["index_to_value"][duration_class]))
            pitches.append(pitch_value)
            durations.append(duration_steps)
            token = pitch_class
        if not pitches:
            # This is only a defensive fallback for a malformed distribution.
            continue
        generated.append({"onset": current_onset, "delta_steps": delta_steps, "pitches": pitches, "durations": durations, "eos": eos_terminated, "cap": len(pitches) == max_slots and not eos_terminated})
        next_pitch = np.zeros(max_slots, dtype=np.int32)
        next_duration = np.zeros(max_slots, dtype=np.int32)
        next_pitch[: len(pitches)] = [pitch_map["value_to_index"][str(p)] for p in pitches]
        next_duration[: len(durations)] = [duration_map["value_to_index"][str(d)] for d in durations]
        context_pitch = np.concatenate([context_pitch[1:], next_pitch[None, :]], axis=0)
        context_duration = np.concatenate([context_duration[1:], next_duration[None, :]], axis=0)
        context_delta = np.concatenate([context_delta[1:], np.asarray([delta_class], dtype=np.int32)])

    midi_path = _write_midi(generated, args, current_onset)
    diagnostics = _diagnostics(generated, first_eos_resamples, args.length, grid)
    _print_report(midi_path, args, seed_index, diagnostics)
    return {"path": str(midi_path), "seed_index": seed_index, "diagnostics": diagnostics}


def _write_midi(groups: list[dict[str, Any]], args: argparse.Namespace, duration: float) -> Path:
    music = stream.Stream()
    music.insert(0, tempo.MetronomeMark(number=float(args.tempo)))
    for group in groups:
        for pitch_value, duration_steps in zip(group["pitches"], group["durations"]):
            item = note.Note(midi=int(pitch_value))
            item.duration.quarterLength = float(duration_steps) * 0.25
            music.insert(float(group["onset"]), item)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp = str(args.temperature).replace(".", "p")
    filename = f"{args.genre}_v4_{args.length}_t{temp}_k{args.top_k}_seed{args.seed}_{stamp}.mid"
    path = Path(args.output_path) if args.output_path else OUTPUTS / filename
    music.write("midi", fp=str(path))
    return path


def _diagnostics(groups: list[dict[str, Any]], first_eos_resamples: int, requested: int, grid: float) -> dict[str, Any]:
    sizes = np.asarray([len(g["pitches"]) for g in groups], dtype=np.int32)
    deltas = np.asarray([g["delta_steps"] for g in groups], dtype=np.int32)
    durations = np.asarray([d for g in groups for d in g["durations"]], dtype=np.int32)
    all_pitches = np.asarray([p for g in groups for p in g["pitches"]], dtype=np.int32)
    total = max(1, len(groups))
    size_counts = {f"{i}_note" if i == 1 else f"{i}_notes": int(np.sum(sizes == i)) for i in range(1, 6)}
    size_counts["6_plus_notes"] = int(np.sum(sizes >= 6))
    size_percent = {key: round(value * 100 / total, 4) for key, value in size_counts.items()}
    pitch_sets = [tuple(sorted(g["pitches"])) for g in groups]
    melody = [max(pitches) for pitches in pitch_sets if pitches]
    movements = np.abs(np.diff(melody)) if len(melody) > 1 else np.asarray([], dtype=np.int32)
    movement_percent = {
        "repeated_pitch": round(float(np.mean(movements == 0)) * 100, 4) if len(movements) else 0.0,
        "1_2_semitones": round(float(np.mean((movements >= 1) & (movements <= 2))) * 100, 4) if len(movements) else 0.0,
        "3_5_semitones": round(float(np.mean((movements >= 3) & (movements <= 5))) * 100, 4) if len(movements) else 0.0,
        "6_12_semitones": round(float(np.mean((movements >= 6) & (movements <= 12))) * 100, 4) if len(movements) else 0.0,
        "over_12_semitones": round(float(np.mean(movements > 12)) * 100, 4) if len(movements) else 0.0,
    }
    multi = [pitches for pitches in pitch_sets if len(pitches) > 1]
    intervals = [b - a for pitches in multi for i, a in enumerate(pitches) for b in pitches[i + 1 :]]
    harmony = {
        "multi_note_onsets": len(multi),
        "one_semitone_clash_rate_percent": round(sum(1 for p in multi if any(b - a == 1 for i, a in enumerate(p) for b in p[i + 1 :])) * 100 / max(1, len(multi)), 4),
        "two_semitone_interval_rate_percent": round(sum(1 for p in multi if any(b - a == 2 for i, a in enumerate(p) for b in p[i + 1 :])) * 100 / max(1, len(multi)), 4),
        "average_pitches_per_multi_note_onset": round(float(np.mean([len(p) for p in multi])) if multi else 0.0, 4),
        "pitch_span_mean": round(float(np.mean([max(p) - min(p) for p in multi])) if multi else 0.0, 4),
        "pitch_span_median": float(np.median([max(p) - min(p) for p in multi])) if multi else 0.0,
        "interval_counts": dict(Counter(intervals)),
    }
    return {
        "onset_groups": len(groups), "total_notes": int(len(all_pitches)), "mean_notes_per_onset": round(float(np.mean(sizes)) if len(sizes) else 0.0, 4),
        "onset_size_distribution_percent": size_percent, "onset_size_counts": size_counts,
        "eos_termination_rate_percent": round(sum(bool(g["eos"]) for g in groups) * 100 / total, 4), "cap_rate_percent": round(sum(bool(g["cap"]) for g in groups) * 100 / total, 4), "first_position_eos_resamples": first_eos_resamples,
        "delta_one_percent": round(float(np.mean(deltas == 1)) * 100 if len(deltas) else 0.0, 4), "duration_one_percent": round(float(np.mean(durations == 1)) * 100 if len(durations) else 0.0, 4), "mean_delta_steps": round(float(np.mean(deltas)) if len(deltas) else 0.0, 4), "mean_duration_steps": round(float(np.mean(durations)) if len(durations) else 0.0, 4), "unique_delta_values": sorted(set(deltas.tolist())), "unique_duration_values": sorted(set(durations.tolist())),
        "pitch_mean": round(float(np.mean(all_pitches)) if len(all_pitches) else 0.0, 4), "pitch_median": float(np.median(all_pitches)) if len(all_pitches) else 0.0, "pitch_min": int(np.min(all_pitches)) if len(all_pitches) else None, "pitch_max": int(np.max(all_pitches)) if len(all_pitches) else None, "pitch_p05": float(np.percentile(all_pitches, 5)) if len(all_pitches) else None, "pitch_p95": float(np.percentile(all_pitches, 95)) if len(all_pitches) else None, "unique_pitches": int(len(set(all_pitches.tolist()))),
        "repeated_onset_pitch_set_rate_percent": round(sum(a == b for a, b in zip(pitch_sets, pitch_sets[1:])) * 100 / max(1, len(pitch_sets) - 1), 4), "repeated_melody_proxy_rate_percent": movement_percent["repeated_pitch"], "melody_mean_absolute_movement": round(float(np.mean(movements)) if len(movements) else 0.0, 4), "melody_median_movement": float(np.median(movements)) if len(movements) else 0.0, "melody_movement_percent": movement_percent,
        "harmony": harmony, "notes_per_quarter_length": round(len(all_pitches) / max(grid * float(np.sum(deltas)), 1e-9), 4),
    }


def _print_report(path: Path, args: argparse.Namespace, seed_index: int, diagnostics: dict[str, Any]) -> None:
    print(f"Output MIDI: {path}")
    print(f"Seed sequence: {seed_index} | events requested/generated: {args.length}/{diagnostics['onset_groups']} | temperature={args.temperature} top-k={args.top_k} tempo={args.tempo}")
    for key in ("total_notes", "mean_notes_per_onset", "onset_size_distribution_percent", "eos_termination_rate_percent", "cap_rate_percent", "first_position_eos_resamples", "delta_one_percent", "duration_one_percent", "pitch_mean", "pitch_median", "pitch_min", "pitch_max", "unique_pitches", "repeated_onset_pitch_set_rate_percent", "repeated_melody_proxy_rate_percent", "melody_mean_absolute_movement", "melody_movement_percent"):
        print(f"{key}: {diagnostics[key]}")
    print(f"harmony: {diagnostics['harmony']}")


def compare_training(diagnostics: dict[str, Any], genre: str = "classical") -> None:
    path = PROCESSED / f"{genre}_v4_diagnostic.json"
    if not path.exists():
        print("Training reference unavailable")
        return
    with path.open(encoding="utf-8") as handle:
        aggregate = json.load(handle)["aggregate"]
    ref_sizes = aggregate["notes_per_onset"]["percentages"]
    print("Training reference comparison:")
    print(f"mean notes/onset: generated={diagnostics['mean_notes_per_onset']:.3f} | training={aggregate['notes_per_onset']['statistics']['mean']:.3f}")
    print(f"onset sizes: generated={diagnostics['onset_size_distribution_percent']} | training={ref_sizes}")
    print(f"delta=1: generated={diagnostics['delta_one_percent']:.2f}% | training={aggregate['rhythm']['delta_steps']['percentage_delta_one_step']:.2f}%")
    print(f"duration=1: generated={diagnostics['duration_one_percent']:.2f}% | training={aggregate['rhythm']['duration_steps']['percentage_duration_one_step']:.2f}%")
    print(f"pitch mean/median/P5/P95: generated={diagnostics['pitch_mean']:.2f}/{diagnostics['pitch_median']}/{diagnostics['pitch_p05']:.1f}/{diagnostics['pitch_p95']:.1f} | training={aggregate['pitch_register']['mean']:.2f}/{aggregate['pitch_register']['median']}/{aggregate['pitch_register']['p05']}/{aggregate['pitch_register']['p95']}")
    print(f"repeated melody: generated={diagnostics['repeated_melody_proxy_rate_percent']:.2f}% | training={aggregate['repetition']['repeated_melody_proxy_rate_percentage']:.2f}%")
    print(f">12 semitone movement: generated={diagnostics['melody_movement_percent']['over_12_semitones']:.2f}% | training={aggregate['melody_proxy']['movement_percentages']['over_12_semitones']:.2f}%")
    print(f"1-semitone harmony clash: generated={diagnostics['harmony']['one_semitone_clash_rate_percent']:.2f}% | training={aggregate['simultaneous_harmony']['percentage_with_1_semitone_interval']:.2f}%")
    print(f"2-semitone harmony interval: generated={diagnostics['harmony']['two_semitone_interval_rate_percent']:.2f}% | training={aggregate['simultaneous_harmony']['percentage_with_2_semitone_interval']:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and diagnose V4 MIDI.")
    parser.add_argument("--genre", default="classical")
    parser.add_argument("--length", type=int, default=200, help="Number of new onset groups")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--delta-temperature", type=float, default=0.8)
    parser.add_argument("--duration-temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tempo", type=float, default=100)
    parser.add_argument("--model-path", default=None, help="Optional V4 model checkpoint")
    parser.add_argument("--training-metadata-path", default=None, help="Optional V4 training metadata")
    parser.add_argument("--output-path", default=None, help="Optional exact MIDI output path")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = generate(arguments)
    compare_training(result["diagnostics"], arguments.genre)
