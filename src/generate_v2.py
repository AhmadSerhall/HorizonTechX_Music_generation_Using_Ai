"""Generate timing-aware MIDI from NeuraTune's separate V2 multi-head model."""

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from music21 import note, stream, tempo
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ATTRIBUTE_NAMES = ("pitch", "delta_steps", "duration_steps")
OUTPUT_NAMES = ("pitch_output", "delta_output", "duration_output")


def load_v2_generation_artifacts(
    genre: str,
) -> tuple[keras.Model, np.ndarray, dict[str, np.ndarray], float, int]:
    """Load V2 model, raw seed sequences, saved mappings, and grid metadata."""
    model_path = MODELS_DIR / f"{genre}_lstm_v2.keras"
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_v2_sequences.npz"
    preprocessing_metadata_path = PROCESSED_DATA_DIR / f"{genre}_v2_metadata.json"
    training_metadata_path = MODELS_DIR / f"{genre}_training_metadata_v2.json"
    required_paths = (model_path, sequence_path, preprocessing_metadata_path, training_metadata_path)
    if not all(path.exists() for path in required_paths):
        raise FileNotFoundError(
            f"V2 generation artifacts for '{genre}' were not found. Complete V2 training first."
        )

    preprocessing_metadata = json.loads(
        preprocessing_metadata_path.read_text(encoding="utf-8")
    )
    training_metadata = json.loads(training_metadata_path.read_text(encoding="utf-8"))
    with np.load(sequence_path, allow_pickle=False) as data:
        input_sequences = np.asarray(data["X"], dtype=np.int32)

    sequence_length = int(training_metadata["sequence_length"])
    if input_sequences.ndim != 3 or input_sequences.shape[1:] != (sequence_length, 3):
        raise ValueError("Stored V2 sequences do not match the training metadata.")

    class_mappings = training_metadata.get("class_mappings", {})
    mappings: dict[str, np.ndarray] = {}
    for attribute_name in ATTRIBUTE_NAMES:
        mapping = class_mappings.get(attribute_name, {})
        values = np.asarray(mapping.get("index_to_value", []), dtype=np.int32)
        if len(values) == 0:
            raise ValueError(f"Missing saved {attribute_name} mapping in V2 training metadata.")
        mappings[attribute_name] = values

    model = keras.models.load_model(model_path, compile=False)
    output_sizes = {
        output_name: int(output.shape[-1])
        for output_name, output in zip(model.output_names, model.outputs)
    }
    for output_name, attribute_name in zip(OUTPUT_NAMES, ATTRIBUTE_NAMES):
        if output_sizes.get(output_name) != len(mappings[attribute_name]):
            raise ValueError(f"{output_name} does not match the saved {attribute_name} mapping.")

    grid_quarter_length = float(preprocessing_metadata["grid_quarter_length"])
    if not math.isfinite(grid_quarter_length) or grid_quarter_length <= 0:
        raise ValueError("V2 preprocessing grid must be a positive finite number.")
    return model, input_sequences, mappings, grid_quarter_length, sequence_length


def encode_seed_sequence(
    seed_sequence: np.ndarray, mappings: dict[str, np.ndarray]
) -> np.ndarray:
    """Convert one raw V2 seed sequence into the saved class-index representation."""
    encoded = np.empty_like(seed_sequence, dtype=np.int32)
    for column, attribute_name in enumerate(ATTRIBUTE_NAMES):
        class_values = mappings[attribute_name]
        indices = np.searchsorted(class_values, seed_sequence[:, column])
        if np.any(indices >= len(class_values)) or not np.array_equal(
            class_values[indices], seed_sequence[:, column]
        ):
            raise ValueError(f"Seed contains a value absent from the {attribute_name} mapping.")
        encoded[:, column] = indices
    return encoded


def sample_with_temperature_and_top_k(
    probabilities: np.ndarray,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
    allowed_indices: np.ndarray | None = None,
) -> int:
    """Temperature-scale, optionally filter classes, retain top-k, then sample."""
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if top_k < 0:
        raise ValueError("top_k must be zero or a positive integer.")

    safe_probabilities = np.nan_to_num(
        np.asarray(probabilities, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0
    )
    safe_probabilities = np.clip(safe_probabilities, 1e-12, 1.0)
    scaled_logits = np.log(safe_probabilities) / temperature
    scaled_logits -= np.max(scaled_logits)
    adjusted_probabilities = np.exp(scaled_logits)
    adjusted_probabilities /= adjusted_probabilities.sum()

    candidate_indices = (
        np.arange(len(adjusted_probabilities), dtype=np.int32)
        if allowed_indices is None
        else np.asarray(allowed_indices, dtype=np.int32)
    )
    if len(candidate_indices) == 0:
        raise ValueError("Sampling requires at least one allowed class.")
    candidate_probabilities = adjusted_probabilities[candidate_indices]
    candidate_probabilities /= candidate_probabilities.sum()

    if top_k > 0 and top_k < len(candidate_indices):
        top_positions = np.argpartition(candidate_probabilities, -top_k)[-top_k:]
        candidate_indices = candidate_indices[top_positions]
        candidate_probabilities = candidate_probabilities[top_positions]
        candidate_probabilities /= candidate_probabilities.sum()
    return int(rng.choice(candidate_indices, p=candidate_probabilities))


def model_distributions(model: keras.Model, context: np.ndarray) -> dict[str, np.ndarray]:
    """Predict the three V2 distributions for a class-index context window."""
    model_inputs = {
        "pitch_input": context[np.newaxis, :, 0],
        "delta_input": context[np.newaxis, :, 1],
        "duration_input": context[np.newaxis, :, 2],
    }
    predictions = model.predict(model_inputs, verbose=0)
    if isinstance(predictions, dict):
        return {name: np.asarray(predictions[name][0]) for name in OUTPUT_NAMES}
    return {
        name: np.asarray(prediction[0])
        for name, prediction in zip(model.output_names, predictions)
    }


def generate_v2_events(
    model: keras.Model,
    seed_sequence: np.ndarray,
    mappings: dict[str, np.ndarray],
    length: int,
    temperature: float,
    top_k: int,
    duration_temperature: float,
    duration_top_k: int,
    max_notes_per_onset: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, int]], int]:
    """Autoregressively sample raw [pitch, delta_steps, duration_steps] events."""
    context = encode_seed_sequence(seed_sequence, mappings)
    generated_events: list[tuple[int, int, int]] = []
    notes_at_current_onset = 0
    onset_cap_activations = 0
    positive_delta_indices = np.flatnonzero(mappings["delta_steps"] > 0)

    for _ in range(length):
        distributions = model_distributions(model, context)
        pitch_class_index = sample_with_temperature_and_top_k(
            distributions["pitch_output"], temperature, top_k, rng
        )
        delta_allowed_indices = None
        if notes_at_current_onset >= max_notes_per_onset:
            delta_allowed_indices = positive_delta_indices
            onset_cap_activations += 1
        delta_class_index = sample_with_temperature_and_top_k(
            distributions["delta_output"],
            temperature,
            top_k,
            rng,
            allowed_indices=delta_allowed_indices,
        )
        duration_class_index = sample_with_temperature_and_top_k(
            distributions["duration_output"],
            duration_temperature,
            duration_top_k,
            rng,
        )
        class_indices = [pitch_class_index, delta_class_index, duration_class_index]
        raw_event = tuple(
            int(mappings[attribute_name][class_index])
            for attribute_name, class_index in zip(ATTRIBUTE_NAMES, class_indices)
        )
        generated_events.append(raw_event)
        if raw_event[1] > 0:
            notes_at_current_onset = 1
        else:
            notes_at_current_onset += 1
        context = np.vstack((context[1:], np.asarray(class_indices, dtype=np.int32)))
    return generated_events, onset_cap_activations


def reconstruct_v2_midi(
    events: list[tuple[int, int, int]], grid_quarter_length: float, tempo_bpm: float
) -> tuple[stream.Stream, int, int]:
    """Place V2 notes at cumulative delta-time offsets and keep simultaneity intact."""
    generated_stream = stream.Stream()
    generated_stream.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    current_onset = 0.0
    duplicate_events_skipped = 0
    simultaneous_event_count = 0
    seen_pitch_onsets: set[tuple[int, int]] = set()

    for index, (midi_pitch, delta_steps, duration_steps) in enumerate(events):
        current_onset += delta_steps * grid_quarter_length
        if index > 0 and delta_steps == 0:
            simultaneous_event_count += 1

        pitch_onset = (midi_pitch, int(round(current_onset / grid_quarter_length)))
        if pitch_onset in seen_pitch_onsets:
            duplicate_events_skipped += 1
            continue
        seen_pitch_onsets.add(pitch_onset)

        musical_note = note.Note()
        musical_note.pitch.midi = midi_pitch
        musical_note.duration.quarterLength = duration_steps * grid_quarter_length
        generated_stream.insert(current_onset, musical_note)

    return generated_stream, duplicate_events_skipped, simultaneous_event_count


def output_filename(
    genre: str,
    length: int,
    temperature: float,
    top_k: int,
    duration_temperature: float,
    duration_top_k: int,
    max_notes_per_onset: int,
    seed: int | None,
) -> str:
    """Create a V2-specific filename that cannot be confused with V1 output."""
    temperature_label = str(temperature).replace(".", "p")
    duration_temperature_label = str(duration_temperature).replace(".", "p")
    seed_label = str(seed) if seed is not None else "random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{genre}_v2_{length}_t{temperature_label}_k{top_k}"
        f"_dt{duration_temperature_label}_dk{duration_top_k}"
        f"_cap{max_notes_per_onset}_seed{seed_label}_{timestamp}.mid"
    )


def exported_midi_statistics(
    generated_stream: stream.Stream, grid_quarter_length: float
) -> dict[str, object]:
    """Measure the timing and density of the actual exported V2 notes."""
    notes_by_onset: dict[int, list[note.Note]] = {}
    for musical_note in generated_stream.flatten().notes:
        onset_steps = int(round(float(musical_note.offset) / grid_quarter_length))
        notes_by_onset.setdefault(onset_steps, []).append(musical_note)

    onset_steps = sorted(notes_by_onset)
    note_counts = [len(notes_by_onset[onset]) for onset in onset_steps]
    duration_steps = [
        max(1, int(round(float(musical_note.duration.quarterLength) / grid_quarter_length)))
        for onset in onset_steps
        for musical_note in notes_by_onset[onset]
    ]
    reconstructed_deltas: list[int] = []
    previous_onset = 0
    for onset in onset_steps:
        reconstructed_deltas.append(onset - previous_onset)
        reconstructed_deltas.extend([0] * (len(notes_by_onset[onset]) - 1))
        previous_onset = onset

    exported_notes = sum(note_counts)
    total_duration = float(generated_stream.highestTime)
    return {
        "exported_notes": exported_notes,
        "unique_onsets": len(onset_steps),
        "average_notes_per_onset": (sum(note_counts) / len(note_counts) if note_counts else 0.0),
        "maximum_notes_per_onset": max(note_counts, default=0),
        "delta_zero_percentage": (
            sum(delta == 0 for delta in reconstructed_deltas) / len(reconstructed_deltas) * 100
            if reconstructed_deltas
            else 0.0
        ),
        "duration_steps_distribution": dict(sorted(Counter(duration_steps).items())),
        "notes_per_quarter_length": (
            exported_notes / total_duration if total_duration > 0 else 0.0
        ),
    }


def main() -> None:
    """Generate and export a V2 MIDI composition without affecting V1."""
    parser = argparse.ArgumentParser(description="Generate timing-aware MIDI with NeuraTune V2.")
    parser.add_argument("--genre", default="classical", help="V2 model genre to use.")
    parser.add_argument("--length", type=int, default=200, help="Number of events to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=10, help="Classes retained per output head; 0 disables it.")
    parser.add_argument(
        "--max-notes-per-onset",
        type=int,
        default=4,
        help="Maximum generated notes at one onset before delta sampling must advance.",
    )
    parser.add_argument(
        "--duration-top-k",
        type=int,
        default=0,
        help="Duration-head top-k; 0 reuses --top-k.",
    )
    parser.add_argument(
        "--duration-temperature",
        type=float,
        help="Duration-head temperature; defaults to --temperature.",
    )
    parser.add_argument("--seed", type=int, help="Seed-sequence and sampling random seed.")
    parser.add_argument("--tempo", type=float, default=100, help="Output MIDI tempo in BPM.")
    args = parser.parse_args()

    if args.length < 1:
        parser.error("--length must be a positive integer.")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero.")
    if args.top_k < 0:
        parser.error("--top-k must be zero or a positive integer.")
    if args.max_notes_per_onset < 1:
        parser.error("--max-notes-per-onset must be at least 1.")
    if args.duration_top_k < 0:
        parser.error("--duration-top-k must be zero or a positive integer.")
    if args.duration_temperature is not None and args.duration_temperature <= 0:
        parser.error("--duration-temperature must be greater than zero.")
    if args.tempo <= 0:
        parser.error("--tempo must be greater than zero.")

    rng = np.random.default_rng(args.seed)
    duration_temperature = (
        args.duration_temperature
        if args.duration_temperature is not None
        else args.temperature
    )
    duration_top_k = args.duration_top_k if args.duration_top_k > 0 else args.top_k
    model, input_sequences, mappings, grid_quarter_length, sequence_length = (
        load_v2_generation_artifacts(args.genre)
    )
    seed_index = int(rng.integers(len(input_sequences)))
    seed_sequence = input_sequences[seed_index].copy()
    generated_events, onset_cap_activations = generate_v2_events(
        model,
        seed_sequence,
        mappings,
        args.length,
        args.temperature,
        args.top_k,
        duration_temperature,
        duration_top_k,
        args.max_notes_per_onset,
        rng,
    )
    generated_stream, duplicate_events_skipped, simultaneous_event_count = (
        reconstruct_v2_midi(generated_events, grid_quarter_length, args.tempo)
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / output_filename(
        args.genre,
        args.length,
        args.temperature,
        args.top_k,
        duration_temperature,
        duration_top_k,
        args.max_notes_per_onset,
        args.seed,
    )
    generated_stream.write("midi", fp=str(output_path))
    total_duration = float(generated_stream.highestTime)
    exported_statistics = exported_midi_statistics(generated_stream, grid_quarter_length)
    unique_pitches = len({event[0] for event in generated_events})
    simultaneous_percentage = (
        simultaneous_event_count / len(generated_events) * 100 if generated_events else 0.0
    )

    print("V2 music generation summary")
    print(f"Output MIDI: {output_path}")
    print(f"Seed sequence length: {sequence_length}")
    print(f"Generated events: {len(generated_events)}")
    print(f"Tempo: {args.tempo} BPM")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Duration temperature: {duration_temperature}")
    print(f"Duration top-k: {duration_top_k}")
    print(f"Maximum notes per onset: {args.max_notes_per_onset}")
    print(f"Random seed: {args.seed if args.seed is not None else 'not provided'}")
    print(f"Total musical duration: {total_duration:.2f} quarter lengths")
    print(f"Unique pitches: {unique_pitches}")
    print(
        "Simultaneous events: "
        f"{simultaneous_event_count} ({simultaneous_percentage:.3f}%)"
    )
    print(f"Duplicate same-pitch/onset events skipped: {duplicate_events_skipped}")
    print(f"Exported notes: {exported_statistics['exported_notes']}")
    print(f"Unique onsets: {exported_statistics['unique_onsets']}")
    print(
        "Average notes per onset: "
        f"{exported_statistics['average_notes_per_onset']:.3f}"
    )
    print(f"Maximum notes per onset: {exported_statistics['maximum_notes_per_onset']}")
    print(f"Delta=0 percentage: {exported_statistics['delta_zero_percentage']:.3f}%")
    print(
        "Duration steps distribution: "
        f"{exported_statistics['duration_steps_distribution']}"
    )
    print(
        "Notes per quarter length: "
        f"{exported_statistics['notes_per_quarter_length']:.3f}"
    )
    print(f"Onset-cap safeguard activations: {onset_cap_activations}")


if __name__ == "__main__":
    main()
