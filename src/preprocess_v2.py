"""Create timing-aware V2 MIDI preprocessing artifacts without replacing V1."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from music21 import chord, converter, note, stream

from src.dataset import PROJECT_ROOT, discover_midi_files, validate_midi_files


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_GRID_QUARTER_LENGTH = 0.25
FEATURE_NAMES = ["pitch", "delta_steps", "duration_steps"]


def parse_midi_file(midi_file: Path) -> tuple[stream.Score | None, str | None]:
    """Parse one MIDI file while allowing preprocessing to continue on errors."""
    try:
        return converter.parse(midi_file), None
    except Exception as error:
        message = str(error).strip() or error.__class__.__name__
        return None, message


def quantize_to_steps(
    value: float, grid_quarter_length: float, minimum_steps: int = 0
) -> int:
    """Convert a quarter-length value to an integer number of grid steps."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("Timing values must be finite.")
    return max(minimum_steps, int(round(numeric_value / grid_quarter_length)))


def extract_timed_note_events(
    score: stream.Score, grid_quarter_length: float = DEFAULT_GRID_QUARTER_LENGTH
) -> tuple[list[tuple[int, int, int]], int]:
    """Return quantized (pitch, onset_steps, duration_steps) events from one score.

    Chords become one event per component pitch at the same onset. Only exact
    pitch/onset/duration duplicates are removed, so different chord pitches and
    same-pitch notes with different durations remain available to future models.
    """
    candidates: list[tuple[int, int, int]] = []

    for element in score.flatten().notes:
        try:
            onset_steps = quantize_to_steps(
                element.offset, grid_quarter_length, minimum_steps=0
            )
            duration_steps = quantize_to_steps(
                element.duration.quarterLength,
                grid_quarter_length,
                minimum_steps=1,
            )
            if isinstance(element, note.Note):
                pitches = [element.pitch]
            elif isinstance(element, chord.Chord):
                pitches = list(element.pitches)
            else:
                continue

            for pitch in pitches:
                midi_pitch = int(pitch.midi)
                if 0 <= midi_pitch <= 127:
                    candidates.append((midi_pitch, onset_steps, duration_steps))
        except (AttributeError, TypeError, ValueError):
            # Ignore an individual malformed event without discarding its score.
            continue

    candidates.sort(key=lambda event: (event[1], event[0], event[2]))
    unique_events: list[tuple[int, int, int]] = []
    seen_events: set[tuple[int, int, int]] = set()
    duplicates_removed = 0
    for event in candidates:
        if event in seen_events:
            duplicates_removed += 1
            continue
        seen_events.add(event)
        unique_events.append(event)
    return unique_events, duplicates_removed


def events_to_features(
    timed_events: list[tuple[int, int, int]]
) -> tuple[np.ndarray, int]:
    """Encode one piece as [pitch, delta_steps, duration_steps] rows.

    Summing delta_steps from the first event reconstructs each quantized onset.
    A zero delta after the first event therefore represents simultaneity.
    """
    features = np.empty((len(timed_events), len(FEATURE_NAMES)), dtype=np.int32)
    previous_onset = 0
    simultaneous_events = 0

    for index, (pitch, onset_steps, duration_steps) in enumerate(timed_events):
        delta_steps = onset_steps if index == 0 else onset_steps - previous_onset
        if index > 0 and delta_steps == 0:
            simultaneous_events += 1
        features[index] = (pitch, delta_steps, duration_steps)
        previous_onset = onset_steps
    return features, simultaneous_events


def process_midi_files(
    midi_files: list[Path], grid_quarter_length: float
) -> tuple[list[tuple[Path, np.ndarray, int]], list[tuple[Path, str]]]:
    """Process each piece independently and retain per-piece boundaries."""
    processed_pieces: list[tuple[Path, np.ndarray, int]] = []
    failed_files: list[tuple[Path, str]] = []

    for midi_file in midi_files:
        score, error = parse_midi_file(midi_file)
        if score is None:
            failed_files.append((midi_file, error or "Unable to parse MIDI file."))
            continue

        timed_events, duplicates_removed = extract_timed_note_events(
            score, grid_quarter_length
        )
        features, _ = events_to_features(timed_events)
        processed_pieces.append((midi_file, features, duplicates_removed))

    return processed_pieces, failed_files


def create_piece_sequences(
    pieces: list[tuple[Path, np.ndarray, int]], sequence_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Create V2 sequences within each piece, never across source boundaries."""
    sequence_count = sum(
        max(0, len(piece_features) - sequence_length)
        for _, piece_features, _ in pieces
    )
    input_sequences = np.empty(
        (sequence_count, sequence_length, len(FEATURE_NAMES)), dtype=np.int32
    )
    targets = np.empty((sequence_count, len(FEATURE_NAMES)), dtype=np.int32)
    sequence_piece_ids = np.empty(sequence_count, dtype=np.int32)

    all_events = [piece_features for _, piece_features, _ in pieces]
    events = (
        np.concatenate(all_events, axis=0)
        if all_events
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.int32)
    )
    piece_offsets = [0]
    for _, piece_features, _ in pieces:
        piece_offsets.append(piece_offsets[-1] + len(piece_features))

    sequence_index = 0
    for piece_id, (_, piece_features, _) in enumerate(pieces):
        piece_sequence_count = max(0, len(piece_features) - sequence_length)
        for start_index in range(piece_sequence_count):
            input_sequences[sequence_index] = piece_features[
                start_index : start_index + sequence_length
            ]
            targets[sequence_index] = piece_features[start_index + sequence_length]
            sequence_piece_ids[sequence_index] = piece_id
            sequence_index += 1

    return (
        input_sequences,
        targets,
        events,
        np.asarray(piece_offsets, dtype=np.int32),
        sequence_piece_ids,
    )


def relative_path(path: Path) -> str:
    """Return a stable project-relative source path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_v1_vocabulary_size(genre: str) -> int | None:
    """Read V1 size for a diagnostic comparison without changing V1 artifacts."""
    v1_metadata_path = PROCESSED_DATA_DIR / f"{genre}_preprocessed.json"
    try:
        artifact = json.loads(v1_metadata_path.read_text(encoding="utf-8"))
        return len(artifact.get("id_to_event", []))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def save_v2_artifacts(
    genre: str,
    input_sequences: np.ndarray,
    targets: np.ndarray,
    events: np.ndarray,
    piece_offsets: np.ndarray,
    sequence_piece_ids: np.ndarray,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    """Save numeric V2 arrays separately from lightweight metadata."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_v2_sequences.npz"
    metadata_path = PROCESSED_DATA_DIR / f"{genre}_v2_metadata.json"
    np.savez_compressed(
        sequence_path,
        X=input_sequences,
        y=targets,
        events=events,
        piece_offsets=piece_offsets,
        sequence_piece_ids=sequence_piece_ids,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return sequence_path, metadata_path


def main() -> None:
    """Run V2 preprocessing without changing existing V1 artifacts."""
    parser = argparse.ArgumentParser(description="Create timing-aware V2 MIDI artifacts.")
    parser.add_argument("--genre", default="classical", help="Genre folder to preprocess.")
    parser.add_argument("--limit", type=int, help="Maximum MIDI files to preprocess.")
    parser.add_argument(
        "--sequence-length", type=int, default=50, help="Events per input sequence."
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")
    if args.sequence_length < 1:
        parser.error("--sequence-length must be a positive integer.")

    discovered_files = discover_midi_files(genre=args.genre)
    selected_files = (
        discovered_files[: args.limit] if args.limit is not None else discovered_files
    )
    valid_files, invalid_files = validate_midi_files(selected_files)
    pieces, parse_failures = process_midi_files(valid_files, DEFAULT_GRID_QUARTER_LENGTH)
    input_sequences, targets, events, piece_offsets, sequence_piece_ids = (
        create_piece_sequences(pieces, args.sequence_length)
    )

    duration_steps = events[:, 2] if len(events) else np.empty(0, dtype=np.int32)
    delta_steps = events[:, 1] if len(events) else np.empty(0, dtype=np.int32)
    simultaneous_event_count = 0
    source_pieces = []
    for piece_id, (source_path, piece_features, duplicates_removed) in enumerate(pieces):
        piece_simultaneous_events = int(
            np.count_nonzero(piece_features[1:, 1] == 0)
        )
        simultaneous_event_count += piece_simultaneous_events
        source_pieces.append(
            {
                "piece_id": piece_id,
                "source_path": relative_path(source_path),
                "note_events": int(len(piece_features)),
                "duplicates_removed": duplicates_removed,
                "simultaneous_events": piece_simultaneous_events,
            }
        )

    v1_vocabulary_size = load_v1_vocabulary_size(args.genre)
    metadata: dict[str, object] = {
        "representation_version": "v2",
        "genre": args.genre,
        "grid_quarter_length": DEFAULT_GRID_QUARTER_LENGTH,
        "feature_names": FEATURE_NAMES,
        "feature_description": {
            "pitch": "MIDI pitch integer from 0 to 127.",
            "delta_steps": "Quantized time since the previous onset in this piece.",
            "duration_steps": "Quantized note duration; multiply steps by grid_quarter_length.",
        },
        "sequence_length": args.sequence_length,
        "source_midi_files": len(pieces),
        "source_pieces": source_pieces,
        "statistics": {
            "total_note_events": int(len(events)),
            "unique_midi_pitches": int(len(np.unique(events[:, 0]))) if len(events) else 0,
            "unique_duration_values": int(len(np.unique(duration_steps))),
            "unique_delta_time_values": int(len(np.unique(delta_steps))),
            "simultaneous_event_count": simultaneous_event_count,
            "simultaneous_event_percentage": (
                round(simultaneous_event_count / len(events) * 100, 3) if len(events) else 0.0
            ),
            "duplicates_removed": int(sum(piece[2] for piece in pieces)),
            "training_sequences": int(len(input_sequences)),
        },
        "v1_combined_token_vocabulary_size": v1_vocabulary_size,
    }
    sequence_path, metadata_path = save_v2_artifacts(
        args.genre,
        input_sequences,
        targets,
        events,
        piece_offsets,
        sequence_piece_ids,
        metadata,
    )

    statistics = metadata["statistics"]
    print("V2 MIDI preprocessing summary")
    print(f"MIDI files processed: {len(pieces)}")
    print(f"MIDI files skipped: {len(invalid_files) + len(parse_failures)}")
    print(f"Total note events: {statistics['total_note_events']}")
    print(f"Unique MIDI pitches: {statistics['unique_midi_pitches']}")
    print(f"Unique quantized durations: {statistics['unique_duration_values']}")
    print(f"Unique quantized delta-times: {statistics['unique_delta_time_values']}")
    print(
        "Simultaneous events: "
        f"{statistics['simultaneous_event_count']} "
        f"({statistics['simultaneous_event_percentage']}%)"
    )
    print(f"Training sequences: {statistics['training_sequences']}")
    print(f"Sequence length: {args.sequence_length}")
    if v1_vocabulary_size is not None:
        print(f"V1 combined-token vocabulary: {v1_vocabulary_size}")
    print(
        "V2 independent attribute spaces: "
        f"pitch={statistics['unique_midi_pitches']}, "
        f"duration={statistics['unique_duration_values']}, "
        f"delta-time={statistics['unique_delta_time_values']}"
    )
    print(f"Sequence artifact: {sequence_path}")
    print(f"Metadata artifact: {metadata_path}")


if __name__ == "__main__":
    main()
