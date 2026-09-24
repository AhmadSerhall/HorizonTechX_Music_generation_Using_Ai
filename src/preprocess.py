"""Convert validated MIDI files into event sequences for future model training."""

import argparse
import json
from pathlib import Path

import numpy as np
from music21 import chord, converter, note, stream

from src.dataset import PROJECT_ROOT, discover_midi_files, validate_midi_files


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def parse_midi_file(midi_file: Path) -> tuple[stream.Score | None, str | None]:
    """Parse one MIDI file without stopping the full preprocessing run on failure."""
    try:
        return converter.parse(midi_file), None
    except Exception as error:
        message = str(error).strip() or error.__class__.__name__
        return None, message


def _duration_text(quarter_length: float) -> str:
    """Quantize a duration to a quarter beat for consistent event tokens."""
    quantized_duration = max(0.25, round(float(quarter_length) * 4) / 4)
    return str(quantized_duration)


def extract_musical_events(score: stream.Score) -> list[str]:
    """Extract note and chord events with pitch content and quarter-length duration."""
    events: list[str] = []

    for element in score.flatten().notes:
        duration = _duration_text(element.duration.quarterLength)

        if isinstance(element, note.Note):
            events.append(f"note:{element.pitch.nameWithOctave}|duration:{duration}")
        elif isinstance(element, chord.Chord):
            pitches = ".".join(sorted(pitch.nameWithOctave for pitch in element.pitches))
            events.append(f"chord:{pitches}|duration:{duration}")

    return events


def preprocess_midi_files(
    midi_files: list[Path],
) -> tuple[list[str], list[Path], list[tuple[Path, str]]]:
    """Parse MIDI files and collect events while recording files that fail to parse."""
    events: list[str] = []
    processed_files: list[Path] = []
    failed_files: list[tuple[Path, str]] = []

    for midi_file in midi_files:
        score, error = parse_midi_file(midi_file)
        if score is None:
            failed_files.append((midi_file, error or "Unable to parse MIDI file."))
            continue

        events.extend(extract_musical_events(score))
        processed_files.append(midi_file)

    return events, processed_files, failed_files


def create_vocabulary(events: list[str]) -> tuple[dict[str, int], list[str]]:
    """Create stable event-to-ID and ID-to-event vocabulary mappings."""
    id_to_event = sorted(set(events))
    event_to_id = {event: event_id for event_id, event in enumerate(id_to_event)}
    return event_to_id, id_to_event


def create_training_sequences(
    events: list[str], event_to_id: dict[str, int], sequence_length: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length input sequences and their next-event targets."""
    event_ids = [event_to_id[event] for event in events]

    if len(event_ids) <= sequence_length:
        return (
            np.empty((0, sequence_length), dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    input_sequences = np.asarray(
        [event_ids[index : index + sequence_length]
         for index in range(len(event_ids) - sequence_length)],
        dtype=np.int32,
    )
    next_events = np.asarray(event_ids[sequence_length:], dtype=np.int32)
    return input_sequences, next_events


def save_processed_data(
    genre: str,
    events: list[str],
    event_to_id: dict[str, int],
    id_to_event: list[str],
    input_sequences: np.ndarray,
    next_events: np.ndarray,
    sequence_length: int,
    source_midi_files: int,
) -> tuple[Path, Path]:
    """Save lightweight data as JSON and sequences as a compressed NumPy archive."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = PROCESSED_DATA_DIR / f"{genre}_preprocessed.json"
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_sequences.npz"
    artifact = {
        "metadata": {
            "genre": genre,
            "sequence_length": sequence_length,
            "source_midi_files": source_midi_files,
            "sequence_file": sequence_path.name,
        },
        "events": events,
        "event_to_id": event_to_id,
        "id_to_event": id_to_event,
    }
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    np.savez_compressed(sequence_path, X=input_sequences, y=next_events)
    return json_path, sequence_path


def main() -> None:
    """Run MIDI preprocessing and print a concise summary."""
    parser = argparse.ArgumentParser(description="Preprocess MIDI files into event sequences.")
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
    events, processed_files, parse_failures = preprocess_midi_files(valid_files)
    event_to_id, id_to_event = create_vocabulary(events)
    input_sequences, next_events = create_training_sequences(
        events, event_to_id, args.sequence_length
    )
    json_path, sequence_path = save_processed_data(
        args.genre,
        events,
        event_to_id,
        id_to_event,
        input_sequences,
        next_events,
        args.sequence_length,
        len(processed_files),
    )

    print("MIDI preprocessing summary")
    print(f"MIDI files processed: {len(processed_files)}")
    print(f"Total musical events extracted: {len(events)}")
    print(f"Vocabulary size: {len(id_to_event)}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Training sequences: {len(input_sequences)}")
    print(f"JSON artifact: {json_path}")
    print(f"Sequence artifact: {sequence_path}")
    print(f"MIDI files skipped: {len(invalid_files) + len(parse_failures)}")


if __name__ == "__main__":
    main()
