"""Create onset-grouped V3 MIDI preprocessing artifacts without changing V1 or V2."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from music21 import chord, converter, note, stream

from src.dataset import PROJECT_ROOT, discover_midi_files, validate_midi_files


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_GRID_QUARTER_LENGTH = 0.25
MAX_PITCHES_PER_ONSET = 8
PAD_VALUE = -1


def parse_midi_file(midi_file: Path) -> tuple[stream.Score | None, str | None]:
    """Parse one MIDI file without stopping preprocessing on an unreadable score."""
    try:
        return converter.parse(midi_file), None
    except Exception as error:
        message = str(error).strip() or error.__class__.__name__
        return None, message


def quantize_to_steps(
    value: float, grid_quarter_length: float, minimum_steps: int = 0
) -> int:
    """Convert quarter lengths to stable integer grid steps."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("Timing values must be finite.")
    return max(minimum_steps, int(round(numeric_value / grid_quarter_length)))


def extract_note_records(
    score: stream.Score, grid_quarter_length: float = DEFAULT_GRID_QUARTER_LENGTH
) -> tuple[list[tuple[int, int, int]], int]:
    """Extract deduplicated (pitch, onset_steps, duration_steps) note records."""
    records: list[tuple[int, int, int]] = []
    for element in score.flatten().notes:
        try:
            onset_steps = quantize_to_steps(element.offset, grid_quarter_length)
            duration_steps = quantize_to_steps(
                element.duration.quarterLength, grid_quarter_length, minimum_steps=1
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
                    records.append((midi_pitch, onset_steps, duration_steps))
        except (AttributeError, TypeError, ValueError):
            continue

    records.sort(key=lambda record: (record[1], record[0], record[2]))
    unique_records: list[tuple[int, int, int]] = []
    seen_records: set[tuple[int, int, int]] = set()
    duplicates_removed = 0
    for record in records:
        if record in seen_records:
            duplicates_removed += 1
            continue
        seen_records.add(record)
        unique_records.append(record)
    return unique_records, duplicates_removed


def build_onset_groups(
    records: list[tuple[int, int, int]], max_pitches_per_onset: int = MAX_PITCHES_PER_ONSET
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Group simultaneous records into padded pitch/duration slots by onset.

    Groups are ordered by quantized onset. Within an onset, records are ordered
    by ascending MIDI pitch and then duration. Groups larger than the slot limit
    retain this deterministic first portion while recording the omitted notes.
    """
    records_by_onset: dict[int, list[tuple[int, int]]] = {}
    for pitch, onset_steps, duration_steps in records:
        records_by_onset.setdefault(onset_steps, []).append((pitch, duration_steps))

    onset_steps_list = sorted(records_by_onset)
    pitches = np.full(
        (len(onset_steps_list), max_pitches_per_onset), PAD_VALUE, dtype=np.int16
    )
    durations = np.full_like(pitches, PAD_VALUE)
    deltas = np.empty(len(onset_steps_list), dtype=np.int16)
    previous_onset = 0
    group_sizes: list[int] = []
    overflow_groups = 0
    truncated_notes = 0

    for index, onset_steps in enumerate(onset_steps_list):
        ordered_records = sorted(records_by_onset[onset_steps], key=lambda item: (item[0], item[1]))
        group_sizes.append(len(ordered_records))
        retained_records = ordered_records[:max_pitches_per_onset]
        if len(ordered_records) > max_pitches_per_onset:
            overflow_groups += 1
            truncated_notes += len(ordered_records) - max_pitches_per_onset
        for slot, (pitch, duration_steps) in enumerate(retained_records):
            pitches[index, slot] = pitch
            durations[index, slot] = duration_steps
        deltas[index] = onset_steps if index == 0 else onset_steps - previous_onset
        previous_onset = onset_steps

    statistics = {
        "onset_groups": len(onset_steps_list),
        "overflow_groups": overflow_groups,
        "truncated_notes": truncated_notes,
        "maximum_group_size": max(group_sizes, default=0),
        "single_pitch_groups": sum(size == 1 for size in group_sizes),
        "two_to_three_pitch_groups": sum(2 <= size <= 3 for size in group_sizes),
        "four_to_five_pitch_groups": sum(4 <= size <= 5 for size in group_sizes),
        "six_or_more_pitch_groups": sum(size >= 6 for size in group_sizes),
        "total_group_pitches": sum(group_sizes),
        "group_size_counts": {
            str(size): group_sizes.count(size) for size in sorted(set(group_sizes))
        },
    }
    return pitches, durations, deltas, statistics


def process_midi_files(
    midi_files: list[Path], grid_quarter_length: float
) -> tuple[
    list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]],
    list[tuple[Path, str]],
]:
    """Process pieces independently so V3 groups and sequences retain boundaries."""
    pieces = []
    failed_files: list[tuple[Path, str]] = []
    for midi_file in midi_files:
        score, error = parse_midi_file(midi_file)
        if score is None:
            failed_files.append((midi_file, error or "Unable to parse MIDI file."))
            continue
        records, duplicates_removed = extract_note_records(score, grid_quarter_length)
        pitches, durations, deltas, statistics = build_onset_groups(records)
        statistics["note_events"] = len(records)
        statistics["duplicates_removed"] = duplicates_removed
        pieces.append((midi_file, pitches, durations, deltas, statistics))
    return pieces, failed_files


def create_piece_sequences(
    pieces: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]],
    sequence_length: int,
) -> tuple[np.ndarray, ...]:
    """Create 50-onset-group windows without ever crossing a source-piece boundary."""
    sequence_count = sum(max(0, len(pitches) - sequence_length) for _, pitches, *_ in pieces)
    slot_count = MAX_PITCHES_PER_ONSET
    x_pitches = np.empty((sequence_count, sequence_length, slot_count), dtype=np.int16)
    x_durations = np.empty_like(x_pitches)
    x_deltas = np.empty((sequence_count, sequence_length), dtype=np.int16)
    y_pitches = np.empty((sequence_count, slot_count), dtype=np.int16)
    y_durations = np.empty_like(y_pitches)
    y_deltas = np.empty(sequence_count, dtype=np.int16)
    sequence_piece_ids = np.empty(sequence_count, dtype=np.int16)

    all_pitches = [pitches for _, pitches, *_ in pieces]
    all_durations = [durations for _, _, durations, *_ in pieces]
    all_deltas = [deltas for _, _, _, deltas, _ in pieces]
    group_pitches = (
        np.concatenate(all_pitches, axis=0)
        if all_pitches
        else np.empty((0, slot_count), dtype=np.int16)
    )
    group_durations = (
        np.concatenate(all_durations, axis=0)
        if all_durations
        else np.empty((0, slot_count), dtype=np.int16)
    )
    group_deltas = (
        np.concatenate(all_deltas, axis=0)
        if all_deltas
        else np.empty(0, dtype=np.int16)
    )
    piece_offsets = [0]
    for _, pitches, *_ in pieces:
        piece_offsets.append(piece_offsets[-1] + len(pitches))

    sequence_index = 0
    for piece_id, (_, pitches, durations, deltas, _) in enumerate(pieces):
        piece_sequence_count = max(0, len(pitches) - sequence_length)
        for start_index in range(piece_sequence_count):
            stop_index = start_index + sequence_length
            x_pitches[sequence_index] = pitches[start_index:stop_index]
            x_durations[sequence_index] = durations[start_index:stop_index]
            x_deltas[sequence_index] = deltas[start_index:stop_index]
            y_pitches[sequence_index] = pitches[stop_index]
            y_durations[sequence_index] = durations[stop_index]
            y_deltas[sequence_index] = deltas[stop_index]
            sequence_piece_ids[sequence_index] = piece_id
            sequence_index += 1

    return (
        x_pitches,
        x_durations,
        x_deltas,
        y_pitches,
        y_durations,
        y_deltas,
        group_pitches,
        group_durations,
        group_deltas,
        np.asarray(piece_offsets, dtype=np.int32),
        sequence_piece_ids,
    )


def relative_path(path: Path) -> str:
    """Return a project-relative source path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def save_v3_artifacts(
    genre: str,
    arrays: tuple[np.ndarray, ...],
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    """Save numeric onset-group arrays separately from lightweight V3 metadata."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_v3_sequences.npz"
    metadata_path = PROCESSED_DATA_DIR / f"{genre}_v3_metadata.json"
    (
        x_pitches,
        x_durations,
        x_deltas,
        y_pitches,
        y_durations,
        y_deltas,
        group_pitches,
        group_durations,
        group_deltas,
        piece_offsets,
        sequence_piece_ids,
    ) = arrays
    np.savez_compressed(
        sequence_path,
        X_pitches=x_pitches,
        X_durations=x_durations,
        X_deltas=x_deltas,
        y_pitches=y_pitches,
        y_durations=y_durations,
        y_deltas=y_deltas,
        group_pitches=group_pitches,
        group_durations=group_durations,
        group_deltas=group_deltas,
        piece_offsets=piece_offsets,
        sequence_piece_ids=sequence_piece_ids,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return sequence_path, metadata_path


def main() -> None:
    """Create V3 onset-group artifacts without modifying existing versions."""
    parser = argparse.ArgumentParser(description="Create onset-grouped NeuraTune V3 artifacts.")
    parser.add_argument("--genre", default="classical", help="Genre folder to preprocess.")
    parser.add_argument("--limit", type=int, help="Maximum MIDI files to preprocess.")
    parser.add_argument(
        "--sequence-length", type=int, default=50, help="Onset groups per input sequence."
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
    arrays = create_piece_sequences(pieces, args.sequence_length)
    (
        _,
        _,
        _,
        _,
        _,
        _,
        group_pitches,
        group_durations,
        group_deltas,
        _,
        _,
    ) = arrays

    source_pieces = []
    total_notes = 0
    total_overflow_groups = 0
    total_truncated_notes = 0
    for piece_id, (source_path, pitches, _, _, statistics) in enumerate(pieces):
        total_notes += statistics["note_events"]
        total_overflow_groups += statistics["overflow_groups"]
        total_truncated_notes += statistics["truncated_notes"]
        source_pieces.append(
            {
                "piece_id": piece_id,
                "source_path": relative_path(source_path),
                "note_events": statistics["note_events"],
                "onset_groups": statistics["onset_groups"],
                "duplicates_removed": statistics["duplicates_removed"],
                "overflow_groups": statistics["overflow_groups"],
                "truncated_notes": statistics["truncated_notes"],
            }
        )

    valid_pitch_values = group_pitches[group_pitches != PAD_VALUE]
    valid_duration_values = group_durations[group_durations != PAD_VALUE]
    raw_group_size_counts: dict[int, int] = {}
    for _, _, _, _, piece_statistics in pieces:
        for size, count in piece_statistics["group_size_counts"].items():
            raw_group_size_counts[int(size)] = raw_group_size_counts.get(int(size), 0) + count
    raw_group_sizes = np.repeat(
        np.asarray(sorted(raw_group_size_counts), dtype=np.int16),
        [raw_group_size_counts[size] for size in sorted(raw_group_size_counts)],
    )
    total_groups = len(group_pitches)
    statistics = {
        "total_note_events_before_slot_truncation": total_notes,
        "retained_note_slots": int(np.count_nonzero(group_pitches != PAD_VALUE)),
        "total_onset_groups": total_groups,
        "average_pitches_per_onset": (
            round(total_notes / total_groups, 3) if total_groups else 0.0
        ),
        "median_pitches_per_onset": float(np.median(raw_group_sizes)) if total_groups else 0.0,
        "maximum_pitches_per_onset": int(max((piece[4]["maximum_group_size"] for piece in pieces), default=0)),
        "single_pitch_group_percentage": (
            round(raw_group_size_counts.get(1, 0) / total_groups * 100, 3)
            if total_groups else 0.0
        ),
        "two_to_three_pitch_group_percentage": (
            round(sum(raw_group_size_counts.get(size, 0) for size in (2, 3)) / total_groups * 100, 3)
            if total_groups else 0.0
        ),
        "four_to_five_pitch_group_percentage": (
            round(sum(raw_group_size_counts.get(size, 0) for size in (4, 5)) / total_groups * 100, 3)
            if total_groups else 0.0
        ),
        "six_or_more_pitch_group_percentage": (
            round(sum(count for size, count in raw_group_size_counts.items() if size >= 6) / total_groups * 100, 3)
            if total_groups else 0.0
        ),
        "groups_over_max_pitches": total_overflow_groups,
        "groups_over_max_pitches_percentage": (
            round(total_overflow_groups / total_groups * 100, 3) if total_groups else 0.0
        ),
        "truncated_note_slots": total_truncated_notes,
        "unique_delta_step_values": int(len(np.unique(group_deltas))),
        "unique_duration_step_values": int(len(np.unique(valid_duration_values))),
        "training_sequences": int(len(arrays[0])),
    }
    metadata: dict[str, object] = {
        "representation_version": "v3_onset_grouped",
        "genre": args.genre,
        "grid_quarter_length": DEFAULT_GRID_QUARTER_LENGTH,
        "sequence_length": args.sequence_length,
        "max_pitches_per_onset": MAX_PITCHES_PER_ONSET,
        "padding": {
            "value": PAD_VALUE,
            "pitch_slots": "-1 is padding; MIDI pitches are 0 through 127.",
            "duration_slots": "-1 is padding; retained durations are at least one step.",
        },
        "timestep_layout": {
            "delta_steps": "One delta for the whole onset group.",
            "pitch_slots": MAX_PITCHES_PER_ONSET,
            "duration_slots": MAX_PITCHES_PER_ONSET,
            "slot_alignment": "pitch_slots[i] and duration_slots[i] describe the same note.",
        },
        "first_onset_delta_convention": (
            "The first group in each piece stores its onset distance from time zero; "
            "it is usually zero. Every later group stores a strictly positive distance "
            "from the previous unique quantized onset."
        ),
        "overflow_retention_rule": (
            "For groups larger than eight, retain the first eight records sorted by "
            "ascending MIDI pitch and then duration; report omitted slots."
        ),
        "pitch_values": [int(value) for value in np.unique(valid_pitch_values)],
        "duration_step_values": [int(value) for value in np.unique(valid_duration_values)],
        "delta_step_values": [int(value) for value in np.unique(group_deltas)],
        "source_midi_files": len(pieces),
        "source_pieces": source_pieces,
        "statistics": statistics,
    }
    sequence_path, metadata_path = save_v3_artifacts(args.genre, arrays, metadata)

    print("V3 onset-grouped preprocessing summary")
    print(f"MIDI files processed: {len(pieces)}")
    print(f"MIDI files skipped: {len(invalid_files) + len(parse_failures)}")
    print(f"Total note events: {statistics['total_note_events_before_slot_truncation']}")
    print(f"Total onset groups: {statistics['total_onset_groups']}")
    print(f"Average pitches per onset: {statistics['average_pitches_per_onset']}")
    print(f"Median pitches per onset: {statistics['median_pitches_per_onset']}")
    print(f"Maximum pitches per onset: {statistics['maximum_pitches_per_onset']}")
    print(f"Groups over {MAX_PITCHES_PER_ONSET} pitches: {total_overflow_groups}")
    print(f"Unique delta-step values: {statistics['unique_delta_step_values']}")
    print(f"Unique duration-step values: {statistics['unique_duration_step_values']}")
    print(f"Training sequences: {statistics['training_sequences']}")
    print(f"Sequence artifact: {sequence_path}")
    print(f"Metadata artifact: {metadata_path}")


if __name__ == "__main__":
    main()
