"""Create onset-grouped V4 MIDI artifacts without changing earlier versions."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from music21 import chord, converter, note, stream

from src.dataset import PROJECT_ROOT, discover_midi_files
from src.preprocess_v2 import quantize_to_steps


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_GRID_QUARTER_LENGTH = 0.25
DEFAULT_SEQUENCE_LENGTH = 50
MAX_PITCHES_PER_ONSET = 8
PAD_VALUE = -1


def parse_midi_file(midi_file: Path) -> tuple[stream.Score | None, str | None]:
    """Parse one MIDI file while allowing other pieces to continue."""
    try:
        return converter.parse(midi_file), None
    except Exception as error:
        message = str(error).strip() or error.__class__.__name__
        return None, message


def extract_note_records(
    score: stream.Score, grid_quarter_length: float = DEFAULT_GRID_QUARTER_LENGTH
) -> tuple[list[tuple[int, int, int]], int]:
    """Extract exact-deduplicated ``(pitch, onset, duration)`` records."""
    candidates: list[tuple[int, int, int]] = []
    for element in score.flatten().notes:
        try:
            onset_steps = quantize_to_steps(element.offset, grid_quarter_length, 0)
            duration_steps = quantize_to_steps(
                element.duration.quarterLength, grid_quarter_length, 1
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
            continue

    candidates.sort(key=lambda item: (item[1], item[0], item[2]))
    unique_records: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    duplicates_removed = 0
    for record in candidates:
        if record in seen:
            duplicates_removed += 1
        else:
            seen.add(record)
            unique_records.append(record)
    return unique_records, duplicates_removed


def select_span_preserving_records(
    records: list[tuple[int, int]], max_pitches: int = MAX_PITCHES_PER_ONSET
) -> list[tuple[int, int]]:
    """Retain the low/high register and evenly sample the interior pitches."""
    if len(records) <= max_pitches:
        return records
    interior_count = max_pitches - 2
    interior_indices = np.rint(
        np.linspace(1, len(records) - 2, interior_count)
    ).astype(int).tolist()
    selected_indices = [0, *interior_indices, len(records) - 1]
    selected_indices = list(dict.fromkeys(selected_indices))
    if len(selected_indices) < max_pitches:
        for index in range(1, len(records) - 1):
            if index not in selected_indices:
                selected_indices.insert(-1, index)
            if len(selected_indices) == max_pitches:
                break
    return [records[index] for index in sorted(selected_indices)]


def build_onset_groups(
    records: list[tuple[int, int, int]],
    max_pitches: int = MAX_PITCHES_PER_ONSET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Group simultaneous notes and create aligned padded slots."""
    records_by_onset: dict[int, list[tuple[int, int]]] = {}
    for pitch, onset_steps, duration_steps in records:
        records_by_onset.setdefault(onset_steps, []).append((pitch, duration_steps))

    onset_values = sorted(records_by_onset)
    pitches = np.full((len(onset_values), max_pitches), PAD_VALUE, dtype=np.int16)
    durations = np.full_like(pitches, PAD_VALUE)
    deltas = np.empty(len(onset_values), dtype=np.int16)
    original_sizes: list[int] = []
    retained_sizes: list[int] = []
    previous_onset = 0
    overflow_groups = 0
    omitted_notes = 0

    for group_index, onset_steps in enumerate(onset_values):
        ordered = sorted(records_by_onset[onset_steps], key=lambda item: (item[0], item[1]))
        retained = select_span_preserving_records(ordered, max_pitches)
        original_sizes.append(len(ordered))
        retained_sizes.append(len(retained))
        if len(ordered) > max_pitches:
            overflow_groups += 1
            omitted_notes += len(ordered) - len(retained)
        for slot, (pitch, duration_steps) in enumerate(retained):
            pitches[group_index, slot] = pitch
            durations[group_index, slot] = duration_steps
        delta = onset_steps if group_index == 0 else onset_steps - previous_onset
        deltas[group_index] = delta
        previous_onset = onset_steps

    group_size_counts = Counter(original_sizes)
    statistics = {
        "total_onset_groups": len(onset_values),
        "maximum_group_size": max(original_sizes, default=0),
        "group_size_counts": {str(size): int(count) for size, count in sorted(group_size_counts.items())},
        "overflow_groups": overflow_groups,
        "omitted_notes": omitted_notes,
        "total_group_notes_before_overflow": int(sum(original_sizes)),
        "retained_group_notes": int(sum(retained_sizes)),
    }
    return (
        pitches,
        durations,
        deltas,
        np.asarray(original_sizes, dtype=np.int16),
        statistics,
    )


def process_files(
    midi_files: list[Path], grid_quarter_length: float
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Parse each selected file independently."""
    pieces: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for midi_file in midi_files:
        score, error = parse_midi_file(midi_file)
        if score is None:
            failures.append({"path": relative_path(midi_file), "error": error or "parse failed"})
            continue
        records, duplicates_removed = extract_note_records(score, grid_quarter_length)
        pitches, durations, deltas, original_sizes, statistics = build_onset_groups(records)
        pieces.append(
            {
                "path": midi_file,
                "records": records,
                "pitches": pitches,
                "durations": durations,
                "deltas": deltas,
                "original_sizes": original_sizes,
                "duplicates_removed": duplicates_removed,
                "statistics": statistics,
            }
        )
    return pieces, failures


def create_sequences(
    pieces: list[dict[str, Any]], sequence_length: int
) -> tuple[np.ndarray, ...]:
    """Create windows within pieces only; targets are the following onset group."""
    sequence_count = sum(max(0, len(piece["pitches"]) - sequence_length) for piece in pieces)
    x_pitches = np.empty((sequence_count, sequence_length, MAX_PITCHES_PER_ONSET), dtype=np.int16)
    x_durations = np.empty_like(x_pitches)
    x_deltas = np.empty((sequence_count, sequence_length), dtype=np.int16)
    y_pitches = np.empty((sequence_count, MAX_PITCHES_PER_ONSET), dtype=np.int16)
    y_durations = np.empty_like(y_pitches)
    y_deltas = np.empty(sequence_count, dtype=np.int16)
    sequence_piece_ids = np.empty(sequence_count, dtype=np.int16)

    all_pitches = [piece["pitches"] for piece in pieces]
    all_durations = [piece["durations"] for piece in pieces]
    all_deltas = [piece["deltas"] for piece in pieces]
    group_pitches = np.concatenate(all_pitches, axis=0) if all_pitches else np.empty((0, 8), dtype=np.int16)
    group_durations = np.concatenate(all_durations, axis=0) if all_durations else np.empty((0, 8), dtype=np.int16)
    group_deltas = np.concatenate(all_deltas, axis=0) if all_deltas else np.empty(0, dtype=np.int16)
    piece_offsets = [0]
    for piece in pieces:
        piece_offsets.append(piece_offsets[-1] + len(piece["pitches"]))

    index = 0
    per_piece_sequence_counts: list[int] = []
    for piece_id, piece in enumerate(pieces):
        count = max(0, len(piece["pitches"]) - sequence_length)
        per_piece_sequence_counts.append(count)
        for start in range(count):
            stop = start + sequence_length
            x_pitches[index] = piece["pitches"][start:stop]
            x_durations[index] = piece["durations"][start:stop]
            x_deltas[index] = piece["deltas"][start:stop]
            y_pitches[index] = piece["pitches"][stop]
            y_durations[index] = piece["durations"][stop]
            y_deltas[index] = piece["deltas"][stop]
            sequence_piece_ids[index] = piece_id
            index += 1

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
        np.asarray(per_piece_sequence_counts, dtype=np.int32),
    )


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def mapping(values: np.ndarray, include_pad: bool) -> dict[str, Any]:
    actual_values = [int(value) for value in np.unique(values)]
    index_values = ([PAD_VALUE] + actual_values) if include_pad else actual_values
    return {
        "index_to_value": index_values,
        "value_to_index": {str(value): index for index, value in enumerate(index_values)},
        "pad_value": PAD_VALUE,
        "pad_class_index": 0 if include_pad else None,
    }


def numeric_statistics(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "unique_values": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "unique_values": int(np.unique(array).size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def top_frequencies(values: list[int], limit: int = 20) -> list[dict[str, float | int]]:
    counts = Counter(int(value) for value in values)
    total = len(values)
    return [
        {
            "value": int(value),
            "count": int(count),
            "percentage": float(count / total * 100),
        }
        for value, count in counts.most_common(limit)
    ]


def piece_statistics(piece: dict[str, Any], grid: float) -> dict[str, Any]:
    records = piece["records"]
    durations = [duration for _, _, duration in records]
    deltas = piece["deltas"].tolist()
    pitches = [pitch for pitch, _, _ in records]
    return {
        "source_path": relative_path(piece["path"]),
        "note_events": len(records),
        "unique_onsets": len(piece["deltas"]),
        "duplicates_removed": piece["duplicates_removed"],
        "overflow_groups": piece["statistics"]["overflow_groups"],
        "omitted_notes": piece["statistics"]["omitted_notes"],
        "average_notes_per_onset": float(len(records) / len(piece["deltas"])) if piece["deltas"].size else 0.0,
        "pitch_minimum": min(pitches) if pitches else None,
        "pitch_maximum": max(pitches) if pitches else None,
        "average_duration_steps": float(np.mean(durations)) if durations else None,
        "average_delta_steps": float(np.mean(deltas)) if deltas else None,
        "sequence_count": 0,
        "duration_quarter_lengths": float(
            max((onset + duration for _, onset, duration in records), default=0) * grid
        ),
    }


def validate_arrays(
    pieces: list[dict[str, Any]], arrays: tuple[np.ndarray, ...], sequence_length: int
) -> dict[str, Any]:
    """Validate shapes, padding, alignment, timing, and piece boundaries."""
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
        per_piece_sequence_counts,
    ) = arrays
    assert x_pitches.shape[1:] == (sequence_length, 8)
    assert x_durations.shape == x_pitches.shape
    assert x_deltas.shape == (len(x_pitches), sequence_length)
    assert y_pitches.shape == (len(x_pitches), 8)
    assert y_durations.shape == y_pitches.shape
    assert y_deltas.shape == (len(x_pitches),)
    for pitch_array, duration_array in (
        (x_pitches, x_durations),
        (y_pitches, y_durations),
        (group_pitches, group_durations),
    ):
        assert np.array_equal((pitch_array == PAD_VALUE), (duration_array == PAD_VALUE))
    valid_pitch = np.concatenate(
        [array[array != PAD_VALUE] for array in (x_pitches, y_pitches, group_pitches)]
    )
    valid_duration = np.concatenate(
        [array[array != PAD_VALUE] for array in (x_durations, y_durations, group_durations)]
    )
    assert np.all((0 <= valid_pitch) & (valid_pitch <= 127))
    assert np.all(valid_duration >= 1)
    assert np.all(sequence_piece_ids >= 0)
    assert np.array_equal(piece_offsets, np.cumsum([0] + [len(piece["pitches"]) for piece in pieces]))
    expected_counts = np.asarray(
        [max(0, len(piece["pitches"]) - sequence_length) for piece in pieces], dtype=np.int32
    )
    assert np.array_equal(per_piece_sequence_counts, expected_counts)
    assert int(per_piece_sequence_counts.sum()) == len(x_pitches)

    for piece in pieces:
        deltas = piece["deltas"]
        assert int(piece["original_sizes"].sum()) == len(piece["records"])
        if len(deltas) > 1:
            assert np.all(deltas[1:] > 0)
        reconstructed = np.cumsum(deltas)
        assert np.all(np.diff(reconstructed) > 0) if len(reconstructed) > 1 else True
        assert np.all(piece["pitches"][:, 0] != PAD_VALUE) if len(piece["pitches"]) else True

    # Every sequence ID corresponds to a single source piece by construction.
    cursor = 0
    for piece_id, count in enumerate(per_piece_sequence_counts):
        assert np.all(sequence_piece_ids[cursor : cursor + count] == piece_id)
        cursor += int(count)
    return {
        "tensor_shapes": {
            "X_pitches": list(x_pitches.shape),
            "X_durations": list(x_durations.shape),
            "X_deltas": list(x_deltas.shape),
            "y_pitches": list(y_pitches.shape),
            "y_durations": list(y_durations.shape),
            "y_deltas": list(y_deltas.shape),
        },
        "piece_boundaries_preserved": True,
        "pitch_duration_alignment": True,
        "pad_only_in_unused_slots": True,
        "real_pitch_range_valid": True,
        "real_durations_at_least_one": True,
        "later_deltas_positive": True,
        "onsets_reconstruct_monotonically": True,
        "simultaneous_notes_grouped": all(
            int(piece["original_sizes"].sum()) == len(piece["records"])
            for piece in pieces
        ),
    }


def compare_v4a(
    aggregate: dict[str, Any], pieces: list[dict[str, Any]], grid: float
) -> dict[str, Any]:
    """Compare raw/group statistics with the completed V4A report."""
    diagnostic_path = PROCESSED_DATA_DIR / "classical_v4_diagnostic.json"
    if not diagnostic_path.exists():
        return {"status": "unavailable", "path": relative_path(diagnostic_path)}
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    reference_file_count = int(diagnostic.get("files_analyzed", 0))
    if len(pieces) != reference_file_count:
        return {
            "status": "not_comparable_subset",
            "reference_path": relative_path(diagnostic_path),
            "reference_files": reference_file_count,
            "current_files": len(pieces),
            "interpretation": "The current run is a development subset; compare against V4A only when the same file count is processed.",
        }
    reference = diagnostic.get("aggregate", {})
    actual_pitch = aggregate["pitch_statistics"]
    reference_pitch = reference.get("pitch_register", {})
    differences = {
        "total_notes": aggregate["total_raw_notes"] - reference.get("total_notes", 0),
        "unique_onsets": aggregate["total_unique_onsets"] - reference.get("total_unique_onsets", 0),
        "average_notes_per_onset": aggregate["average_notes_per_onset"] - reference.get("notes_per_onset", {}).get("statistics", {}).get("mean", 0),
        "pitch_mean": actual_pitch["mean"] - reference_pitch.get("mean", 0),
        "pitch_median": actual_pitch["median"] - reference_pitch.get("median", 0),
        "pitch_p05": actual_pitch["p05"] - reference_pitch.get("p05", 0),
        "pitch_p95": actual_pitch["p95"] - reference_pitch.get("p95", 0),
    }
    matches = (
        differences["total_notes"] == 0
        and differences["unique_onsets"] == 0
        and abs(differences["average_notes_per_onset"]) < 0.001
        and abs(differences["pitch_mean"]) < 0.001
        and abs(differences["pitch_median"]) < 0.001
        and abs(differences["pitch_p05"]) < 0.001
        and abs(differences["pitch_p95"]) < 0.001
    )
    return {
        "status": "matches_reference" if matches else "discrepancy_detected",
        "reference_path": relative_path(diagnostic_path),
        "differences": differences,
        "interpretation": "Raw note/onset/pitch measurements match V4A; retained slot values differ only for measured overflow groups." if matches else "Large discrepancy requires investigation before model training.",
    }


def save_artifacts(genre: str, arrays: tuple[np.ndarray, ...], metadata: dict[str, Any]) -> tuple[Path, Path]:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sequence_path = PROCESSED_DATA_DIR / f"{genre}_v4_sequences.npz"
    metadata_path = PROCESSED_DATA_DIR / f"{genre}_v4_metadata.json"
    names = (
        "X_pitches", "X_durations", "X_deltas", "y_pitches", "y_durations", "y_deltas",
        "group_pitches", "group_durations", "group_deltas", "piece_offsets",
        "sequence_piece_ids", "per_piece_sequence_counts",
    )
    np.savez_compressed(sequence_path, **dict(zip(names, arrays)))
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return sequence_path, metadata_path


def build_metadata(
    genre: str,
    selected_files: list[Path],
    pieces: list[dict[str, Any]],
    arrays: tuple[np.ndarray, ...],
    failures: list[dict[str, str]],
    grid: float,
    sequence_length: int,
) -> dict[str, Any]:
    records = [record for piece in pieces for record in piece["records"]]
    pitches = [pitch for pitch, _, _ in records]
    durations = [duration for _, _, duration in records]
    groups = [group for piece in pieces for group in piece["statistics"]["group_size_counts"].items()]
    total_groups = sum(piece["statistics"]["total_onset_groups"] for piece in pieces)
    group_sizes = [
        int(size)
        for piece in pieces
        for size, count in piece["statistics"]["group_size_counts"].items()
        for _ in range(count)
    ]
    _, _, _, _, _, _, group_pitches, group_durations, group_deltas, _, _, per_piece_counts = arrays
    retained_pitches = group_pitches[group_pitches != PAD_VALUE]
    retained_durations = group_durations[group_durations != PAD_VALUE]
    all_deltas = group_deltas.tolist()
    pitch_stats = numeric_statistics(pitches)
    pitch_stats.update(
        {
            "minimum": int(min(pitches)) if pitches else None,
            "maximum": int(max(pitches)) if pitches else None,
            "p05": float(np.percentile(pitches, 5)) if pitches else None,
            "p95": float(np.percentile(pitches, 95)) if pitches else None,
        }
    )
    average_notes = len(records) / total_groups if total_groups else 0.0
    median_notes = float(np.median(group_sizes)) if group_sizes else 0.0
    sequence_count = int(len(arrays[0]))
    piece_metadata = []
    for piece, sequence_count_for_piece in zip(pieces, per_piece_counts.tolist()):
        stats = piece_statistics(piece, grid)
        stats["piece_id"] = len(piece_metadata)
        stats["sequence_count"] = int(sequence_count_for_piece)
        piece_metadata.append(stats)

    return {
        "representation_version": "V4_onset_grouped",
        "genre": genre,
        "source_files": [relative_path(path) for path in selected_files],
        "source_midi_files": len(pieces),
        "failed_files": failures,
        "grid_quarter_length": grid,
        "sequence_length": sequence_length,
        "max_pitches_per_onset": MAX_PITCHES_PER_ONSET,
        "padding": {
            "value": PAD_VALUE,
            "pitch_slots": "-1 is padding; valid MIDI pitch values are 0 through 127.",
            "duration_slots": "-1 is padding; valid duration steps are at least 1.",
            "delta_slots": "No delta padding is used; each onset group has one delta value.",
        },
        "timestep_layout": {
            "delta_steps": "Distance from time zero for the first onset, then positive distance from the previous unique onset.",
            "pitch_slots": MAX_PITCHES_PER_ONSET,
            "duration_slots": MAX_PITCHES_PER_ONSET,
            "slot_alignment": "pitch_slots[i] and duration_slots[i] describe the same retained note.",
        },
        "overflow_retention_rule": "For groups over eight, retain lowest and highest pitches, plus six evenly spaced sorted interior pitches; durations remain paired with selected pitches.",
        "mappings": {
            "pitch": mapping(retained_pitches, include_pad=True),
            "duration_steps": mapping(retained_durations, include_pad=True),
            "delta_steps": mapping(group_deltas, include_pad=False),
        },
        "pitch_statistics": pitch_stats,
        "total_raw_notes": len(records),
        "retained_notes": int(len(retained_pitches)),
        "unique_onset_groups": total_groups,
        "notes_per_onset": {
            "mean": float(average_notes),
            "median": median_notes,
            "maximum": max(group_sizes, default=0),
            "percentages": {
                "1_note": float(sum(size == 1 for size in group_sizes) / total_groups * 100) if total_groups else 0.0,
                "2_notes": float(sum(size == 2 for size in group_sizes) / total_groups * 100) if total_groups else 0.0,
                "3_notes": float(sum(size == 3 for size in group_sizes) / total_groups * 100) if total_groups else 0.0,
                "4_notes": float(sum(size == 4 for size in group_sizes) / total_groups * 100) if total_groups else 0.0,
                "5_notes": float(sum(size == 5 for size in group_sizes) / total_groups * 100) if total_groups else 0.0,
                "6_plus_notes": float(sum(size >= 6 for size in group_sizes) / total_groups * 100) if total_groups else 0.0,
            },
            "original_group_size_counts": {str(size): int(count) for size, count in sorted(Counter(group_sizes).items())},
        },
        "duration_statistics": {
            **numeric_statistics(durations),
            "top_20": top_frequencies(durations),
        },
        "delta_statistics": {
            **numeric_statistics(all_deltas),
            "top_20": top_frequencies(all_deltas),
        },
        "overflow_statistics": {
            "groups_over_8": sum(piece["statistics"]["overflow_groups"] for piece in pieces),
            "percentage_of_groups_over_8": float(sum(piece["statistics"]["overflow_groups"] for piece in pieces) / total_groups * 100) if total_groups else 0.0,
            "notes_omitted": sum(piece["statistics"]["omitted_notes"] for piece in pieces),
            "retained_notes_after_overflow": int(len(retained_pitches)),
        },
        "sequence_count": sequence_count,
        "per_piece_sequence_counts": [int(value) for value in per_piece_counts.tolist()],
        "source_piece_statistics": piece_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create onset-grouped NeuraTune V4 artifacts.")
    parser.add_argument("--genre", default="classical")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.sequence_length < 1:
        parser.error("--sequence-length must be positive.")

    discovered = discover_midi_files(genre=args.genre)
    selected = discovered[: args.limit]
    pieces, failures = process_files(selected, DEFAULT_GRID_QUARTER_LENGTH)
    if len(pieces) + len(failures) != len(selected):
        raise RuntimeError("File accounting failed during V4 preprocessing.")
    arrays = create_sequences(pieces, args.sequence_length)
    validation = validate_arrays(pieces, arrays, args.sequence_length)

    metadata = build_metadata(
        args.genre, selected, pieces, arrays, failures, DEFAULT_GRID_QUARTER_LENGTH, args.sequence_length
    )
    metadata["validation"] = validation
    comparison = compare_v4a(metadata_for_comparison(metadata), pieces, DEFAULT_GRID_QUARTER_LENGTH)
    metadata["v4a_comparison"] = comparison
    if comparison.get("status") == "discrepancy_detected":
        raise RuntimeError(f"V4A comparison discrepancy: {comparison['differences']}")

    sequence_path, metadata_path = save_artifacts(args.genre, arrays, metadata)
    reload_validation = reload_and_validate(sequence_path, metadata_path, arrays, metadata)
    metadata["validation"]["npz_json_reload_consistent"] = reload_validation
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    print("V4 onset-grouped preprocessing summary")
    print(f"MIDI files processed: {len(pieces)}")
    print(f"MIDI files failed: {len(failures)}")
    print(f"Total notes before overflow: {metadata['total_raw_notes']}")
    print(f"Retained notes: {metadata['retained_notes']}")
    print(f"Unique onset groups: {metadata['unique_onset_groups']}")
    print(f"Average/median/max pitches per onset: {metadata['notes_per_onset']['mean']:.3f} / {metadata['notes_per_onset']['median']:.3f} / {metadata['notes_per_onset']['maximum']}")
    print(f"Overflow groups (>8): {metadata['overflow_statistics']['groups_over_8']} ({metadata['overflow_statistics']['percentage_of_groups_over_8']:.3f}%), omitted notes: {metadata['overflow_statistics']['notes_omitted']}")
    print(f"Unique pitch classes: {len(metadata['mappings']['pitch']['index_to_value']) - 1}")
    print(f"Unique duration classes: {len(metadata['mappings']['duration_steps']['index_to_value']) - 1}")
    print(f"Unique delta classes: {len(metadata['mappings']['delta_steps']['index_to_value'])}")
    print(f"Training sequences: {metadata['sequence_count']}")
    print(f"NPZ artifact: {sequence_path}")
    print(f"Metadata artifact: {metadata_path}")
    print(f"V4A comparison: {comparison['status']}")


def metadata_for_comparison(metadata: dict[str, Any]) -> dict[str, Any]:
    """Expose the aggregate fields needed by the V4A comparison helper."""
    return {
        "total_raw_notes": metadata["total_raw_notes"],
        "total_unique_onsets": metadata["unique_onset_groups"],
        "average_notes_per_onset": metadata["notes_per_onset"]["mean"],
        "pitch_statistics": metadata["pitch_statistics"],
    }


def reload_and_validate(
    sequence_path: Path,
    metadata_path: Path,
    arrays: tuple[np.ndarray, ...],
    metadata: dict[str, Any],
) -> bool:
    """Verify saved NPZ shapes and JSON mappings match the in-memory artifacts."""
    with np.load(sequence_path, allow_pickle=False) as loaded:
        names = (
            "X_pitches", "X_durations", "X_deltas", "y_pitches", "y_durations", "y_deltas",
            "group_pitches", "group_durations", "group_deltas", "piece_offsets",
            "sequence_piece_ids", "per_piece_sequence_counts",
        )
        assert all(np.array_equal(loaded[name], array) for name, array in zip(names, arrays))
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["mappings"] == metadata["mappings"]
    assert saved_metadata["sequence_count"] == metadata["sequence_count"]
    return True


if __name__ == "__main__":
    main()
