"""Experimental V4E preprocessing with compact rhythm buckets.

This module intentionally keeps the V4 onset-group representation while
mapping sparse raw timing values into eight deterministic classes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import PROJECT_ROOT, discover_midi_files
from .preprocess_v4 import (
    DEFAULT_GRID_QUARTER_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    MAX_PITCHES_PER_ONSET,
    PAD_VALUE,
    build_onset_groups,
    create_sequences,
    process_files,
    relative_path,
    validate_arrays,
)

PROCESSED = PROJECT_ROOT / "data" / "processed"
BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, None))


def bucket_for(value: int) -> int:
    # A piece may begin exactly at time zero.  Keep the eight-class design by
    # assigning this rare first-onset value to the shortest timing bucket.
    if value == 0:
        return 1
    for index, (low, high) in enumerate(BUCKETS, start=1):
        if value >= low and (high is None or value <= high):
            return index
    raise ValueError(f"Rhythm value must be positive: {value}")


def bucket_label(index: int) -> str:
    low, high = BUCKETS[index - 1]
    return f"{low}+" if high is None else (str(low) if low == high else f"{low}-{high}")


def representative_values(pieces: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive decode values from training pieces only (IDs 0-39)."""
    training = pieces[:40] if len(pieces) >= 40 else pieces
    deltas = [int(v) for piece in training for v in piece["deltas"].tolist()]
    durations = [int(d) for piece in training for _, _, d in piece["records"]]

    def reps(values: list[int]) -> dict[str, int]:
        result: dict[str, int] = {}
        for index, (low, high) in enumerate(BUCKETS, start=1):
            selected = [v for v in values if v >= low and (high is None or v <= high)]
            if not selected:
                raise ValueError(f"Bucket {bucket_label(index)} is empty in training data")
            result[str(index)] = int(np.median(selected))
        return result

    return {"training_piece_count": len(training), "delta": reps(deltas), "duration": reps(durations)}


def bucket_distribution(values: list[int]) -> list[dict[str, Any]]:
    counts = Counter(bucket_for(int(v)) for v in values)
    total = max(1, len(values))
    return [{"class_index": i, "bucket": bucket_label(i), "count": int(counts[i]), "percentage": float(counts[i] * 100 / total)} for i in range(1, 9)]


def transform_pieces(pieces: list[dict[str, Any]]) -> None:
    """Replace timing values in arrays with compact class IDs, preserving PAD."""
    for piece in pieces:
        durations = piece["durations"].copy()
        valid = durations != PAD_VALUE
        durations[valid] = np.asarray([bucket_for(int(v)) for v in durations[valid]], dtype=np.int16)
        piece["durations"] = durations.astype(np.int16)
        piece["deltas"] = np.asarray([bucket_for(int(v)) for v in piece["deltas"]], dtype=np.int16)


def encode_mapping(values: np.ndarray, include_pad: bool) -> dict[str, Any]:
    classes = sorted(int(v) for v in np.unique(values) if int(v) != PAD_VALUE)
    index_to_value = ([PAD_VALUE] + classes) if include_pad else classes
    return {"index_to_value": index_to_value, "value_to_index": {str(v): i for i, v in enumerate(index_to_value)}, "pad_value": PAD_VALUE, "pad_class_index": 0 if include_pad else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact-rhythm V4E artifacts.")
    parser.add_argument("--genre", default="classical")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    args = parser.parse_args()
    if args.limit < 1 or args.sequence_length < 1:
        parser.error("--limit and --sequence-length must be positive")
    selected = discover_midi_files(genre=args.genre)[: args.limit]
    pieces, failures = process_files(selected, DEFAULT_GRID_QUARTER_LENGTH)
    if len(pieces) + len(failures) != len(selected):
        raise RuntimeError("File accounting failed")
    reps = representative_values(pieces)
    raw_delta = [int(v) for piece in pieces for v in piece["deltas"].tolist()]
    raw_duration = [int(d) for piece in pieces for _, _, d in piece["records"]]
    training_pieces = pieces[:40] if len(pieces) >= 40 else pieces
    training_delta = [int(v) for piece in training_pieces for v in piece["deltas"].tolist()]
    training_duration = [int(d) for piece in training_pieces for _, _, d in piece["records"]]
    delta_dist = bucket_distribution(raw_delta)
    duration_dist = bucket_distribution(raw_duration)
    transform_pieces(pieces)
    arrays = create_sequences(pieces, args.sequence_length)
    validation = validate_arrays(pieces, arrays, args.sequence_length)
    (xp, xd, xdelta, yp, yd, ydelta, gp, gd, gdelta, offsets, ids, counts) = arrays
    metadata = {
        "representation_version": "V4E_compact_rhythm",
        "genre": args.genre,
        "source_files": [relative_path(p) for p in selected],
        "source_midi_files": len(pieces),
        "failed_files": failures,
        "grid_quarter_length": DEFAULT_GRID_QUARTER_LENGTH,
        "sequence_length": args.sequence_length,
        "max_pitches_per_onset": MAX_PITCHES_PER_ONSET,
        "padding": {"value": PAD_VALUE, "pitch_duration_class_index": 0, "delta_has_pad": False},
        "bucket_definitions": [{"class_index": i, "label": bucket_label(i), "minimum": low, "maximum": high} for i, (low, high) in enumerate(BUCKETS, 1)],
        "representative_decode_steps": reps,
        "mappings": {
            "pitch": encode_mapping(gp, True),
            "duration_steps": {"index_to_value": [-1, *range(1, 9)], "value_to_index": {str(v): i for i, v in enumerate([-1, *range(1, 9)])}, "pad_value": PAD_VALUE, "pad_class_index": 0},
            "delta_steps": {"index_to_value": list(range(1, 9)), "value_to_index": {str(v): i - 1 for i, v in enumerate(range(1, 9), 1)}, "pad_value": PAD_VALUE, "pad_class_index": None},
        },
        "total_raw_notes": len(raw_duration), "unique_onset_groups": int(len(gdelta)), "sequence_count": int(len(xp)),
        "pitch_classes": int(len(encode_mapping(gp, True)["index_to_value"]) - 1),
        "delta_bucket_distribution": delta_dist, "duration_bucket_distribution": duration_dist,
        "training_bucket_distribution": {"delta": bucket_distribution(training_delta), "duration": bucket_distribution(training_duration)},
        "representative_training_piece_count": reps["training_piece_count"],
        "raw_rhythm_statistics": {"delta_unique": len(set(raw_delta)), "duration_unique": len(set(raw_duration))},
        "piece_boundary_validation": validation,
        "per_piece_sequence_counts": counts.tolist(),
    }
    names = ("X_pitches", "X_durations", "X_deltas", "y_pitches", "y_durations", "y_deltas", "group_pitches", "group_durations", "group_deltas", "piece_offsets", "sequence_piece_ids", "per_piece_sequence_counts")
    out_npz = PROCESSED / f"{args.genre}_v4e_sequences.npz"; out_json = PROCESSED / f"{args.genre}_v4e_metadata.json"
    np.savez_compressed(out_npz, **dict(zip(names, arrays)))
    out_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("V4E preprocessing summary")
    print(f"Files processed: {len(pieces)} | failed: {len(failures)} | notes: {len(raw_duration)} | onsets: {len(gdelta)} | sequences: {len(xp)}")
    print("All-file delta buckets:", delta_dist); print("All-file duration buckets:", duration_dist)
    print("Training-piece delta buckets:", metadata["training_bucket_distribution"]["delta"])
    print("Training-piece duration buckets:", metadata["training_bucket_distribution"]["duration"])
    print("Representative decode values:", reps)
    print("Tensor shapes:", {name: list(value.shape) for name, value in zip(names, arrays) if name.startswith(("X_", "y_"))})
    print(f"Artifacts: {out_npz} | {out_json}")


if __name__ == "__main__":
    main()
