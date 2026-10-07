"""Diagnostic analysis of the original MIDI training music for V4A.

This module only measures the existing classical MIDI subset.  It does not
create sequences, alter MIDI files, train a model, or change generation.
"""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from music21 import chord, converter, note

from src.dataset import PROJECT_ROOT, discover_midi_files
from src.preprocess_v2 import quantize_to_steps


GRID_QUARTER_LENGTH = 0.25
DEFAULT_FILE_LIMIT = 50
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIRECTORY = PROJECT_ROOT / "outputs"


def _numeric_summary(values: list[float | int]) -> dict[str, Any]:
    """Return robust descriptive statistics for a numeric collection."""
    if not values:
        return {
            "count": 0,
            "unique_values": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "unique_values": int(np.unique(array).size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def _frequency_table(values: list[int], limit: int = 20) -> list[dict[str, float | int]]:
    """Return the most common values with counts and percentages."""
    if not values:
        return []
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


def _percentage(values: list[int], target: int) -> float:
    return float(sum(value == target for value in values) / len(values) * 100) if values else 0.0


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def extract_quantized_events(
    score: object, grid_quarter_length: float = GRID_QUARTER_LENGTH
) -> list[tuple[int, int, int]]:
    """Extract deduplicated ``(pitch, onset_steps, duration_steps)`` events."""
    candidates: list[tuple[int, int, int]] = []
    for element in score.flatten().notes:
        try:
            onset_steps = quantize_to_steps(element.offset, grid_quarter_length, minimum_steps=0)
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
                    candidates.append((midi_pitch, onset_steps, duration_steps))
        except (AttributeError, TypeError, ValueError):
            continue

    unique_events = sorted(set(candidates), key=lambda event: (event[1], event[0], event[2]))
    return unique_events


def group_events(events: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    """Group note occurrences by quantized onset without crossing a piece."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    for pitch, onset_steps, duration_steps in events:
        grouped.setdefault(onset_steps, []).append((pitch, duration_steps))
    return [
        {
            "onset_steps": int(onset_steps),
            "notes": sorted(notes, key=lambda item: (item[0], item[1])),
        }
        for onset_steps, notes in sorted(grouped.items())
    ]


def analyze_piece(path: Path, grid_quarter_length: float) -> dict[str, Any]:
    """Parse and summarize one MIDI piece."""
    score = converter.parse(path)
    events = extract_quantized_events(score, grid_quarter_length)
    groups = group_events(events)
    onset_deltas: list[int] = []
    previous_onset = 0
    for group in groups:
        onset = int(group["onset_steps"])
        onset_deltas.append(onset - previous_onset)
        previous_onset = onset

    durations = [duration for _, _, duration in events]
    pitches = [pitch for pitch, _, _ in events]
    end_step = max((onset + duration for _, onset, duration in events), default=0)
    notes_per_onset = [len(group["notes"]) for group in groups]
    return {
        "path": _relative_path(path),
        "events": events,
        "groups": groups,
        "notes": len(events),
        "unique_onsets": len(groups),
        "notes_per_onset": notes_per_onset,
        "durations": durations,
        "onset_deltas": onset_deltas,
        "pitches": pitches,
        "duration_quarter_lengths": float(end_step * grid_quarter_length),
        "average_duration_steps": float(np.mean(durations)) if durations else None,
        "average_onset_delta_steps": float(np.mean(onset_deltas)) if onset_deltas else None,
    }


def _onset_distribution(note_counts: list[int]) -> dict[str, Any]:
    percentages = {}
    total = len(note_counts)
    for label, predicate in (
        ("1_note", lambda value: value == 1),
        ("2_notes", lambda value: value == 2),
        ("3_notes", lambda value: value == 3),
        ("4_notes", lambda value: value == 4),
        ("5_notes", lambda value: value == 5),
        ("6_plus_notes", lambda value: value >= 6),
    ):
        percentages[label] = float(sum(predicate(value) for value in note_counts) / total * 100) if total else 0.0
    return {"statistics": _numeric_summary(note_counts), "percentages": percentages}


def _pitch_register(pitches: list[int]) -> dict[str, Any]:
    if not pitches:
        return {"minimum": None, "maximum": None, "mean": None, "median": None, "p05": None, "p95": None, "unique_midi_pitches": 0, "pitch_class_distribution": {}, "octave_distribution": {}}
    array = np.asarray(pitches, dtype=np.float64)
    pitch_classes = Counter(int(pitch) % 12 for pitch in pitches)
    octaves = Counter(int(pitch) // 12 - 1 for pitch in pitches)
    total = len(pitches)
    return {
        "minimum": int(np.min(array)),
        "maximum": int(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "unique_midi_pitches": len(set(pitches)),
        "pitch_class_distribution": {
            str(value): {"count": int(count), "percentage": float(count / total * 100)}
            for value, count in sorted(pitch_classes.items())
        },
        "octave_distribution": {
            str(value): {"count": int(count), "percentage": float(count / total * 100)}
            for value, count in sorted(octaves.items())
        },
    }


def _melody_proxy(groups_by_piece: list[list[dict[str, Any]]]) -> dict[str, Any]:
    movements: list[int] = []
    representative_pitches: list[int] = []
    repeated_pitch_count = 0
    comparisons = 0
    for groups in groups_by_piece:
        representatives = [max(pitch for pitch, _ in group["notes"]) for group in groups if group["notes"]]
        representative_pitches.extend(representatives)
        for previous, current in zip(representatives, representatives[1:]):
            interval = abs(current - previous)
            movements.append(interval)
            comparisons += 1
            repeated_pitch_count += int(current == previous)
    movement_distribution = {
        "repeated_pitch": float(sum(value == 0 for value in movements) / len(movements) * 100) if movements else 0.0,
        "1_2_semitones": float(sum(1 <= value <= 2 for value in movements) / len(movements) * 100) if movements else 0.0,
        "3_5_semitones": float(sum(3 <= value <= 5 for value in movements) / len(movements) * 100) if movements else 0.0,
        "6_12_semitones": float(sum(6 <= value <= 12 for value in movements) / len(movements) * 100) if movements else 0.0,
        "over_12_semitones": float(sum(value > 12 for value in movements) / len(movements) * 100) if movements else 0.0,
    }
    summary = _numeric_summary(movements)
    summary["mean_absolute_movement"] = summary.pop("mean")
    return {
        "definition": "Highest pitch of each unique onset; diagnostic melody proxy, not the true melody.",
        "movement_statistics": summary,
        "movement_percentages": movement_distribution,
        "repeated_representative_pitch_percentage": float(repeated_pitch_count / comparisons * 100) if comparisons else 0.0,
        "representative_pitch_count": len(representative_pitches),
    }


def _harmony(groups_by_piece: list[list[dict[str, Any]]]) -> dict[str, Any]:
    multi_note_groups = [
        group
        for groups in groups_by_piece
        for group in groups
        if len({pitch for pitch, _ in group["notes"]}) >= 2
    ]
    intervals: list[int] = []
    spans: list[int] = []
    clash_one = clash_two = over_octave = 0
    for group in multi_note_groups:
        pitches = sorted({pitch for pitch, _ in group["notes"]})
        if len(pitches) < 2:
            continue
        group_intervals = [right - left for left, right in combinations(pitches, 2)]
        intervals.extend(group_intervals)
        spans.append(max(pitches) - min(pitches))
        clash_one += int(any(interval == 1 for interval in group_intervals))
        clash_two += int(any(interval == 2 for interval in group_intervals))
        over_octave += int(any(interval > 12 for interval in group_intervals))
    total_groups = len(multi_note_groups)
    interval_classes = Counter(interval % 12 for interval in intervals)
    return {
        "multi_note_onsets": total_groups,
        "average_pitches_per_multi_note_onset": float(np.mean([len(group["notes"]) for group in multi_note_groups])) if multi_note_groups else 0.0,
        "percentage_with_1_semitone_interval": float(clash_one / total_groups * 100) if total_groups else 0.0,
        "percentage_with_2_semitone_interval": float(clash_two / total_groups * 100) if total_groups else 0.0,
        "percentage_with_interval_over_octave": float(over_octave / total_groups * 100) if total_groups else 0.0,
        "most_common_interval_classes": _frequency_table(
            [int(value % 12) for value in intervals], limit=12
        ),
        "interval_class_counts": {str(value): int(count) for value, count in sorted(interval_classes.items())},
        "pitch_span_statistics": _numeric_summary(spans),
    }


def _repetition(groups_by_piece: list[list[dict[str, Any]]]) -> dict[str, Any]:
    pitch_set_comparisons = pitch_set_repeats = melody_comparisons = melody_repeats = 0
    for groups in groups_by_piece:
        pitch_sets = [tuple(sorted({pitch for pitch, _ in group["notes"]})) for group in groups]
        representatives = [max(pitch_set) for pitch_set in pitch_sets if pitch_set]
        for previous, current in zip(pitch_sets, pitch_sets[1:]):
            pitch_set_comparisons += 1
            pitch_set_repeats += int(previous == current)
        for previous, current in zip(representatives, representatives[1:]):
            melody_comparisons += 1
            melody_repeats += int(previous == current)
    return {
        "consecutive_identical_onset_pitch_sets": pitch_set_repeats,
        "pitch_set_comparisons": pitch_set_comparisons,
        "repeated_pitch_set_rate_percentage": float(pitch_set_repeats / pitch_set_comparisons * 100) if pitch_set_comparisons else 0.0,
        "consecutive_identical_representative_pitches": melody_repeats,
        "melody_proxy_comparisons": melody_comparisons,
        "repeated_melody_proxy_rate_percentage": float(melody_repeats / melody_comparisons * 100) if melody_comparisons else 0.0,
    }


def analyze_collection(pieces: list[dict[str, Any]], grid_quarter_length: float) -> dict[str, Any]:
    """Aggregate piece analyses while preserving piece boundaries for transitions."""
    events = [event for piece in pieces for event in piece["events"]]
    groups_by_piece = [piece["groups"] for piece in pieces]
    groups = [group for piece_groups in groups_by_piece for group in piece_groups]
    notes_per_onset = [len(group["notes"]) for group in groups]
    durations = [duration for piece in pieces for duration in piece["durations"]]
    deltas = [delta for piece in pieces for delta in piece["onset_deltas"]]
    pitches = [pitch for piece in pieces for pitch in piece["pitches"]]
    melody = _melody_proxy(groups_by_piece)
    harmony = _harmony(groups_by_piece)
    total_duration = sum(piece["duration_quarter_lengths"] for piece in pieces)
    strong_imbalance = {
        "single_note_onset_percentage": _onset_distribution(notes_per_onset)["percentages"].get("1_note", 0.0),
        "duration_one_step_percentage": _percentage(durations, 1),
        "delta_one_step_percentage": _percentage(deltas, 1),
        "top_duration_values": _frequency_table(durations, 5),
        "top_delta_values": _frequency_table(deltas, 5),
    }
    return {
        "total_notes": len(events),
        "total_unique_onsets": len(groups),
        "notes_per_onset": _onset_distribution(notes_per_onset),
        "rhythm": {
            "delta_steps": {
                **_numeric_summary(deltas),
                "percentage_delta_zero": _percentage(deltas, 0),
                "percentage_delta_one_step": _percentage(deltas, 1),
                "top_20": _frequency_table(deltas),
            },
            "duration_steps": {
                **_numeric_summary(durations),
                "percentage_duration_one_step": _percentage(durations, 1),
                "top_20": _frequency_table(durations),
            },
        },
        "pitch_register": _pitch_register(pitches),
        "melody_proxy": melody,
        "simultaneous_harmony": harmony,
        "repetition": _repetition(groups_by_piece),
        "notes_per_quarter_length": float(len(events) / total_duration) if total_duration else 0.0,
        "unique_onsets_per_quarter_length": float(len(groups) / total_duration) if total_duration else 0.0,
        "total_duration_quarter_lengths": float(total_duration),
        "strong_imbalance_indicators": strong_imbalance,
        "grid_quarter_length": grid_quarter_length,
    }


def _public_piece_statistics(piece: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": piece["path"],
        "notes": piece["notes"],
        "unique_onsets": piece["unique_onsets"],
        "average_notes_per_onset": float(np.mean(piece["notes_per_onset"])) if piece["notes_per_onset"] else 0.0,
        "pitch_minimum": min(piece["pitches"]) if piece["pitches"] else None,
        "pitch_maximum": max(piece["pitches"]) if piece["pitches"] else None,
        "average_duration_steps": piece["average_duration_steps"],
        "average_onset_delta_steps": piece["average_onset_delta_steps"],
    }


def compare_generated_v2(
    grid_quarter_length: float, outputs_directory: Path = OUTPUTS_DIRECTORY
) -> dict[str, Any]:
    """Analyze available V2-generated tracks using the same diagnostic metrics."""
    generated_pieces: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for midi_path in sorted(outputs_directory.glob("classical_v2_*.mid")):
        try:
            generated_pieces.append(analyze_piece(midi_path, grid_quarter_length))
        except Exception as error:
            failures.append({"path": _relative_path(midi_path), "error": str(error)})
    return {
        "files_analyzed": len(generated_pieces),
        "files_failed": failures,
        "tracks": [_public_piece_statistics(piece) for piece in generated_pieces],
        "aggregate": analyze_collection(generated_pieces, grid_quarter_length) if generated_pieces else None,
    }


def build_report(
    files: list[Path], limit: int, grid_quarter_length: float
) -> dict[str, Any]:
    analyzed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for midi_path in files[:limit]:
        try:
            analyzed.append(analyze_piece(midi_path, grid_quarter_length))
        except Exception as error:
            failures.append({"path": _relative_path(midi_path), "error": str(error)})

    report = {
        "diagnostic": "V4A musical dataset diagnostic",
        "genre": "classical",
        "grid_quarter_length": grid_quarter_length,
        "files_discovered": len(files),
        "files_requested": limit,
        "files_analyzed": len(analyzed),
        "files_failed": failures,
        "piece_statistics": [_public_piece_statistics(piece) for piece in analyzed],
        "aggregate": analyze_collection(analyzed, grid_quarter_length) if analyzed else None,
        "generated_v2_comparison": compare_generated_v2(grid_quarter_length),
        "interpretation": {
            "measured_facts_only": True,
            "melody_proxy_warning": "Highest pitch per onset is a diagnostic proxy and is not claimed to be the true melody.",
        },
    }
    return report


def print_summary(report: dict[str, Any], output_path: Path) -> None:
    aggregate = report["aggregate"]
    print("V4A training-music diagnostic")
    print(f"Files analyzed: {report['files_analyzed']} / {report['files_discovered']} discovered")
    print(f"Total notes: {aggregate['total_notes'] if aggregate else 0}")
    print(f"Total unique onsets: {aggregate['total_unique_onsets'] if aggregate else 0}")
    if aggregate:
        onset_stats = aggregate["notes_per_onset"]["statistics"]
        print(
            "Notes/onset: "
            f"mean {onset_stats['mean']:.3f}, median {onset_stats['median']:.3f}, "
            f"p90 {onset_stats['p90']:.3f}, max {onset_stats['maximum']:.0f}"
        )
        rhythm = aggregate["rhythm"]
        print(
            "Rhythm: "
            f"{rhythm['delta_steps']['unique_values']} delta values, "
            f"{rhythm['duration_steps']['unique_values']} duration values, "
            f"delta=1 {rhythm['delta_steps']['percentage_delta_one_step']:.2f}%, "
            f"duration=1 {rhythm['duration_steps']['percentage_duration_one_step']:.2f}%"
        )
        pitch = aggregate["pitch_register"]
        print(
            f"Pitch/register: MIDI {pitch['minimum']}–{pitch['maximum']}, "
            f"mean {pitch['mean']:.2f}, unique {pitch['unique_midi_pitches']}"
        )
        harmony = aggregate["simultaneous_harmony"]
        print(
            "Harmony: "
            f"{harmony['multi_note_onsets']} multi-note onsets, "
            f"1-semitone clash {harmony['percentage_with_1_semitone_interval']:.2f}%"
        )
        repetition = aggregate["repetition"]
        print(
            "Repetition: "
            f"pitch-set rate {repetition['repeated_pitch_set_rate_percentage']:.2f}%, "
            f"melody-proxy rate {repetition['repeated_melody_proxy_rate_percentage']:.2f}%"
        )
    generated = report["generated_v2_comparison"]
    print(f"V2 generated tracks compared: {generated['files_analyzed']}")
    if generated["aggregate"] and aggregate:
        training_density = aggregate["notes_per_quarter_length"]
        generated_density = generated["aggregate"]["notes_per_quarter_length"]
        print(
            f"Notes/quarter: training {training_density:.3f}, "
            f"generated V2 aggregate {generated_density:.3f}"
        )
    print(f"Machine-readable report: {output_path}")
    print("Per-piece statistics are stored in the JSON report.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze original MIDI training music for V4A.")
    parser.add_argument("--genre", default="classical", help="Raw MIDI genre folder to analyze.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_FILE_LIMIT,
        help=f"Maximum files to analyze in deterministic order (default: {DEFAULT_FILE_LIMIT}).",
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=GRID_QUARTER_LENGTH,
        help="Quantization grid in quarter lengths (default: 0.25).",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.grid <= 0:
        parser.error("--grid must be greater than zero.")

    discovered_files = discover_midi_files(genre=args.genre)
    report = build_report(discovered_files, args.limit, args.grid)
    output_path = OUTPUT_DIRECTORY / f"{args.genre}_v4_diagnostic.json"
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print_summary(report, output_path)


if __name__ == "__main__":
    main()
