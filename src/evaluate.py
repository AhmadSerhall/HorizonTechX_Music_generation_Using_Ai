"""Evaluate basic musical properties of a generated MIDI file."""

import argparse
from collections import Counter
from pathlib import Path

from music21 import chord, converter, note, tempo


PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyze_midi(midi_path: Path) -> dict[str, object]:
    """Parse a MIDI file and return lightweight note, pitch, and duration metrics."""
    score = converter.parse(midi_path)
    pitches = []
    note_durations: list[float] = []
    note_count = 0
    chord_count = 0

    for element in score.flatten().notes:
        if isinstance(element, note.Note):
            element_pitches = [element.pitch]
            note_count += 1
        elif isinstance(element, chord.Chord):
            element_pitches = list(element.pitches)
            chord_count += 1
        else:
            continue

        duration = float(element.duration.quarterLength)
        pitches.extend(element_pitches)
        note_durations.extend([duration] * len(element_pitches))

    pitch_class_counts = Counter(pitch.pitchClass for pitch in pitches)
    duration_counts = Counter(note_durations)
    tempo_marks = list(score.recurse().getElementsByClass(tempo.MetronomeMark))

    metrics: dict[str, object] = {
        "total_notes": len(pitches),
        "unique_pitches": len({pitch.nameWithOctave for pitch in pitches}),
        "lowest_pitch": None,
        "highest_pitch": None,
        "total_duration": float(score.highestTime),
        "average_note_duration": 0.0,
        "shortest_note_duration": 0.0,
        "longest_note_duration": 0.0,
        "note_count": note_count,
        "chord_count": chord_count,
        "tempo_bpm": tempo_marks[0].number if tempo_marks else None,
        "pitch_class_distribution": {
            PITCH_CLASS_NAMES[pitch_class]: pitch_class_counts[pitch_class]
            for pitch_class in range(12)
            if pitch_class_counts[pitch_class]
        },
        "duration_distribution": {
            str(duration): duration_counts[duration] for duration in sorted(duration_counts)
        },
    }

    if pitches:
        metrics["lowest_pitch"] = min(pitches, key=lambda pitch: pitch.midi).nameWithOctave
        metrics["highest_pitch"] = max(pitches, key=lambda pitch: pitch.midi).nameWithOctave
        metrics["average_note_duration"] = sum(note_durations) / len(note_durations)
        metrics["shortest_note_duration"] = min(note_durations)
        metrics["longest_note_duration"] = max(note_durations)

    return metrics


def _format_distribution(distribution: dict[str, int]) -> str:
    """Format a small distribution for concise terminal output."""
    return ", ".join(f"{name}: {count}" for name, count in distribution.items()) or "none"


def main() -> None:
    """Print evaluation metrics for one generated MIDI file."""
    parser = argparse.ArgumentParser(description="Evaluate a generated MIDI file.")
    parser.add_argument("midi_path", type=Path, help="Path to a generated .mid or .midi file.")
    args = parser.parse_args()

    if not args.midi_path.is_file():
        parser.error(f"MIDI file not found: {args.midi_path}")

    metrics = analyze_midi(args.midi_path)
    print("Generated MIDI evaluation")
    print(f"Total generated notes: {metrics['total_notes']}")
    print(f"Unique pitches: {metrics['unique_pitches']}")
    print(f"Lowest pitch: {metrics['lowest_pitch'] or 'not available'}")
    print(f"Highest pitch: {metrics['highest_pitch'] or 'not available'}")
    print(f"Total duration (quarter lengths): {metrics['total_duration']}")
    print(f"Average note duration: {metrics['average_note_duration']}")
    print(f"Shortest note duration: {metrics['shortest_note_duration']}")
    print(f"Longest note duration: {metrics['longest_note_duration']}")
    print(f"Note objects: {metrics['note_count']}")
    print(f"Chord objects: {metrics['chord_count']}")
    print(f"Tempo: {metrics['tempo_bpm'] or 'not available'} BPM")
    print("Pitch-class distribution:")
    print(_format_distribution(metrics["pitch_class_distribution"]))
    print("Duration distribution:")
    print(_format_distribution(metrics["duration_distribution"]))


if __name__ == "__main__":
    main()
