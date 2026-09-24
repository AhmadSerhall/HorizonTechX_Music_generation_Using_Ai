"""Discover and validate MIDI files for the AI Music Generation project."""

from pathlib import Path

from music21 import converter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_EXTENSIONS = {".mid", ".midi"}


def discover_midi_files(
    data_directory: Path = RAW_DATA_DIR, genre: str | None = None
) -> list[Path]:
    """Return MIDI files found recursively, optionally within one genre folder."""
    search_directory = data_directory / genre if genre else data_directory

    if not search_directory.exists():
        return []

    return sorted(
        path
        for path in search_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def find_non_midi_files(data_directory: Path = RAW_DATA_DIR) -> list[Path]:
    """Return non-hidden files that are not supported MIDI files."""
    if not data_directory.exists():
        return []

    return sorted(
        path
        for path in data_directory.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() not in SUPPORTED_EXTENSIONS
    )


def validate_midi_files(midi_files: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Parse MIDI files with music21 and separate valid and unreadable files."""
    valid_files: list[Path] = []
    invalid_files: list[tuple[Path, str]] = []

    for midi_file in midi_files:
        try:
            converter.parse(midi_file)
        except Exception as error:
            message = str(error).strip() or error.__class__.__name__
            invalid_files.append((midi_file, message))
        else:
            valid_files.append(midi_file)

    return valid_files, invalid_files


def _display_path(path: Path) -> str:
    """Show paths relative to the project when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    """Print a concise summary of the raw MIDI dataset."""
    midi_files = discover_midi_files()
    classical_files = discover_midi_files(genre="classical")
    jazz_files = discover_midi_files(genre="jazz")
    non_midi_files = find_non_midi_files()
    valid_files, invalid_files = validate_midi_files(midi_files)

    print("MIDI dataset summary")
    print(f"Total MIDI files: {len(midi_files)}")
    print(f"Classical files: {len(classical_files)}")
    print(f"Jazz files: {len(jazz_files)}")
    print(f"Valid files: {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")
    print(f"Non-MIDI files ignored: {len(non_midi_files)}")

    for midi_file, error in invalid_files:
        print(f"Invalid: {_display_path(midi_file)} ({error})")

    for file_path in non_midi_files:
        print(f"Ignored non-MIDI file: {_display_path(file_path)}")


if __name__ == "__main__":
    main()
