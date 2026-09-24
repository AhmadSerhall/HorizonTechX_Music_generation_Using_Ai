"""Optional local WAV preview rendering for generated MIDI files."""

import os
import shutil
import subprocess
from pathlib import Path


def render_midi_preview(midi_path: Path) -> tuple[Path | None, str | None]:
    """Render a cached WAV with FluidSynth when a local soundfont is configured."""
    preview_directory = midi_path.parent / "previews"
    preview_path = preview_directory / f"{midi_path.stem}.wav"

    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path, None

    fluidsynth = shutil.which("fluidsynth")
    soundfont_value = os.environ.get("NEURATUNE_SOUNDFONT")
    soundfont_path = Path(soundfont_value) if soundfont_value else None

    if not fluidsynth:
        return None, "FluidSynth is not installed or not on PATH."
    if soundfont_path is None or not soundfont_path.is_file():
        return None, "Set NEURATUNE_SOUNDFONT to the path of a General MIDI .sf2 soundfont."

    preview_directory.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                fluidsynth,
                "-ni",
                str(soundfont_path),
                str(midi_path),
                "-F",
                str(preview_path),
                "-r",
                "44100",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "Local MIDI preview could not be rendered."

    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path, None
    return None, "FluidSynth did not create a WAV preview."
