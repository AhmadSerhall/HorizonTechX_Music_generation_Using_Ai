"""Local WAV preview rendering for generated MIDI files."""

import math
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
from music21 import chord, converter, note, tempo


def _render_basic_preview(midi_path: Path, preview_path: Path) -> tuple[Path | None, str | None]:
    """Render a simple local sine-wave preview when FluidSynth is unavailable."""
    try:
        score = converter.parse(midi_path)
        tempo_marks = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
        bpm = tempo_marks[0].number if tempo_marks and tempo_marks[0].number else 100
        seconds_per_quarter = 60 / bpm
        sample_rate = 22050
        total_samples = max(1, int((score.highestTime * seconds_per_quarter + 0.2) * sample_rate))
        audio = np.zeros(total_samples, dtype=np.float32)

        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                pitches = [element.pitch]
            elif isinstance(element, chord.Chord):
                pitches = list(element.pitches)
            else:
                continue

            start = int(float(element.offset) * seconds_per_quarter * sample_rate)
            length = max(1, int(float(element.duration.quarterLength) * seconds_per_quarter * sample_rate))
            end = min(start + length, total_samples)
            sample_count = end - start
            if sample_count <= 0:
                continue

            time_values = np.arange(sample_count, dtype=np.float32) / sample_rate
            fade_samples = min(int(sample_rate * 0.01), sample_count // 2)
            envelope = np.ones(sample_count, dtype=np.float32)
            if fade_samples:
                fade = np.linspace(0, 1, fade_samples, dtype=np.float32)
                envelope[:fade_samples] = fade
                envelope[-fade_samples:] = fade[::-1]

            for pitch in pitches:
                frequency = 440 * math.pow(2, (pitch.midi - 69) / 12)
                audio[start:end] += 0.10 * np.sin(2 * math.pi * frequency * time_values) * envelope

        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio *= 0.85 / peak
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(preview_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio * 32767).astype("<i2").tobytes())
        return preview_path, "Basic local MIDI preview"
    except Exception:
        return None, "Local audio preview could not be rendered."


def _fluidsynth_command() -> str | None:
    """Find an optional user-configured FluidSynth executable or PATH command."""
    configured_path = os.environ.get("FLUIDSYNTH_PATH")
    if configured_path:
        configured = Path(configured_path)
        if configured.is_file():
            return str(configured)
        command = shutil.which(configured_path)
        if command:
            return command
    return shutil.which("fluidsynth")


def render_midi_preview(midi_path: Path) -> tuple[Path | None, str | None]:
    """Render a cached WAV with FluidSynth or a built-in local fallback."""
    preview_directory = midi_path.parent / "previews"
    fluidsynth_preview = preview_directory / f"{midi_path.stem}_fluidsynth.wav"
    basic_preview = preview_directory / f"{midi_path.stem}_basic.wav"
    legacy_preview = preview_directory / f"{midi_path.stem}.wav"
    fluidsynth = _fluidsynth_command()
    soundfont_value = os.environ.get("SOUNDFONT_PATH") or os.environ.get("NEURATUNE_SOUNDFONT")
    soundfont_path = Path(soundfont_value) if soundfont_value else None

    if fluidsynth and soundfont_path is not None and soundfont_path.is_file():
        if fluidsynth_preview.exists() and fluidsynth_preview.stat().st_size > 0:
            return fluidsynth_preview, "Cached FluidSynth piano preview"
        preview_directory.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    fluidsynth,
                    "-ni",
                    str(soundfont_path),
                    str(midi_path),
                    "-F",
                    str(fluidsynth_preview),
                    "-r",
                    "44100",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            if fluidsynth_preview.exists() and fluidsynth_preview.stat().st_size > 0:
                return fluidsynth_preview, "FluidSynth piano preview"

    if basic_preview.exists() and basic_preview.stat().st_size > 0:
        return basic_preview, "Cached basic local MIDI preview"
    if legacy_preview.exists() and legacy_preview.stat().st_size > 0:
        return legacy_preview, "Cached basic local MIDI preview"
    return _render_basic_preview(midi_path, basic_preview)
