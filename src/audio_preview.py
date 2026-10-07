"""Local WAV preview rendering for generated MIDI files."""

import hashlib
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
    """Find an optional FluidSynth executable on Windows or another OS."""
    configured_value = os.environ.get("FLUIDSYNTH_PATH")
    if configured_value:
        configured_value = configured_value.strip().strip('"')
        configured = Path(configured_value)
        candidates = [configured]
        if configured.is_dir():
            candidates.extend((configured / "fluidsynth.exe", configured / "fluidsynth"))
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        command = shutil.which(configured_value)
        if command:
            return command

    for command_name in ("fluidsynth.exe", "fluidsynth"):
        command = shutil.which(command_name)
        if command:
            return command
    return None


def _soundfont_path() -> Path | None:
    """Return the preferred, validated local SoundFont configuration."""
    configured_value = os.environ.get("NEURATUNE_SOUNDFONT") or os.environ.get("SOUNDFONT_PATH")
    if not configured_value:
        return None
    candidate = Path(configured_value.strip().strip('"'))
    if not candidate.is_file() or candidate.suffix.lower() not in {".sf2", ".sf3"}:
        return None
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(midi_path: Path, renderer: str, extra: str = "") -> str:
    """Tie preview filenames to the exact MIDI bytes and renderer settings."""
    identity = f"{renderer}|{extra}|{_sha256_file(midi_path)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _path_identity(path: Path) -> str:
    """Identify a renderer binary/resource without copying it into the cache key."""
    try:
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        return str(path)


def _valid_wav(path: Path, sample_rate: int, channels: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                wav_file.getframerate() == sample_rate
                and wav_file.getnchannels() == channels
                and wav_file.getsampwidth() == 2
                and wav_file.getnframes() > 0
            )
    except (OSError, wave.Error):
        return False


def render_midi_preview(midi_path: Path) -> tuple[Path | None, str | None]:
    """Render a read-only MIDI preview with FluidSynth or a local fallback."""
    preview_directory = midi_path.parent / "previews"
    fluidsynth = _fluidsynth_command()
    soundfont_path = _soundfont_path()

    if fluidsynth and soundfont_path is not None:
        soundfont_identity = f"{_path_identity(Path(fluidsynth))}|{_path_identity(soundfont_path)}"
        cache_key = _cache_key(midi_path, "fluidsynth-v1", soundfont_identity)
        fluidsynth_preview = preview_directory / f"{midi_path.stem}_{cache_key}_fluidsynth.wav"
        if _valid_wav(fluidsynth_preview, 44100, 2):
            return fluidsynth_preview, "High-quality SoundFont preview"
        preview_directory.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    fluidsynth,
                    "-ni",
                    "-F",
                    str(fluidsynth_preview),
                    "-r",
                    "44100",
                    str(soundfont_path),
                    str(midi_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        if _valid_wav(fluidsynth_preview, 44100, 2):
            return fluidsynth_preview, "High-quality SoundFont preview"

    cache_key = _cache_key(midi_path, "basic-v2", "22050|mono|pcm16")
    basic_preview = preview_directory / f"{midi_path.stem}_{cache_key}_basic.wav"
    if _valid_wav(basic_preview, 22050, 1):
        return basic_preview, "Basic local MIDI preview"
    return _render_basic_preview(midi_path, basic_preview)
