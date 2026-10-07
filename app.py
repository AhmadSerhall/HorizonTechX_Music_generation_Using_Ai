"""NeuraTune: a compact Streamlit studio for AI MIDI generation."""

import base64
import hashlib
import json
import uuid
import wave
from datetime import datetime
from io import BytesIO
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from music21 import chord, converter, note, tempo

from src.audio_preview import render_midi_preview
from src.evaluate import PITCH_CLASS_NAMES, analyze_midi
from src.generate_v4 import (
    MODELS as MODELS_DIR,
    OUTPUTS as OUTPUTS_DIR,
    PROCESSED as PROCESSED_DATA_DIR,
    _load_artifacts as load_v4_generation_artifacts,
    generate as generate_v4,
)


GENRE = "classical"
HISTORY_PATH = OUTPUTS_DIR / "history.json"
HISTORY_LIMIT = 10
TEMPERATURE_PRESETS = (("Focused", 0.8), ("Balanced", 1.0), ("Experimental", 1.1))
PRODUCTION_MODEL_PATH = MODELS_DIR / f"{GENRE}_lstm_v4_production.keras"
TRAINING_METADATA_PATH = MODELS_DIR / f"{GENRE}_training_metadata_v4_production.json"
V4_METADATA_PATH = PROCESSED_DATA_DIR / f"{GENRE}_v4_metadata.json"
V4_SEQUENCES_PATH = PROCESSED_DATA_DIR / f"{GENRE}_v4_sequences.npz"


def apply_theme() -> None:
    """Apply a compact dark music-studio design system."""
    st.markdown(
        """
        <style>
        :root { --bg: #070B0F; --surface: #0D1418; --surface-2: #111B20; --line: rgba(148,163,184,.14); --text: #F8FAFC; --muted: #94A3B8; --teal: #14B8A6; --teal-bright: #2DD4BF; --sky: #38BDF8; --green: #22C55E; }
        #MainMenu, footer { visibility: hidden; }
        /* Keep Streamlit's local-development toolbar (including Deploy) from
           colliding with NeuraTune's own status area during demos. */
        [data-testid="stToolbar"] { display: none !important; }
        header[data-testid="stHeader"] { background: transparent; }
        .stApp { background: radial-gradient(circle at 82% -10%, rgba(20,184,166,.12), transparent 32%), #070B0F; color: var(--text); }
        .block-container { max-width: 1460px; min-height: 100vh; padding: 1rem 1.25rem 1.1rem; }
        h1, h2, h3, p, label, .stMarkdown { color: var(--text); }
        h2, h3 { letter-spacing: -.02em; }
        div[data-testid="stWidgetLabel"] p, div[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: linear-gradient(145deg, rgba(17,27,32,.94), rgba(13,20,24,.94)); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 14px 40px rgba(0,0,0,.18); }
        div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: .55rem .7rem; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: #0B1115 !important; border-color: var(--line) !important; color: var(--text) !important; }
        div[data-baseweb="select"] span, div[data-baseweb="input"] input { color: var(--text) !important; }
        div[data-baseweb="select"] svg { fill: var(--muted); }
        [data-testid="stSelectbox"], [data-testid="stSelectbox"] * { cursor: pointer !important; user-select: none; }
        [data-testid="stSelectbox"] input { caret-color: transparent !important; }
        [data-testid="stSlider"] [role="slider"] { background: var(--teal) !important; border-color: var(--teal-bright) !important; }
        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--teal); }
        .stButton > button, [data-testid="stDownloadButton"] > button { border: 0; border-radius: 12px; color: #fff; font-weight: 700; background: linear-gradient(95deg, #14B8A6, #38BDF8); box-shadow: 0 8px 22px rgba(20,184,166,.20); }
        .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover { border: 0; color: #fff; filter: brightness(1.08); }
        .stButton > button[kind="secondary"] { border: 1px solid var(--line); background: #111B20; box-shadow: none; color: var(--muted); }
        .stButton > button[kind="secondary"]:hover { border-color: rgba(45,212,191,.45); color: var(--text); }
        button[kind="primary"] { background: linear-gradient(95deg, #14B8A6, #38BDF8) !important; }
        button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; padding: .45rem .75rem; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #CCFBF1; border-bottom-color: var(--teal); }
        .studio-header { min-height: 78px; display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .65rem 0 1rem; border-bottom: 1px solid var(--line); margin-bottom: 1rem; }
        .brand { display: flex; align-items: center; gap: .7rem; }
        .note-mark { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 12px; background: linear-gradient(145deg, #14B8A6, #38BDF8); font-size: 1.35rem; box-shadow: 0 8px 20px rgba(20,184,166,.25); }
        .brand-name { color: var(--text); font-size: 1.18rem; font-weight: 800; letter-spacing: -.03em; }
        .brand-subtitle { color: var(--muted); font-size: .78rem; margin-top: .1rem; }
        .header-copy { color: var(--muted); font-size: .76rem; margin: .25rem 0 0 3.05rem; }
        .header-right { text-align: right; }
        .ready { color: #BBF7D0; font-size: .78rem; font-weight: 700; }
        .ready-dot { color: var(--green); }
        .badge { display: inline-block; margin: .32rem 0 0 .25rem; padding: .22rem .5rem; border: 1px solid rgba(20,184,166,.32); border-radius: 999px; background: rgba(20,184,166,.10); color: #99F6E4; font-size: .67rem; font-weight: 700; }
        .card-title { color: #99F6E4; font-size: .69rem; font-weight: 800; letter-spacing: .12em; margin: .1rem 0 .5rem; }
        .temperature-state { color: #99F6E4; font-weight: 700; font-size: .8rem; }
        .control-help { color: var(--muted); font-size: .73rem; margin-top: -.4rem; }
        .settings-context { color: var(--muted); font-size: .72rem; margin: -.15rem 0 .8rem; }
        .player-top { display: flex; align-items: center; gap: 1rem; min-height: 150px; }
        .composition-copy { flex: 1; min-width: 0; }
        .album-art { box-sizing: border-box; flex: 0 0 126px; width: 126px; height: 126px; position: relative; overflow: hidden; border-radius: 16px; border: 1px solid rgba(45,212,191,.24); background: radial-gradient(circle at var(--record-light-x) var(--record-light-y), rgba(20,184,166,.22), transparent 65%), #0A141B; box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 8px 24px rgba(0,0,0,.22); }
        .record-disc { position: absolute; inset: 10px; border-radius: 50%; border: 1px solid rgba(125,211,252,.2); background: repeating-radial-gradient(circle, transparent 0 4px, rgba(125,211,252,.07) 4px 5px), conic-gradient(from var(--record-angle), #0C1C25, #18383E, #0B1822, #16434A, #0C1C25); box-shadow: 0 0 15px rgba(20,184,166,.1); }
        .record-orbit { position: absolute; inset: 5px; border: 1px solid transparent; border-top-color: rgba(94,234,212,.5); border-right-color: rgba(56,189,248,.2); border-radius: 50%; transform: rotate(var(--record-angle)); }
        .record-orbit-inner { inset: 21px; border-top-color: rgba(125,211,252,.35); border-right-color: transparent; border-bottom-color: rgba(45,212,191,.22); transform: rotate(var(--record-inner-angle)); }
        .record-orbit-ambient { animation: record-orbit 90s linear infinite; }
        .record-center { position: absolute; inset: 43px; display: grid; place-items: center; border-radius: 50%; border: 1px solid rgba(94,234,212,.36); color: #B5EEE8; background: radial-gradient(circle at 30% 25%, #1E5961, #102630); box-shadow: 0 0 14px rgba(45,212,191,.15); font-size: .55rem; font-weight: 700; letter-spacing: .12em; padding-left: .12em; }
        .composition-enter { animation: composition-arrive .5s ease-out both; }
        @keyframes record-orbit { from { transform: rotate(var(--record-angle)); } to { transform: rotate(calc(var(--record-angle) + 360deg)); } }
        @keyframes composition-arrive { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
        .composition-kicker { color: #7DD3FC; font-size: .68rem; font-weight: 800; letter-spacing: .12em; }
        .composition-title { color: var(--text); font-size: 1.18rem; line-height: 1.3; font-weight: 800; margin: .3rem 0 .35rem; overflow-wrap: anywhere; }
        .composition-meta { color: var(--muted); font-size: .8rem; }
        .midi-badge { display: inline-block; margin-top: .55rem; color: #BAE6FD; border: 1px solid rgba(56,189,248,.32); border-radius: 999px; padding: .18rem .46rem; font-size: .67rem; }
        .ambient-sequence-wrap { min-height: 92px; display: grid; place-items: center; margin: .15rem 0 .35rem; }
        .ambient-sequence { width: min(520px, 82%); height: 74px; display: flex; align-items: center; justify-content: center; gap: 5px; padding: .35rem 1rem; border-radius: 16px; background: radial-gradient(circle at 50% 50%, rgba(20,184,166,.08), transparent 68%); overflow: hidden; }
        .ambient-sequence span { width: 6px; height: var(--bar-height); border-radius: 999px; background: linear-gradient(180deg, #67E8F9, #2DD4BF 56%, #38BDF8); opacity: .84; transform-origin: center; animation: ambient-sequence-pulse var(--bar-duration) ease-in-out infinite; animation-delay: var(--bar-delay); animation-play-state: paused; box-shadow: 0 0 10px rgba(45,212,191,.16); }
        .ambient-sequence-wrap.is-playing .ambient-sequence span { animation-play-state: running; }
        @keyframes ambient-sequence-pulse { 0%, 100% { transform: scaleY(.42); opacity: .46; } 50% { transform: scaleY(1); opacity: .98; box-shadow: 0 0 16px rgba(45,212,191,.30); } }
        .listen-label { color: #99F6E4; font-size: .69rem; font-weight: 800; letter-spacing: .10em; margin: .7rem 0 .3rem; }
        [data-testid="stAudio"] { border: 1px solid rgba(45,212,191,.20); border-radius: 16px !important; background: #0B1115; overflow: hidden; }
        [data-testid="stAudio"] audio { display: block; width: 100%; border-radius: 16px !important; }
        .metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: .4rem; margin: .6rem 0; }
        .metric-block { padding: .48rem; border-radius: 10px; border: 1px solid rgba(255,255,255,.08); background: rgba(5,9,20,.42); }
        .metric-value { color: var(--text); font-size: 1rem; font-weight: 800; overflow-wrap: anywhere; }
        .metric-label { color: var(--muted); font-size: .63rem; margin-top: .08rem; }
        .secondary-meta { color: var(--muted); font-size: .71rem; line-height: 1.55; display: flex; flex-wrap: wrap; gap: .25rem .9rem; margin: .7rem 0 .55rem; }
        .empty-player { min-height: 230px; display: grid; place-items: center; text-align: center; color: var(--muted); }
        .empty-icon { color: #2DD4BF; font-size: 2rem; }
        .analysis-title { color: #E2E8F0; font-size: .92rem; font-weight: 800; margin: .1rem 0 .25rem; }
        .loading-backdrop { position: fixed; inset: 0; z-index: 999999; display: grid; place-items: center; background: rgba(3,7,10,.76); backdrop-filter: blur(8px); }
        .loading-card { width: min(330px, calc(100vw - 2rem)); padding: 1.4rem; border: 1px solid rgba(45,212,191,.28); border-radius: 18px; background: linear-gradient(145deg, #111B20, #0D1418); text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,.42); }
        .neural-pulse { width: 42px; height: 42px; margin: 0 auto .75rem; border: 3px solid rgba(45,212,191,.22); border-top-color: #2DD4BF; border-right-color: #38BDF8; border-radius: 50%; animation: spin 1s linear infinite; }
        .loading-title { color: #F8FAFC; font-weight: 800; font-size: 1rem; } .loading-copy { color: #94A3B8; font-size: .78rem; margin-top: .32rem; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 800px) { .block-container { min-height: auto; padding: .75rem .8rem 1.5rem; } .studio-header { align-items: flex-start; } .header-copy { display: none; } .header-right { max-width: 52%; } .badge { font-size: .61rem; } .metric-row { grid-template-columns: repeat(2, 1fr); } .player-top { align-items: flex-start; } }
        @media (max-width: 640px) { .studio-header { flex-wrap: wrap; gap: .65rem; } .header-right { width: 100%; max-width: none; text-align: left; } .header-right .badge { margin-left: 0; margin-right: .25rem; } }
        @media (max-width: 480px) { .player-top { flex-wrap: wrap; gap: .85rem; } .composition-copy { flex-basis: 150px; } .composition-title { font-size: 1.08rem; } .composition-kicker { font-size: .6rem; letter-spacing: .08em; } }
        @media (prefers-reduced-motion: reduce) { .record-orbit-ambient, .composition-enter, .ambient-sequence span { animation: none; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_resources() -> tuple[object, ...]:
    """Cache the frozen V4 production model and its exact artifacts."""
    return load_v4_generation_artifacts(
        GENRE,
        str(PRODUCTION_MODEL_PATH),
        str(TRAINING_METADATA_PATH),
    )


def initialize_session_state() -> None:
    """Preserve the latest generated composition across widget reruns."""
    if "generation_history" not in st.session_state:
        history = load_persistent_history()
        st.session_state.generation_history = history
        set_selected_composition(history[0] if history else None)
        if history:
            save_persistent_history(history)
    else:
        history = st.session_state.generation_history
        for result in history:
            result.setdefault("id", str(uuid.uuid5(uuid.NAMESPACE_URL, str(result.get("midi_path", "")))))
            result.setdefault("display_name", "")
            result.setdefault("audio_preview_path", None)
            result.setdefault("favorite", False)
        for result in history:
            if not result.get("display_name"):
                result["display_name"] = next_display_name(history)
    pending_selection = st.session_state.pop("pending_library_selection", None)
    if pending_selection is not None:
        # Apply this before the selectbox is created on the rerun triggered by
        # generation, so the new composition is immediately selected.
        st.session_state.composition_library_selection = str(pending_selection)
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("latest_midi_path", None)
    st.session_state.setdefault("latest_settings", None)
    st.session_state.setdefault("latest_metrics", None)
    st.session_state.setdefault("delete_confirmation_id", None)


def set_selected_composition(result: dict[str, object] | None) -> None:
    """Update the current player and analysis result from one library entry."""
    st.session_state.latest_result = result
    st.session_state.latest_midi_path = result.get("midi_path") if result else None
    st.session_state.latest_settings = result.get("settings") if result else None
    st.session_state.latest_metrics = result.get("metrics") if result else None


def safe_output_artifact_path(path_value: object, allowed_suffixes: set[str]) -> Path | None:
    """Return an output artifact path only when it stays inside outputs/."""
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = OUTPUTS_DIR / path
    try:
        path.resolve().relative_to(OUTPUTS_DIR.resolve())
    except (OSError, ValueError):
        return None
    return path if path.suffix.lower() in allowed_suffixes else None


def next_display_name(history: list[dict[str, object]]) -> str:
    """Choose the next available beginner-friendly default library name."""
    existing_names = {str(item.get("display_name", "")) for item in history}
    for number in range(1, 1000):
        name = f"Neural Composition {number:03d}"
        if name not in existing_names:
            return name
    return "Neural Composition"


def history_record(result: dict[str, object]) -> dict[str, object]:
    """Convert an in-memory generation result into a JSON-safe history record."""
    settings = result["settings"]
    midi_path = Path(str(result["midi_path"]))
    return {
        "id": str(result.get("id") or uuid.uuid4()),
        "display_name": str(result.get("display_name") or "Neural Composition"),
        "midi_path": str(midi_path),
        "midi_filename": midi_path.name,
        "audio_preview_path": result.get("audio_preview_path"),
        "created_at": result.get("created_at", datetime.now().isoformat(timespec="seconds")),
        "length": settings["length"],
        "temperature": settings["temperature"],
        "tempo": settings["tempo"],
        "seed": settings["seed"],
        "min_duration": settings["min_duration"],
        "generated_events": result["generated_events"],
        "model_version": result.get("model_version", "V1/V2 historical"),
        "seed_sequence_length": result["seed_sequence_length"],
        "skipped_events": result["skipped_events"],
        "metrics": result.get("metrics"),
        "favorite": bool(result.get("favorite", False)),
    }


def result_from_history(record: dict[str, object]) -> dict[str, object] | None:
    """Restore one history record only when its MIDI file is still available."""
    midi_path = Path(str(record.get("midi_path") or record.get("midi_filename", "")))
    if not midi_path.is_absolute():
        midi_path = OUTPUTS_DIR / midi_path
    if not midi_path.is_file():
        return None

    preview_path = safe_output_artifact_path(record.get("audio_preview_path"), {".wav"})

    return {
        "id": str(record.get("id") or uuid.uuid5(uuid.NAMESPACE_URL, str(midi_path))),
        "display_name": str(record.get("display_name") or ""),
        "midi_path": str(midi_path),
        "audio_preview_path": str(preview_path) if preview_path and preview_path.is_file() else None,
        "created_at": record.get("created_at", ""),
        "settings": {
            "length": record.get("length", record.get("generated_events", 0)),
            "temperature": record.get("temperature", 1.0),
            "tempo": record.get("tempo", 100),
            "seed": record.get("seed", 42),
            "min_duration": record.get("min_duration"),
        },
        "generated_events": record.get("generated_events", record.get("length", 0)),
        "model_version": record.get("model_version", "V1/V2 historical"),
        "seed_sequence_length": record.get("seed_sequence_length", 50),
        "skipped_events": record.get("skipped_events", 0),
        "metrics": record.get("metrics"),
        "favorite": bool(record.get("favorite", False)),
    }


def save_persistent_history(history: list[dict[str, object]]) -> None:
    """Store recent compositions in outputs/history.json without a database."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    records = [
        history_record(result)
        for result in history[:HISTORY_LIMIT]
        if "midi_path" in result and "settings" in result
    ]
    temporary_path = HISTORY_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary_path.replace(HISTORY_PATH)


def load_persistent_history() -> list[dict[str, object]]:
    """Load valid output history and safely ignore corrupt or missing entries."""
    if not HISTORY_PATH.is_file():
        return []
    try:
        records = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(records, list):
        return []

    history: list[dict[str, object]] = []
    for record in records:
        if isinstance(record, dict):
            result = result_from_history(record)
            if result:
                history.append(result)
    history.sort(key=lambda result: str(result.get("created_at", "")), reverse=True)
    history = history[:HISTORY_LIMIT]
    for result in history:
        if not result.get("display_name"):
            result["display_name"] = next_display_name(history)
    return history


def delete_composition_artifacts(result: dict[str, object]) -> None:
    """Remove only this composition's MIDI and cached WAV preview from outputs/."""
    midi_path = safe_output_artifact_path(result.get("midi_path"), {".mid", ".midi"})
    preview_path = safe_output_artifact_path(result.get("audio_preview_path"), {".wav"})
    if midi_path and not preview_path:
        preview_path = safe_output_artifact_path(
            midi_path.parent / "previews" / f"{midi_path.stem}.wav", {".wav"}
        )
    for path in (midi_path, preview_path):
        if path and path.is_file():
            path.unlink()


def show_loading_overlay() -> object:
    """Display a lightweight modal-style overlay during neural generation."""
    overlay = st.empty()
    overlay.markdown(
        """
        <div class="loading-backdrop"><div class="loading-card"><div class="note-mark">♪</div>
        <div class="neural-pulse"></div><div class="loading-title">Composing your music...</div>
        <div class="loading-copy">The neural network is generating your sequence</div></div></div>
        """,
        unsafe_allow_html=True,
    )
    return overlay


def model_ready() -> bool:
    """Check whether the backend files needed by the studio are present."""
    return all(
        path.exists()
        for path in (
            PRODUCTION_MODEL_PATH,
            TRAINING_METADATA_PATH,
            V4_METADATA_PATH,
            V4_SEQUENCES_PATH,
        )
    )


def render_header() -> None:
    """Render the compact product header and model-status badges."""
    status = "<span class='ready-dot'>●</span> Model Ready" if model_ready() else "● Model Missing"
    st.markdown(
        f"""
        <div class="studio-header">
          <div>
            <div class="brand"><div class="note-mark">♪</div><div><div class="brand-name">NeuraTune</div><div class="brand-subtitle">AI Music Generation Studio</div></div></div>
          </div>
          <div class="header-right"><div class="ready">{status}</div>
            <span class="badge">LSTM</span><span class="badge">TensorFlow</span><span class="badge">music21</span><span class="badge">MAESTRO</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def creativity_state(temperature: float) -> str:
    """Return a compact human-readable temperature label."""
    if temperature <= 0.7:
        return "Conservative"
    if temperature <= 1.2:
        return "Balanced"
    return "Experimental"


def render_controls() -> tuple[dict[str, int | float | None], bool]:
    """Render a compact settings card for the existing generation controls."""
    with st.container(border=True):
        st.markdown("<div class='card-title'>GENERATION SETTINGS</div>", unsafe_allow_html=True)
        st.markdown("<div class='settings-context'>For your next composition</div>", unsafe_allow_html=True)
        length = st.slider("Composition Length", 50, 500, 200, 10)
        st.markdown("<div class='control-help'>Length is measured as generated musical moments.</div>", unsafe_allow_html=True)
        st.markdown("<div class='control-help'>CREATIVITY PRESET</div>", unsafe_allow_html=True)
        preset_columns = st.columns(3)
        current_temperature = float(st.session_state.get("temperature_control", 1.0))
        for column, (label, value) in zip(preset_columns, TEMPERATURE_PRESETS):
            active = current_temperature == value
            button_label = f"✓ {label}" if active else label
            if column.button(button_label, key=f"preset_{label.lower()}", type="primary" if active else "secondary", use_container_width=True):
                st.session_state.temperature_control = value
                st.rerun()
        temperature = st.slider("Creativity", 0.3, 1.5, 1.0, 0.1, key="temperature_control")
        st.markdown(
            f"<div class='temperature-state'>{temperature:.1f} · {creativity_state(temperature)}</div>",
            unsafe_allow_html=True,
        )
        st.caption("0.3–0.7 conservative · 0.8–1.2 balanced · 1.3–1.5 experimental")
        tempo = st.slider("Tempo", 60, 180, 100, 1, format="%d BPM")

        seed_column, _ = st.columns(2)
        with seed_column:
            seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
        st.markdown(
            "<div class='control-help'>V4 preserves the model-predicted note durations.</div>",
            unsafe_allow_html=True,
        )
        generate_clicked = st.button("✦ Generate Music", type="primary", use_container_width=True)

    settings: dict[str, int | float | None] = {
        "length": length,
        "temperature": temperature,
        "seed": int(seed),
        "tempo": tempo,
        # Retained for compatibility with older history records; V4 does not
        # apply post-generation duration rewriting.
        "min_duration": None,
    }
    return settings, generate_clicked


def generate_composition(settings: dict[str, int | float | None]) -> dict[str, object]:
    """Generate one composition through the frozen V4 grouped-onset pipeline."""
    resources = load_model_resources()
    temperature = float(settings["temperature"])
    output_path = OUTPUTS_DIR / f"neuraltune_v4_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mid"
    arguments = type("V4GenerationArgs", (), {
        "genre": GENRE,
        "length": int(settings["length"]),
        "temperature": temperature,
        "top_k": 10,
        "delta_temperature": 0.8,
        "duration_temperature": 0.8,
        "seed": int(settings["seed"]),
        "tempo": float(settings["tempo"]),
        "model_path": str(PRODUCTION_MODEL_PATH),
        "training_metadata_path": str(TRAINING_METADATA_PATH),
        "output_path": str(output_path),
    })()
    generated = generate_v4(arguments, resources=resources)
    output_path = Path(generated["path"])
    # The V4 export is authoritative. Audio preview rendering reads it without
    # parsing, rewriting, or otherwise mutating the MIDI file.
    diagnostics = generated["diagnostics"]
    result: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "display_name": next_display_name(st.session_state.generation_history),
        "midi_path": str(output_path),
        "audio_preview_path": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "seed_sequence_length": 50,
        "generated_events": int(diagnostics["onset_groups"]),
        "skipped_events": 0,
        "simultaneous_event_count": int(diagnostics["total_notes"] - diagnostics["onset_groups"]),
        "onset_cap_activations": 0,
        "model_version": "V4 Production",
        "favorite": False,
    }
    try:
        result["metrics"] = analyze_midi(output_path)
    except Exception as error:
        result["metrics"] = None
        result["evaluation_error"] = str(error)
    return result


def ambient_sequence_markup() -> str:
    """Return the decorative sequence visual for the listening-area whitespace."""
    heights = [18, 32, 48, 26, 41, 56, 30, 50, 22, 38, 54, 28, 44, 20, 36, 52, 25, 34, 46, 19, 40, 55, 29, 45, 23, 37, 50, 27]
    bars = "".join(
        (
            f"<span style='--bar-height:{height}px;"
            f"--bar-duration:{1.15 + (index % 5) * 0.13:.2f}s;"
            f"--bar-delay:-{(index % 7) * 0.11:.2f}s'></span>"
        )
        for index, height in enumerate(heights)
    )
    return f"<div id='neuraltune-ambient-sequence' class='ambient-sequence-wrap'><div class='ambient-sequence' aria-hidden='true'>{bars}</div></div>"


def neural_record_markup(seed: object) -> str:
    """Decorative record geometry, stable per saved seed and independent of RNG."""
    identity = hashlib.sha256(str(seed).encode("utf-8")).digest()
    angle = int.from_bytes(identity[:2], "big") % 360
    inner_angle = int.from_bytes(identity[2:4], "big") % 360
    light_x, light_y = 20 + identity[4] % 61, 15 + identity[5] % 41
    return (
        f'<div class="album-art" role="img" aria-label="Neural Record — decorative composition artwork" '
        f'style="--record-angle:{angle}deg;--record-inner-angle:{inner_angle}deg;'
        f'--record-light-x:{light_x}%;--record-light-y:{light_y}%">'
        '<div class="record-disc"></div><div class="record-orbit record-orbit-ambient"></div>'
        '<div class="record-orbit record-orbit-inner"></div><div class="record-center"></div></div>'
    )


def player_markup(result: dict[str, object] | None, *, animate: bool = False) -> str:
    """Create the compact music-player visual for empty and generated states."""
    if not result:
        return """
        <div class="empty-player"><div><div class="empty-icon">✦</div>
        <strong>Your composition will appear here</strong><br><span>Configure the studio controls and generate a new sequence.</span></div></div>
        """

    metrics = result.get("metrics") or {}
    settings = result["settings"]
    if result.get("model_version") == "V4 Production":
        minimum_duration = "Model-predicted"
    elif settings["min_duration"] is not None:
        minimum_duration = f"Historical {settings['min_duration']}"
    else:
        minimum_duration = "Original"
    display_name = escape(str(result.get("display_name") or f"Neural Composition #{settings['seed']}"))
    entrance_class = " composition-enter" if animate else ""
    return f"""
    <div class="composition-summary{entrance_class}">
    <div class="player-top">{neural_record_markup(settings['seed'])}<div class="composition-copy">
      <div class="composition-kicker">SELECTED COMPOSITION</div>
      <div class="composition-title">{display_name}</div>
      <div class="composition-meta">Classical Piano · AI Generated</div>
      <div class="midi-badge">MIDI · Generated by NeuraTune</div>
    </div></div>
    <div class="metric-row">
      <div class="metric-block"><div class="metric-value">{result['generated_events']}</div><div class="metric-label">MOMENTS</div></div>
      <div class="metric-block"><div class="metric-value">{settings['temperature']}</div><div class="metric-label">CREATIVITY</div></div>
      <div class="metric-block"><div class="metric-value">{settings['tempo']}</div><div class="metric-label">BPM</div></div>
      <div class="metric-block"><div class="metric-value">{metrics.get('unique_pitches', '—')}</div><div class="metric-label">PITCHES</div></div>
    </div>
    <div class="secondary-meta"><span>Pitch range: {metrics.get('lowest_pitch', '—')}–{metrics.get('highest_pitch', '—')}</span><span>Saved seed: {settings['seed']}</span><span>Note durations: {minimum_duration}</span></div>
    </div>
    """


@st.cache_data(show_spinner=False)
def synthesize_midi_preview(midi_path_string: str, tempo_bpm: float) -> bytes | None:
    """Create a lightweight WAV preview using music21 + NumPy only."""
    try:
        score = converter.parse(midi_path_string)
        spq = 60.0 / max(float(tempo_bpm), 1.0)
        events = []
        for element in score.flatten().notes:
            start = float(element.offset) * spq
            duration = max(float(element.duration.quarterLength) * spq, 0.04)
            pitches = [element.pitch] if isinstance(element, note.Note) else list(element.pitches) if isinstance(element, chord.Chord) else []
            events.extend((start, duration, float(pitch.frequency)) for pitch in pitches)
        if not events:
            return None
        sample_rate = 22050
        end_time = min(max(start + duration for start, duration, _ in events) + .25, 180.0)
        audio = np.zeros(int(end_time * sample_rate) + 1, dtype=np.float32)
        for start, duration, frequency in events:
            if start >= end_time:
                continue
            duration = min(duration, end_time - start)
            count = max(1, int(duration * sample_rate))
            t = np.arange(count, dtype=np.float32) / sample_rate
            tone = np.sin(2*np.pi*frequency*t) + .34*np.sin(4*np.pi*frequency*t) + .16*np.sin(6*np.pi*frequency*t)
            attack = max(1, min(count, int(.012 * sample_rate)))
            env = np.exp(-2.4*t/max(duration,.05)).astype(np.float32)
            env[:attack] *= np.linspace(0, 1, attack, dtype=np.float32)
            i = int(start * sample_rate); j = min(i + count, len(audio))
            audio[i:j] += tone[:j-i] * env[:j-i]
        peak = float(np.max(np.abs(audio)))
        if peak > 0: audio = .88 * audio / peak
        pcm = (audio * 32767).astype(np.int16)
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(sample_rate); wav_file.writeframes(pcm.tobytes())
        return buffer.getvalue()
    except Exception:
        return None

def get_audio_preview(midi_path: Path, tempo_bpm: float) -> tuple[bytes | None, str, Path | None]:
    """Prefer the existing renderer, then fall back to the built-in synth."""
    try:
        preview_path, status = render_midi_preview(midi_path)
        if preview_path and Path(preview_path).is_file():
            preview_path = Path(preview_path)
            return preview_path.read_bytes(), status or 'Local rendered audio preview', preview_path
    except Exception:
        pass
    preview = synthesize_midi_preview(str(midi_path), tempo_bpm)
    if preview:
        return preview, 'Basic local MIDI preview', None
    return None, 'Audio preview could not be created.', None


def audio_preview_label(audio_preview: bytes | None, status: str) -> str:
    """Summarize the active preview renderer without exposing implementation details."""
    if not audio_preview:
        return "Audio preview unavailable"
    if "SoundFont" in status or "FluidSynth" in status:
        return "Audio: High-quality SoundFont preview"
    return "Audio: Basic local MIDI preview"


def render_player() -> None:
    """Render the result card, download action, and small session history."""
    result = st.session_state.latest_result
    with st.container(border=True):
        st.markdown("<div class='card-title'>YOUR AI COMPOSITION</div>", unsafe_allow_html=True)
        # Presentation-only state: replay the entrance when the selected track
        # changes, not when controls for the next generation are adjusted.
        composition_id = str(result.get("id") or result.get("midi_path")) if result else None
        animate = bool(result) and st.session_state.get("displayed_composition_id") != composition_id
        st.markdown(player_markup(result, animate=animate), unsafe_allow_html=True)
        st.session_state.displayed_composition_id = composition_id
        if result:
            midi_path = Path(str(result["midi_path"]))
            if midi_path.exists():
                preview_bytes, preview_status, preview_path = get_audio_preview(
                    midi_path, float(result["settings"].get("tempo", 100))
                )
                if preview_path and result.get("audio_preview_path") != str(preview_path):
                    result["audio_preview_path"] = str(preview_path)
                    save_persistent_history(st.session_state.generation_history)
                if preview_bytes:
                    st.markdown(
                        "<div class='listen-label'>LISTEN TO YOUR COMPOSITION · PLAY / PAUSE</div>",
                        unsafe_allow_html=True,
                    )
                    render_piano_visualizer_player(
                        midi_path,
                        float(result["settings"].get("tempo", 100)),
                        preview_bytes,
                        audio_preview_label(preview_bytes, preview_status),
                    )
                    st.markdown(ambient_sequence_markup(), unsafe_allow_html=True)
                    st.caption(preview_status)
                else:
                    st.warning(preview_status)
                st.download_button(
                    "Download MIDI",
                    data=midi_path.read_bytes(),
                    file_name=midi_path.name,
                    mime="audio/midi",
                    type="primary",
                    use_container_width=True,
                )
                st.caption("Open this MIDI in a DAW, MIDI player, or MuseScore.")
            else:
                st.error("The generated MIDI is no longer available. Generate a new composition.")
            if result.get("evaluation_error"):
                st.warning("MIDI export succeeded, but analysis was unavailable for this composition.")

    render_composition_library()


def render_composition_library() -> None:
    """Show persisted compositions and keep the selected one in the player."""
    history = [item for item in st.session_state.generation_history if "midi_path" in item]
    if not history:
        return

    with st.container(border=True):
        st.markdown("<div class='card-title'>COMPOSITION LIBRARY</div>", unsafe_allow_html=True)
        library_filter = st.radio(
            "Show",
            ["All", "Favorites"],
            horizontal=True,
            label_visibility="collapsed",
            key="composition_library_filter",
        )
        visible_history = [item for item in history if library_filter == "All" or item.get("favorite")]
        if not visible_history:
            st.caption("No favorite compositions yet.")
            return

        identifiers = [str(item["id"]) for item in visible_history]
        labels = {
            str(item["id"]): (
                f"{'★ ' if item.get('favorite') else ''}{item.get('display_name', 'Neural Composition')}"
                f" · {item['settings']['length']} moments"
            )
            for item in visible_history
        }
        current_id = str((st.session_state.latest_result or {}).get("id", identifiers[0]))
        selected_id = st.selectbox(
            "Composition Library",
            identifiers,
            index=identifiers.index(current_id) if current_id in identifiers else 0,
            format_func=lambda identifier: labels[identifier],
            key="composition_library_selection",
        )
        selected = next(item for item in history if str(item["id"]) == selected_id)
        if selected_id != current_id:
            set_selected_composition(selected)
            st.rerun()

        st.caption("The selected composition is ready to play in the player above.")
        name_column, save_column = st.columns([3, 1])
        with name_column:
            new_name = st.text_input(
                "Name",
                value=str(selected.get("display_name", "Neural Composition")),
                key=f"rename_{selected_id}",
                label_visibility="collapsed",
            )
        with save_column:
            rename_clicked = st.button("Save name", key=f"save_name_{selected_id}", use_container_width=True)

        favorite_column, delete_column = st.columns(2)
        with favorite_column:
            favorite_label = "★ Unfavorite" if selected.get("favorite") else "☆ Favorite"
            favorite_clicked = st.button(favorite_label, key=f"favorite_{selected_id}", use_container_width=True)
        with delete_column:
            delete_clicked = st.button("Delete", key=f"delete_{selected_id}", use_container_width=True)

        if rename_clicked:
            clean_name = new_name.strip()
            if clean_name:
                selected["display_name"] = clean_name
                save_persistent_history(history)
                st.rerun()
            st.warning("Enter a composition name before saving.")

        if favorite_clicked:
            selected["favorite"] = not bool(selected.get("favorite"))
            save_persistent_history(history)
            st.rerun()

        if delete_clicked:
            st.session_state.delete_confirmation_id = selected_id
            st.rerun()

        if st.session_state.delete_confirmation_id == selected_id:
            st.warning(f"Delete {selected.get('display_name', 'this composition')} and its generated files?")
            confirm_column, cancel_column = st.columns(2)
            with confirm_column:
                confirm_delete = st.button("Delete permanently", key=f"confirm_delete_{selected_id}", type="primary", use_container_width=True)
            with cancel_column:
                cancel_delete = st.button("Cancel", key=f"cancel_delete_{selected_id}", use_container_width=True)
            if confirm_delete:
                delete_composition_artifacts(selected)
                remaining_history = [item for item in history if str(item["id"]) != selected_id]
                st.session_state.generation_history = remaining_history
                set_selected_composition(remaining_history[0] if remaining_history else None)
                st.session_state.delete_confirmation_id = None
                save_persistent_history(remaining_history)
                st.rerun()
            if cancel_delete:
                st.session_state.delete_confirmation_id = None
                st.rerun()


def render_expandable_figure(figure: object, title: str, height: int = 540) -> None:
    """Render a Matplotlib figure that opens in browser fullscreen when clicked."""
    image_buffer = BytesIO()
    figure.savefig(image_buffer, format="png", dpi=140, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    chart_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    components.html(
        f"""
        <style>
            html, body {{ margin: 0; background: transparent; overflow: hidden; }}
            #chart {{
                display: block; width: 100%; padding: 0; border: 0; border-radius: 12px;
                background: transparent; cursor: zoom-in;
            }}
            #chart img {{ display: block; width: 100%; height: auto; border-radius: 12px; }}
            #chart:focus-visible {{ outline: 2px solid #2DD4BF; outline-offset: 3px; }}
            #chart:fullscreen {{
                width: 100vw; height: 100vh; display: grid; place-items: center;
                background: #070B0F; cursor: zoom-out;
            }}
            #chart:fullscreen img {{
                width: min(96vw, 1500px); max-height: 94vh; object-fit: contain;
                border-radius: 16px;
            }}
        </style>
        <button id="chart" type="button" aria-label="Expand {title}">
            <img src="data:image/png;base64,{chart_image}" alt="{title}. Click to expand." />
        </button>
        <script>
            const chart = document.getElementById("chart");
            const toggleFullscreen = () => {{
                if (document.fullscreenElement) {{
                    document.exitFullscreen();
                }} else if (chart.requestFullscreen) {{
                    chart.requestFullscreen().catch(() => {{}});
                }}
            }};
            chart.addEventListener("click", toggleFullscreen);
            chart.addEventListener("keydown", (event) => {{
                if (event.key === "Enter" || event.key === " ") {{
                    event.preventDefault();
                    toggleFullscreen();
                }}
            }});
        </script>
        """,
        height=height,
        scrolling=False,
    )


def render_distribution_chart(labels: list[str], values: list[int], title: str, color: str) -> None:
    """Render a compact distribution chart using the shared fullscreen behavior."""
    figure, axis = plt.subplots(figsize=(8, 2.15))
    figure.patch.set_facecolor("#0D1418")
    axis.set_facecolor("#0D1418")
    axis.bar(labels, values, color=color, edgecolor="#B6F3ED", linewidth=0.35)
    axis.set_title(title, color="#F8FAFC", loc="left", fontsize=10, pad=8)
    axis.tick_params(colors="#CBD5E1", labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#475569")
    axis.grid(axis="y", color="#334155", alpha=.35)
    render_expandable_figure(figure, title)


@st.cache_data(show_spinner=False)
def extract_piano_timeline(midi_path_string: str, fallback_bpm: float) -> dict[str, object]:
    """Convert selected MIDI notes and chord pitches into a cached visualizer timeline."""
    score = converter.parse(midi_path_string)
    tempo_marks = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
    bpm = next(
        (float(mark.number) for mark in tempo_marks if mark.number and mark.number > 0),
        float(fallback_bpm),
    )
    seconds_per_quarter = 60.0 / bpm
    notes: list[dict[str, float | int | str]] = []
    for element in score.flatten().notes:
        if isinstance(element, note.Note):
            pitches = [element.pitch]
        elif isinstance(element, chord.Chord):
            pitches = list(element.pitches)
        else:
            continue
        offset = float(element.offset)
        duration = max(float(element.duration.quarterLength), 0.01)
        notes.extend(
            {
                "pitch": int(pitch.midi),
                "name": pitch.nameWithOctave,
                "start": offset * seconds_per_quarter,
                "duration": duration * seconds_per_quarter,
            }
            for pitch in pitches
        )
    notes.sort(key=lambda item: (int(item["pitch"]), float(item["start"])))
    merged_notes: list[dict[str, float | int | str]] = []
    for item in notes:
        if merged_notes:
            previous = merged_notes[-1]
            previous_end = float(previous["start"]) + float(previous["duration"])
            if int(previous["pitch"]) == int(item["pitch"]) and float(item["start"]) <= previous_end:
                previous["duration"] = max(previous_end, float(item["start"]) + float(item["duration"])) - float(previous["start"])
                continue
        merged_notes.append(item)
    merged_notes.sort(key=lambda item: (float(item["start"]), int(item["pitch"])))
    total_duration = max((float(item["start"]) + float(item["duration"]) for item in merged_notes), default=0.0)
    return {"notes": merged_notes, "duration": total_duration, "bpm": bpm}


def render_piano_visualizer_player(
    midi_path: Path, fallback_bpm: float, audio_preview: bytes | None, audio_status: str
) -> None:
    """Render a compact player that expands into a client-side visualizer modal."""
    try:
        timeline = extract_piano_timeline(str(midi_path), fallback_bpm)
    except Exception:
        st.warning("Piano Visualizer is unavailable because this MIDI file could not be parsed.")
        return

    notes = timeline["notes"]
    if not isinstance(notes, list) or not notes:
        st.caption("No playable notes were found in this MIDI file.")
        return
    pitches = [int(item["pitch"]) for item in notes]
    lowest_key = max(0, min(pitches) - 3)
    highest_key = min(127, max(pitches) + 3)
    while highest_key - lowest_key < 12 and (lowest_key > 0 or highest_key < 127):
        lowest_key = max(0, lowest_key - 1)
        highest_key = min(127, highest_key + 1)

    audio_source = (
        json.dumps(f"data:audio/wav;base64,{base64.b64encode(audio_preview).decode('ascii')}")
        if audio_preview
        else "null"
    )
    component_html = """
    <style>
      html, body { height: 100%; margin: 0; background: transparent; font-family: Inter, system-ui, sans-serif; }
      .mini-player { position: relative; height: 54px; box-sizing: border-box; display: flex; align-items: center; gap: 10px; padding: 8px 10px; border: 1px solid rgba(45,212,191,.28); border-radius: 16px; background: #0B1115; color: #F8FAFC; }
      .mini-player.floating { position: fixed; right: 7px; bottom: 7px; z-index: 5; width: 72px; height: 72px; padding: 5px; display: grid; place-items: center; border-radius: 50%; border-color: rgba(45,212,191,.6); background: conic-gradient(#2DD4BF var(--mini-progress, 0%), #1B2A30 0); box-shadow: none; animation: mini-pop-in .42s cubic-bezier(.2,.9,.25,1.2) both; }
      .mini-player.floating::before { content: ''; position: absolute; inset: 5px; border-radius: 50%; background: #0B1115; }
      .mini-player.floating .mini-play { position: relative; z-index: 1; width: 50px; height: 50px; display: grid; place-items: center; padding: 0; font-size: 1.5rem; line-height: 1; font-weight: 900; text-indent: .02em; }
      .mini-player.floating .mini-copy, .mini-player.floating .mini-time { display: none; }
      .mini-player.floating.is-playing .mini-play { animation: mini-pulse 1.8s ease-in-out infinite; }
      @keyframes mini-pop-in { 0% { opacity: 0; transform: translateY(14px) scale(.55) rotate(-8deg); filter: blur(2px); } 65% { opacity: 1; transform: translateY(-2px) scale(1.06) rotate(2deg); filter: blur(0); } 100% { opacity: 1; transform: translateY(0) scale(1) rotate(0); } }
      @keyframes mini-pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(45,212,191,.12); } 50% { box-shadow: 0 0 0 8px rgba(45,212,191,0); } }
      .mini-play { width: 34px; height: 34px; padding: 0; border-radius: 50%; background: linear-gradient(145deg, #14B8A6, #38BDF8); color: #071014; }
      .mini-copy { flex: 1; min-width: 0; } .mini-label { color: #99F6E4; font-size: 10px; font-weight: 800; letter-spacing: .08em; } .mini-progress { height: 4px; margin-top: 6px; overflow: hidden; border-radius: 99px; background: #26333B; } .mini-progress > div { width: 0%; height: 100%; background: linear-gradient(90deg, #14B8A6, #38BDF8); }
      .mini-time { min-width: 74px; color: #CBD5E1; font-size: 12px; font-variant-numeric: tabular-nums; text-align: right; }
      .modal { display: none; position: fixed; inset: 0; z-index: 2; align-items: center; justify-content: center; padding: 20px; box-sizing: border-box; background: rgba(3,7,10,.86); backdrop-filter: blur(10px); }
      .modal.open { display: flex; } .modal-card { width: min(1060px, 96vw); padding: 14px; border: 1px solid rgba(45,212,191,.3); border-radius: 18px; background: #0D1418; box-shadow: 0 28px 80px rgba(0,0,0,.55); }
      .modal-title { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 9px; color: #99F6E4; font-size: 12px; font-weight: 800; letter-spacing: .1em; }
      .visualizer { height: 500px; box-sizing: border-box; padding: 12px; border: 1px solid rgba(148,163,184,.14); border-radius: 16px; background: linear-gradient(145deg, #111B20, #0D1418); color: #F8FAFC; overflow: hidden; }
      .topline { display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 12px; color: #94A3B8; }
      .status { color: #99F6E4; font-weight: 700; }
      .lane { position: relative; height: 330px; margin-top: 10px; overflow: hidden; border: 1px solid rgba(45,212,191,.14); border-radius: 12px 12px 0 0; background: radial-gradient(circle at 50% 0%, rgba(56,189,248,.12), transparent 44%), #070B0F; }
      .lane::after { content: ''; position: absolute; left: 0; right: 0; bottom: 0; height: 2px; background: #2DD4BF; box-shadow: 0 0 14px rgba(45,212,191,.7); }
      .fall-note { position: absolute; min-width: 5px; border-radius: 5px 5px 2px 2px; background: linear-gradient(180deg, #7DD3FC, #14B8A6); box-shadow: 0 0 10px rgba(45,212,191,.34); opacity: .68; will-change: transform, opacity; }
      .fall-note.active { opacity: 1; background: linear-gradient(180deg, #E0F2FE, #2DD4BF); box-shadow: 0 0 18px rgba(45,212,191,.9); }
      .keyboard { position: relative; height: 72px; display: flex; overflow: hidden; border: 1px solid rgba(148,163,184,.2); border-top: 0; border-radius: 0 0 12px 12px; background: #0A1115; }
      .key { position: absolute; bottom: 0; box-sizing: border-box; user-select: none; transition: background .06s ease, box-shadow .06s ease; }
      .white-key { height: 72px; background: #E2E8F0; border: 1px solid #94A3B8; border-radius: 0 0 4px 4px; }
      .black-key { z-index: 3; height: 44px; background: #0B1115; border: 1px solid #334155; border-radius: 0 0 4px 4px; }
      .key.active.white-key { background: #5EEAD4; box-shadow: inset 0 -4px 0 #14B8A6, 0 0 12px rgba(45,212,191,.78); }
      .key.active.black-key { background: #38BDF8; box-shadow: 0 0 13px rgba(56,189,248,.82); }
      .key-label { position: absolute; bottom: 4px; left: 50%; transform: translateX(-50%); color: #334155; font-size: 9px; font-weight: 800; }
      .controls { display: flex; align-items: center; gap: 7px; margin-top: 10px; }
      .close-action { border-color: rgba(248,250,252,.18); color: #CBD5E1; }
      button { border: 1px solid rgba(45,212,191,.3); border-radius: 8px; padding: 6px 9px; background: #111B20; color: #F8FAFC; font-weight: 700; cursor: pointer; }
      button:hover:not(:disabled) { background: #143B3D; } button:disabled { cursor: not-allowed; opacity: .45; }
      .progress { flex: 1; height: 5px; overflow: hidden; border-radius: 99px; background: #26333B; }
      .progress-fill { width: 0%; height: 100%; background: linear-gradient(90deg, #14B8A6, #38BDF8); }
      .time { min-width: 72px; text-align: right; color: #CBD5E1; font-variant-numeric: tabular-nums; font-size: 12px; }
    </style>
    <div class="mini-player">
      <button id="mini-toggle" class="mini-play" aria-label="Play composition">▶</button>
      <div class="mini-copy"><div class="mini-label">PLAY WITH PIANO VISUALIZER</div><div class="mini-progress"><div id="mini-progress"></div></div></div>
      <span class="mini-time" id="mini-time">0:00 / 0:00</span>
    </div>
    <div class="modal" id="modal"><div class="modal-card">
      <div class="modal-title"><span>PIANO VISUALIZER</span><button id="exit" class="close-action">Exit</button></div>
    <div class="visualizer">
      <div class="topline"><span class="status" id="status"></span><span id="tempo"></span></div>
      <div class="lane" id="lane"></div>
      <div class="keyboard" id="keyboard"></div>
      <div class="controls"><button id="play-toggle">▶ Play</button><button id="restart">↺ Restart</button><button id="minimize">Minimize</button><div class="progress"><div class="progress-fill" id="progress"></div></div><span class="time" id="time"></span></div>
      <audio id="audio" preload="metadata"></audio>
    </div></div></div>
    <script>
      const timeline = TIMELINE_DATA;
      let audioSource = AUDIO_SOURCE;
      const notes = timeline.notes;
      const duration = Math.max(Number(timeline.duration) || 0, .1);
      const low = KEY_LOW, high = KEY_HIGH;
      const blackClasses = new Set([1, 3, 6, 8, 10]);
      const lane = document.getElementById('lane'), keyboard = document.getElementById('keyboard');
      const audio = document.getElementById('audio'), status = document.getElementById('status');
      const playButton = document.getElementById('play-toggle');
      const restartButton = document.getElementById('restart'), progress = document.getElementById('progress');
      const timeLabel = document.getElementById('time');
      const miniPlayer = document.querySelector('.mini-player'), miniToggle = document.getElementById('mini-toggle'), miniProgress = document.getElementById('mini-progress'), miniTime = document.getElementById('mini-time');
      const modal = document.getElementById('modal'), minimizeButton = document.getElementById('minimize'), exitButton = document.getElementById('exit');
      let savedFrameStyle;
      document.getElementById('tempo').textContent = `${Math.round(timeline.bpm)} BPM`;
      if (audioSource) audio.src = audioSource;
      status.textContent = audioSource ? AUDIO_LABEL : 'Audio preview unavailable — visualization only';
      const whitePitches = []; for (let p = low; p <= high; p++) if (!blackClasses.has(p % 12)) whitePitches.push(p);
      const whiteWidth = 100 / whitePitches.length, centers = {}, keyNodes = {}, keyWidths = {};
      let whiteIndex = 0;
      for (let pitch = low; pitch <= high; pitch++) {
        const key = document.createElement('div'); const isBlack = blackClasses.has(pitch % 12);
        key.className = `key ${isBlack ? 'black-key' : 'white-key'}`; key.dataset.pitch = pitch;
        if (isBlack) { key.style.width = `${whiteWidth * .62}%`; key.style.left = `${whiteIndex * whiteWidth - whiteWidth * .31}%`; centers[pitch] = whiteIndex * whiteWidth; keyWidths[pitch] = whiteWidth * .62; }
        else { key.style.width = `${whiteWidth}%`; key.style.left = `${whiteIndex * whiteWidth}%`; centers[pitch] = (whiteIndex + .5) * whiteWidth; keyWidths[pitch] = whiteWidth; if (pitch % 12 === 0) { const label = document.createElement('span'); label.className = 'key-label'; label.textContent = `C${Math.floor(pitch / 12) - 1}`; key.appendChild(label); } whiteIndex++; }
        keyboard.appendChild(key); keyNodes[pitch] = key;
      }
      // Give genuinely overlapping notes on the same pitch separate narrow
      // lanes so a sustained bar never hides a shorter retrigger.
      const noteLayout = notes.map(() => ({ lane: 0, count: 1 }));
      const pitchGroups = {};
      notes.forEach((item, index) => { (pitchGroups[item.pitch] ||= []).push(index); });
      Object.values(pitchGroups).forEach(indices => {
        indices.sort((a, b) => Number(notes[a].start) - Number(notes[b].start));
        const active = [], placements = [];
        indices.forEach(index => {
          const start = Number(notes[index].start), end = start + Number(notes[index].duration);
          for (let i = active.length - 1; i >= 0; i--) if (active[i].end <= start) active.splice(i, 1);
          let laneIndex = 0; while (active.some(item => item.lane === laneIndex)) laneIndex++;
          active.push({ lane: laneIndex, end }); placements.push({ index, lane: laneIndex });
        });
        const laneCount = Math.max(1, ...placements.map(item => item.lane + 1));
        placements.forEach(item => { noteLayout[item.index] = { lane: item.lane, count: laneCount }; });
      });
      const bars = notes.map((note, index) => { const bar = document.createElement('div'); bar.className = 'fall-note'; lane.appendChild(bar); return { note, bar, layout: noteLayout[index] }; });
      let frameId = null, visualTime = 0, visualStartedAt = 0, visualPlaying = false;
      const formatTime = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
      const currentTime = now => audioSource ? audio.currentTime : (visualPlaying ? Math.min(duration, visualTime + (now - visualStartedAt) / 1000) : visualTime);
      const render = now => {
        const current = currentTime(now); const lookAhead = 3.4; const laneHeight = lane.clientHeight - 4; const scale = laneHeight / lookAhead;
        const active = new Set();
        bars.forEach(({ note, bar, layout }) => {
          const until = Number(note.start) - current, noteDuration = Number(note.duration);
          if (until > lookAhead || until + noteDuration < 0 || centers[note.pitch] === undefined) { bar.style.display = 'none'; return; }
          const height = Math.max(8, noteDuration * scale); const y = Math.min(laneHeight - height, (lookAhead - until) * scale - height);
          const keyWidth = keyWidths[note.pitch], laneWidth = layout.count > 1 ? keyWidth / layout.count * .9 : keyWidth; const laneCenter = centers[note.pitch] + (layout.lane - (layout.count - 1) / 2) * keyWidth / layout.count;
          bar.style.display = 'block'; bar.style.left = `${laneCenter}%`; bar.style.width = `${laneWidth}%`; bar.style.height = `${height}px`; bar.style.transform = `translate(-50%, ${y}px)`;
          const isActive = current >= Number(note.start) && current < Number(note.start) + noteDuration;
          bar.classList.toggle('active', isActive); if (isActive) active.add(note.pitch);
        });
        Object.entries(keyNodes).forEach(([pitch, key]) => key.classList.toggle('active', active.has(Number(pitch))));
        const progressValue = `${Math.min(100, current / duration * 100)}%`; progress.style.width = progressValue; miniProgress.style.width = progressValue; miniPlayer.style.setProperty('--mini-progress', progressValue); timeLabel.textContent = `${formatTime(current)} / ${formatTime(duration)}`; miniTime.textContent = timeLabel.textContent;
        if ((audioSource && !audio.paused && !audio.ended) || (!audioSource && visualPlaying && current < duration)) frameId = requestAnimationFrame(render);
        else if (!audioSource) { visualTime = current; visualPlaying = false; }
      };
      const updateAmbientSequence = playing => {
        try {
          const sequence = parent.document.getElementById('neuraltune-ambient-sequence');
          if (sequence) sequence.classList.toggle('is-playing', playing);
        } catch (_) {}
      };
      const updatePlayButton = playing => {
        playButton.textContent = playing ? '⏸ Pause' : '▶ Play';
        miniToggle.textContent = playing ? '⏸' : '▶';
        miniToggle.setAttribute('aria-label', playing ? 'Pause composition' : 'Play composition');
        miniPlayer.classList.toggle('is-playing', playing);
        updateAmbientSequence(playing);
      };
      const setFrameExpanded = expanded => { try { const frame = window.frameElement; if (!frame) return; if (expanded) { if (savedFrameStyle === undefined) savedFrameStyle = frame.getAttribute('style'); frame.style.position = 'fixed'; frame.style.inset = '0'; frame.style.width = '100vw'; frame.style.height = '100vh'; frame.style.zIndex = '999999'; frame.style.border = '0'; } else if (savedFrameStyle === null) frame.removeAttribute('style'); else if (savedFrameStyle !== undefined) frame.setAttribute('style', savedFrameStyle); } catch (_) {} };
      const setFrameMinimized = () => { try { const frame = window.frameElement; if (!frame) return; if (savedFrameStyle === undefined) savedFrameStyle = frame.getAttribute('style'); frame.style.position = 'fixed'; frame.style.left = 'auto'; frame.style.top = 'auto'; frame.style.right = '18px'; frame.style.bottom = '18px'; frame.style.width = '86px'; frame.style.height = '86px'; frame.style.zIndex = '999999'; frame.style.border = '0'; frame.style.background = 'transparent'; frame.style.boxShadow = 'none'; } catch (_) {} };
      const openModal = () => { modal.classList.add('open'); miniPlayer.classList.remove('floating'); setFrameExpanded(true); };
      const minimizeModal = () => { modal.classList.remove('open'); miniPlayer.classList.add('floating'); setFrameMinimized(); };
      const start = () => {
        if (audioSource) audio.play().catch(() => { status.textContent = 'Audio playback was blocked — visualization only'; audioSource = null; visualStartedAt = performance.now(); visualPlaying = true; updatePlayButton(true); frameId = requestAnimationFrame(render); });
        else { visualStartedAt = performance.now(); visualPlaying = true; updatePlayButton(true); frameId = requestAnimationFrame(render); }
      };
      const pause = () => { if (audioSource) audio.pause(); else { visualTime = currentTime(performance.now()); visualPlaying = false; updatePlayButton(false); } cancelAnimationFrame(frameId); render(performance.now()); };
      const restart = () => { if (audioSource) { audio.currentTime = 0; } visualTime = 0; visualStartedAt = performance.now(); if (!audioSource || !audio.paused) { visualPlaying = true; cancelAnimationFrame(frameId); frameId = requestAnimationFrame(render); } else render(performance.now()); };
      playButton.onclick = () => { if ((audioSource && !audio.paused) || (!audioSource && visualPlaying)) pause(); else start(); };
      miniToggle.onclick = () => {
         const playing = (audioSource && !audio.paused) || (!audioSource && visualPlaying);
         if (!modal.classList.contains('open')) { openModal(); if (!playing) start(); else { cancelAnimationFrame(frameId); frameId = requestAnimationFrame(render); } return; }
         if (playing) pause(); else { openModal(); start(); }
       };
      restartButton.onclick = restart;
      minimizeButton.onclick = minimizeModal;
      exitButton.onclick = () => { pause(); if (audioSource) audio.currentTime = 0; visualTime = 0; render(performance.now()); miniPlayer.classList.remove('floating'); modal.classList.remove('open'); setFrameExpanded(false); };
      if (audioSource) { audio.onplay = () => { updatePlayButton(true); cancelAnimationFrame(frameId); frameId = requestAnimationFrame(render); }; audio.onpause = () => { updatePlayButton(false); cancelAnimationFrame(frameId); render(performance.now()); }; audio.onended = () => { updatePlayButton(false); cancelAnimationFrame(frameId); render(performance.now()); }; }
      else updatePlayButton(false);
      render(performance.now());
    </script>
    """
    component_html = (
        component_html.replace("TIMELINE_DATA", json.dumps(timeline))
        .replace("AUDIO_SOURCE", audio_source)
        .replace("KEY_LOW", str(lowest_key))
        .replace("KEY_HIGH", str(highest_key))
        .replace("AUDIO_LABEL", json.dumps(audio_status))
    )
    components.html(component_html, height=70, scrolling=False)


@st.cache_data(show_spinner=False)
def load_training_insights() -> dict[str, object]:
    """Read existing processing and training metadata without loading or retraining a model."""
    insights: dict[str, object] = {
        "dataset": "Unavailable",
        "midi_files": None,
        "musical_events": None,
        "training_sequences": None,
        "vocabulary_size": None,
        "class_counts": None,
        "sequence_length": None,
        "parameter_count": None,
        "model": {},
        "history": None,
    }
    preprocessed_path = V4_METADATA_PATH
    try:
        preprocessed = json.loads(preprocessed_path.read_text(encoding="utf-8"))
        insights["dataset"] = f"MAESTRO {preprocessed.get('genre', GENRE)} piano MIDI"
        insights["midi_files"] = preprocessed.get("source_midi_files")
        statistics = preprocessed.get("statistics", {})
        insights["musical_events"] = preprocessed.get("total_raw_notes")
        insights["class_counts"] = {
            "pitch": preprocessed.get("pitch_classes"),
            "delta": len(preprocessed.get("mappings", {}).get("delta_steps", {}).get("index_to_value", [])),
            "duration": len(preprocessed.get("mappings", {}).get("duration_steps", {}).get("index_to_value", [])),
        }
        insights["sequence_length"] = preprocessed.get("sequence_length")
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    try:
        training_metadata = json.loads(TRAINING_METADATA_PATH.read_text(encoding="utf-8"))
        training_examples = training_metadata.get("train_sequence_count")
        validation_examples = training_metadata.get("validation_sequence_count")
        if isinstance(training_examples, int) and isinstance(validation_examples, int):
            insights["training_sequences"] = training_examples + validation_examples
        class_counts = training_metadata.get("class_counts", {})
        if isinstance(class_counts, dict):
            insights["class_counts"] = {
                "pitch": class_counts.get("pitch_decoder_including_eos"),
                "delta": class_counts.get("delta"),
                "duration": class_counts.get("duration"),
            }
        insights["sequence_length"] = training_metadata.get("sequence_length", insights["sequence_length"])
        insights["parameter_count"] = training_metadata.get("parameter_count")
        architecture = training_metadata.get("architecture", {})
        insights["model"] = {
            "pitch_embedding_dimension": 32,
            "duration_embedding_dimension": 16,
            "delta_embedding_dimension": 16,
            "lstm_units": 256,
            "dropout_rate": 0.3,
            "description": architecture.get("conditional_pitch_decoder", "Conditional GRU(256) with START/EOS"),
        }
        history = training_metadata.get("history")
        if isinstance(history, dict) and history.get("loss") and history.get("val_loss"):
            insights["history"] = history
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return insights


def insight_value(value: object) -> str:
    """Format a metadata value without inventing unavailable values."""
    if isinstance(value, int):
        return f"{value:,}"
    return str(value) if value is not None else "Unavailable"


def render_training_insights() -> None:
    """Display real project metadata and any saved per-epoch loss history."""
    insights = load_training_insights()
    first_row = st.columns(3)
    first_row[0].metric("MIDI files", insight_value(insights["midi_files"]))
    first_row[1].metric("Musical events", insight_value(insights["musical_events"]))
    first_row[2].metric("Training sequences", insight_value(insights["training_sequences"]))
    second_row = st.columns(3)
    class_counts = insights.get("class_counts")
    if isinstance(class_counts, dict):
        class_summary = (
            f"P {insight_value(class_counts.get('pitch'))} · "
            f"Δ {insight_value(class_counts.get('delta'))} · "
            f"D {insight_value(class_counts.get('duration'))}"
        )
    else:
        class_summary = "Unavailable"
    second_row[0].metric("Output classes", class_summary)
    second_row[1].metric("Sequence length", insight_value(insights["sequence_length"]))
    second_row[2].metric("Parameters", insight_value(insights["parameter_count"]))
    st.caption(f"Dataset: {insights['dataset']}")

    model = insights["model"] if isinstance(insights["model"], dict) else {}
    pitch_embedding = insight_value(model.get("pitch_embedding_dimension"))
    duration_embedding = insight_value(model.get("duration_embedding_dimension"))
    delta_embedding = insight_value(model.get("delta_embedding_dimension"))
    lstm_units = insight_value(model.get("lstm_units"))
    dropout = insight_value(model.get("dropout_rate"))
    sequence_length = insight_value(insights["sequence_length"])
    description = model.get("description", "Conditional GRU(256) with START/EOS")
    st.markdown(
        f"**Model:** Pitch Embedding({pitch_embedding}) + Duration Embedding({duration_embedding}) + Delta Embedding({delta_embedding}) → LSTM({lstm_units}) → Dropout({dropout}) → {description}"
    )
    st.markdown(
        f"**Pipeline:** {sequence_length}-onset context → grouped onset decoder → START/EOS pitch slots → aligned durations + delta timing"
    )

    history = insights.get("history")
    if not isinstance(history, dict):
        st.caption("Detailed epoch history was not stored for this trained model.")
        return
    losses = history.get("loss", [])
    validation_losses = history.get("val_loss", [])
    if not isinstance(losses, list) or not isinstance(validation_losses, list):
        st.caption("Detailed epoch history was not stored for this trained model.")
        return
    figure, axis = plt.subplots(figsize=(8, 2.15))
    figure.patch.set_facecolor("#0D1418")
    axis.set_facecolor("#0D1418")
    epochs = range(1, min(len(losses), len(validation_losses)) + 1)
    axis.plot(epochs, losses[:len(epochs)], color="#14B8A6", label="Training loss")
    axis.plot(epochs, validation_losses[:len(epochs)], color="#38BDF8", label="Validation loss")
    axis.set_title("Training vs validation loss", color="#F8FAFC", loc="left", fontsize=10, pad=8)
    axis.tick_params(colors="#CBD5E1", labelsize=8)
    axis.legend(facecolor="#111B20", edgecolor="#475569", labelcolor="#F8FAFC", fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#475569")
    axis.grid(axis="y", color="#334155", alpha=.35)
    render_expandable_figure(figure, "Training vs validation loss")


def render_analysis() -> None:
    """Render the compact evaluation dashboard below the studio workspace."""
    result = st.session_state.latest_result
    metrics = result.get("metrics") if result else None
    with st.container(border=True):
        st.markdown("<div class='analysis-title'>COMPOSITION ANALYSIS</div>", unsafe_allow_html=True)
        overview_tab, pitch_tab, rhythm_tab, insights_tab, model_tab = st.tabs(
            ["Overview", "Pitch Analysis", "Rhythm", "Training Insights", "About the Model"]
        )
        with overview_tab:
            if metrics:
                first, second, third, fourth = st.columns(4)
                first.metric("Notes", metrics["total_notes"])
                second.metric("Chords", metrics["chord_count"])
                third.metric("Duration", f"{metrics['total_duration']:.1f} ql")
                fourth.metric("Range", f"{metrics['lowest_pitch']}–{metrics['highest_pitch']}")
            else:
                st.caption("Generate a composition to inspect note, chord, duration, and pitch-range metrics.")
        with pitch_tab:
            if metrics:
                distribution = metrics["pitch_class_distribution"]
                render_distribution_chart(
                    PITCH_CLASS_NAMES,
                    [distribution.get(pitch_class, 0) for pitch_class in PITCH_CLASS_NAMES],
                    "Pitch-class distribution",
                    "#14B8A6",
                )
            else:
                st.caption("Pitch analysis appears after generation.")
        with rhythm_tab:
            if metrics:
                distribution = metrics["duration_distribution"]
                durations = sorted(distribution, key=float)
                render_distribution_chart(
                    durations,
                    [distribution[duration] for duration in durations],
                    "Duration distribution (quarter lengths)",
                    "#38BDF8",
                )
            else:
                st.caption("Rhythm analysis appears after generation.")
        with insights_tab:
            render_training_insights()
        with model_tab:
            st.markdown(
                "**MAESTRO MIDI** → **50-onset context** → **LSTM historical encoder** → **conditional GRU pitch decoder** → **START/EOS onset construction** → **pitch-conditioned durations + learned delta timing** → **MIDI**"
            )
            st.caption("Temperature adjusts randomness and diversity; it does not retrain or alter the model.")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="NeuraTune · AI Music Studio", page_icon="♪", layout="wide")
    apply_theme()
    initialize_session_state()
    render_header()

    settings_column, player_column = st.columns([0.4, 0.6], gap="medium")
    with settings_column:
        settings, generate_clicked = render_controls()
    with player_column:
        render_player()

    if generate_clicked:
        overlay = show_loading_overlay()
        try:
            result = generate_composition(settings)
        except FileNotFoundError:
            overlay.empty()
            st.error("The trained model or processed artifacts are missing. Complete the backend phases first.")
        except ValueError:
            overlay.empty()
            st.error("The frozen V4 production model and metadata are incompatible.")
        except Exception:
            overlay.empty()
            st.error("Generation could not be completed. Check that the trained model and artifacts are valid.")
        else:
            overlay.empty()
            set_selected_composition(result)
            st.session_state.pending_library_selection = str(result["id"])
            history = [result] + [
                item
                for item in st.session_state.generation_history
                if str(item.get("midi_path", "")) != str(result["midi_path"])
            ]
            st.session_state.generation_history = history[:10]
            save_persistent_history(st.session_state.generation_history)
            st.rerun()

    render_analysis()


if __name__ == "__main__":
    main()
