"""NeuraTune: a compact Streamlit studio for AI MIDI generation."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.evaluate import PITCH_CLASS_NAMES, analyze_midi
from src.generate import (
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    export_midi,
    generate_event_ids,
    load_generation_artifacts,
    select_seed_sequence,
)


GENRE = "classical"


def apply_theme() -> None:
    """Apply a compact dark music-studio design system."""
    st.markdown(
        """
        <style>
        :root { --bg: #070A12; --surface: #0F1424; --surface-2: #131A2E; --line: rgba(255,255,255,.10); --text: #F8FAFC; --muted: #94A3B8; --violet: #8B5CF6; --blue: #38BDF8; --green: #34D399; }
        #MainMenu, footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent; }
        .stApp { background: radial-gradient(circle at 82% -10%, rgba(96, 74, 190, .20), transparent 32%), #070A12; color: var(--text); }
        .block-container { max-width: 1460px; min-height: 100vh; padding: 1rem 1.25rem 1.1rem; }
        h1, h2, h3, p, label, .stMarkdown { color: var(--text); }
        h2, h3 { letter-spacing: -.02em; }
        div[data-testid="stWidgetLabel"] p, div[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: linear-gradient(145deg, rgba(19,26,46,.94), rgba(12,17,32,.94)); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 14px 40px rgba(0,0,0,.18); }
        div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: .55rem .7rem; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: #0C1222 !important; border-color: rgba(255,255,255,.14) !important; color: var(--text) !important; }
        div[data-baseweb="select"] span, div[data-baseweb="input"] input { color: var(--text) !important; }
        div[data-baseweb="select"] svg { fill: var(--muted); }
        [data-testid="stSlider"] [role="slider"] { background: var(--violet) !important; border-color: #C4B5FD !important; }
        [data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--violet); }
        .stButton > button, [data-testid="stDownloadButton"] > button { border: 0; border-radius: 12px; color: #fff; font-weight: 700; background: linear-gradient(95deg, #6D4AFF, #4F8DFF); box-shadow: 0 8px 22px rgba(95, 78, 255, .25); }
        .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover { border: 0; color: #fff; filter: brightness(1.08); }
        button[kind="primary"] { background: linear-gradient(95deg, #6D4AFF, #4F8DFF) !important; }
        button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; padding: .45rem .75rem; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #E9E5FF; border-bottom-color: var(--violet); }
        .studio-header { min-height: 78px; display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .65rem 0 1rem; border-bottom: 1px solid var(--line); margin-bottom: 1rem; }
        .brand { display: flex; align-items: center; gap: .7rem; }
        .note-mark { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 12px; background: linear-gradient(145deg, #8B5CF6, #3B82F6); font-size: 1.35rem; box-shadow: 0 8px 20px rgba(91, 79, 235, .30); }
        .brand-name { color: var(--text); font-size: 1.18rem; font-weight: 800; letter-spacing: -.03em; }
        .brand-subtitle { color: var(--muted); font-size: .78rem; margin-top: .1rem; }
        .header-copy { color: var(--muted); font-size: .76rem; margin: .25rem 0 0 3.05rem; }
        .header-right { text-align: right; }
        .ready { color: #A7F3D0; font-size: .78rem; font-weight: 700; }
        .ready-dot { color: var(--green); }
        .badge { display: inline-block; margin: .32rem 0 0 .25rem; padding: .22rem .5rem; border: 1px solid rgba(139,92,246,.38); border-radius: 999px; background: rgba(139,92,246,.12); color: #DDD6FE; font-size: .67rem; font-weight: 700; }
        .card-title { color: #C4B5FD; font-size: .69rem; font-weight: 800; letter-spacing: .12em; margin: .1rem 0 .5rem; }
        .temperature-state { color: #C4B5FD; font-weight: 700; font-size: .8rem; }
        .control-help { color: var(--muted); font-size: .73rem; margin-top: -.4rem; }
        .player-top { display: flex; align-items: center; gap: .85rem; min-height: 150px; }
        .album-art { flex: 0 0 126px; width: 126px; height: 126px; position: relative; overflow: hidden; border-radius: 16px; border: 1px solid rgba(196,181,253,.35); background: radial-gradient(circle at 28% 25%, #C4B5FD 0, #6D4AFF 24%, #111A42 56%, #090D1B 100%); box-shadow: inset 0 0 30px rgba(255,255,255,.12), 0 12px 26px rgba(0,0,0,.28); }
        .album-art::before, .album-art::after { content: ''; position: absolute; border-radius: 50%; border: 1px solid rgba(255,255,255,.35); }
        .album-art::before { width: 115px; height: 115px; left: 5px; top: 5px; }
        .album-art::after { width: 58px; height: 58px; left: 33px; top: 33px; box-shadow: 0 0 26px #38BDF8; }
        .composition-kicker { color: #A5B4FC; font-size: .68rem; font-weight: 800; letter-spacing: .12em; }
        .composition-title { color: var(--text); font-size: 1.18rem; font-weight: 800; margin: .25rem 0; }
        .composition-meta { color: var(--muted); font-size: .8rem; }
        .midi-badge { display: inline-block; margin-top: .55rem; color: #BAE6FD; border: 1px solid rgba(56,189,248,.32); border-radius: 999px; padding: .18rem .46rem; font-size: .67rem; }
        .wave-caption { color: var(--muted); font-size: .67rem; margin: .55rem 0 .2rem; }
        .waveform { height: 28px; display: flex; align-items: center; gap: 3px; overflow: hidden; }
        .waveform span { width: 5px; border-radius: 6px; background: linear-gradient(#A78BFA, #38BDF8); opacity: .9; }
        .metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: .4rem; margin: .6rem 0; }
        .metric-block { padding: .48rem; border-radius: 10px; border: 1px solid rgba(255,255,255,.08); background: rgba(5,9,20,.42); }
        .metric-value { color: var(--text); font-size: 1rem; font-weight: 800; overflow-wrap: anywhere; }
        .metric-label { color: var(--muted); font-size: .63rem; margin-top: .08rem; }
        .secondary-meta { color: var(--muted); font-size: .75rem; display: flex; flex-wrap: wrap; gap: .35rem 1rem; margin-bottom: .55rem; }
        .empty-player { min-height: 230px; display: grid; place-items: center; text-align: center; color: var(--muted); }
        .empty-icon { color: #A78BFA; font-size: 2rem; }
        .analysis-title { color: #E2E8F0; font-size: .92rem; font-weight: 800; margin: .1rem 0 .25rem; }
        @media (max-width: 800px) { .block-container { min-height: auto; padding: .75rem .8rem 1.5rem; } .studio-header { align-items: flex-start; } .header-copy { display: none; } .header-right { max-width: 48%; } .badge { font-size: .61rem; } .metric-row { grid-template-columns: repeat(2, 1fr); } .player-top { align-items: flex-start; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_resources() -> tuple[object, np.ndarray, list[str], int]:
    """Cache the existing Keras model and processed event sequences."""
    return load_generation_artifacts(GENRE)


def initialize_session_state() -> None:
    """Preserve the latest generated composition across widget reruns."""
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("latest_midi_path", None)
    st.session_state.setdefault("latest_settings", None)
    st.session_state.setdefault("latest_metrics", None)
    st.session_state.setdefault("generation_history", [])


def model_ready() -> bool:
    """Check whether the backend files needed by the studio are present."""
    return all(
        path.exists()
        for path in (
            MODELS_DIR / f"{GENRE}_lstm.keras",
            PROCESSED_DATA_DIR / f"{GENRE}_preprocessed.json",
            PROCESSED_DATA_DIR / f"{GENRE}_sequences.npz",
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
        length = st.slider("Generation Length", 50, 500, 200, 10)
        temperature = st.slider("Creativity", 0.3, 1.5, 1.0, 0.1)
        st.markdown(
            f"<div class='temperature-state'>{temperature:.1f} · {creativity_state(temperature)}</div>",
            unsafe_allow_html=True,
        )
        st.caption("0.3–0.7 conservative · 0.8–1.2 balanced · 1.3–1.5 experimental")
        tempo = st.slider("Tempo", 60, 180, 100, 1, format="%d BPM")

        seed_column, duration_column = st.columns(2)
        with seed_column:
            seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
        with duration_column:
            duration_choice = st.selectbox("Min Duration", ["Original", "0.25", "0.5", "1.0"], index=2)
        st.markdown(
            "<div class='control-help'>Minimum duration affects MIDI reconstruction only, never the model prediction.</div>",
            unsafe_allow_html=True,
        )
        generate_clicked = st.button("✦ Generate Music", type="primary", use_container_width=True)

    settings: dict[str, int | float | None] = {
        "length": length,
        "temperature": temperature,
        "seed": int(seed),
        "tempo": tempo,
        "min_duration": None if duration_choice == "Original" else float(duration_choice),
    }
    return settings, generate_clicked


def generate_composition(settings: dict[str, int | float | None]) -> dict[str, object]:
    """Use the existing Phase 5 and 6 functions for one UI generation request."""
    model, input_sequences, id_to_event, sequence_length = load_model_resources()
    random_seed = int(settings["seed"])
    rng = np.random.default_rng(random_seed)
    seed_sequence = select_seed_sequence(input_sequences, rng)
    generated_ids = generate_event_ids(
        model, seed_sequence, int(settings["length"]), float(settings["temperature"]), rng
    )
    generated_tokens = [id_to_event[event_id] for event_id in generated_ids]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / (
        f"{GENRE}_studio_{settings['length']}_t{str(settings['temperature']).replace('.', 'p')}"
        f"_tempo{str(settings['tempo']).replace('.', 'p')}_seed{random_seed}_{timestamp}.mid"
    )
    skipped_events = export_midi(
        generated_tokens,
        output_path,
        min_duration=settings["min_duration"],
        tempo_bpm=float(settings["tempo"]),
    )
    result: dict[str, object] = {
        "midi_path": str(output_path),
        "settings": settings,
        "seed_sequence_length": sequence_length,
        "generated_events": len(generated_tokens),
        "skipped_events": skipped_events,
    }
    try:
        result["metrics"] = analyze_midi(output_path)
    except Exception as error:
        result["metrics"] = None
        result["evaluation_error"] = str(error)
    return result


def waveform_markup() -> str:
    """Return a decorative sequence visualization, not an audio waveform."""
    heights = [8, 15, 24, 11, 20, 28, 14, 25, 10, 18, 27, 13, 22, 9, 19, 26, 12, 17, 23, 8]
    bars = "".join(f"<span style='height:{height}px'></span>" for height in heights)
    return f"<div class='wave-caption'>Decorative sequence visualization</div><div class='waveform'>{bars}</div>"


def player_markup(result: dict[str, object] | None) -> str:
    """Create the compact music-player visual for empty and generated states."""
    if not result:
        return """
        <div class="empty-player"><div><div class="empty-icon">✦</div>
        <strong>Your composition will appear here</strong><br><span>Configure the studio controls and generate a new sequence.</span></div></div>
        """

    metrics = result.get("metrics") or {}
    settings = result["settings"]
    minimum_duration = settings["min_duration"] if settings["min_duration"] is not None else "Original"
    return f"""
    <div class="player-top"><div class="album-art"></div><div>
      <div class="composition-title">Neural Composition #{settings['seed']}</div>
      <div class="composition-meta">Classical Piano · AI Generated</div>
      <div class="midi-badge">MIDI · Generated by NeuraTune</div>
      {waveform_markup()}
    </div></div>
    <div class="metric-row">
      <div class="metric-block"><div class="metric-value">{result['generated_events']}</div><div class="metric-label">EVENTS</div></div>
      <div class="metric-block"><div class="metric-value">{settings['temperature']}</div><div class="metric-label">CREATIVITY</div></div>
      <div class="metric-block"><div class="metric-value">{settings['tempo']}</div><div class="metric-label">BPM</div></div>
      <div class="metric-block"><div class="metric-value">{metrics.get('unique_pitches', '—')}</div><div class="metric-label">PITCHES</div></div>
    </div>
    <div class="secondary-meta"><span>Pitch range: {metrics.get('lowest_pitch', '—')}–{metrics.get('highest_pitch', '—')}</span><span>Seed: {settings['seed']}</span><span>Min duration: {minimum_duration}</span></div>
    """


def render_player() -> None:
    """Render the result card, download action, and small session history."""
    result = st.session_state.latest_result
    with st.container(border=True):
        st.markdown("<div class='card-title'>YOUR AI COMPOSITION</div>", unsafe_allow_html=True)
        st.markdown(player_markup(result), unsafe_allow_html=True)
        if result:
            midi_path = Path(str(result["midi_path"]))
            if midi_path.exists():
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

    if st.session_state.generation_history:
        latest = st.session_state.generation_history[0]
        st.caption(f"Session: {len(st.session_state.generation_history)} compositions · Latest: {latest['name']}")


def render_distribution_chart(labels: list[str], values: list[int], title: str, color: str) -> None:
    """Render a compact dark chart using the existing Matplotlib dependency."""
    figure, axis = plt.subplots(figsize=(8, 2.15))
    figure.patch.set_facecolor("#0F1424")
    axis.set_facecolor("#0F1424")
    axis.bar(labels, values, color=color, edgecolor="#D8D4FE", linewidth=0.35)
    axis.set_title(title, color="#F8FAFC", loc="left", fontsize=10, pad=8)
    axis.tick_params(colors="#CBD5E1", labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#475569")
    axis.grid(axis="y", color="#334155", alpha=.35)
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_analysis() -> None:
    """Render the compact evaluation dashboard below the studio workspace."""
    result = st.session_state.latest_result
    metrics = result.get("metrics") if result else None
    with st.container(border=True):
        st.markdown("<div class='analysis-title'>COMPOSITION ANALYSIS</div>", unsafe_allow_html=True)
        overview_tab, pitch_tab, rhythm_tab, model_tab = st.tabs(
            ["Overview", "Pitch Analysis", "Rhythm", "About the Model"]
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
                    "#8B5CF6",
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
        with model_tab:
            st.markdown(
                "**MAESTRO MIDI** → **50-event sequence** → **Embedding** → **LSTM** → **Softmax** → **Temperature sampling** → **MIDI**"
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
        with st.spinner("Neural network is composing..."):
            try:
                result = generate_composition(settings)
            except FileNotFoundError:
                st.error("The trained model or processed artifacts are missing. Complete the backend phases first.")
            except Exception:
                st.error("Generation could not be completed. Check that the trained model and artifacts are valid.")
            else:
                st.session_state.latest_result = result
                st.session_state.latest_midi_path = result["midi_path"]
                st.session_state.latest_settings = settings
                st.session_state.latest_metrics = result.get("metrics")
                st.session_state.generation_history.insert(
                    0,
                    {
                        "name": Path(str(result["midi_path"])).name,
                        "events": result["generated_events"],
                        "temperature": settings["temperature"],
                        "tempo": settings["tempo"],
                    },
                )
                st.session_state.generation_history = st.session_state.generation_history[:5]
                st.rerun()

    render_analysis()


if __name__ == "__main__":
    main()
