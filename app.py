"""NeuraTune: a Streamlit interface for AI MIDI music generation."""

import json
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
    """Apply the compact dark visual language for the music studio."""
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at 15% 5%, #222052 0, #0e1024 28%, #080b17 65%); color: #eef0ff; }
        .block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #f7f7ff !important; letter-spacing: -0.03em; }
        .hero { padding: 1.4rem 0 1.8rem; }
        .eyebrow { color: #a8a5ff; font-size: 0.82rem; font-weight: 700; letter-spacing: .12em; }
        .hero-title { font-size: clamp(2.3rem, 6vw, 4.6rem); font-weight: 800; line-height: .95; margin: .35rem 0; background: linear-gradient(90deg, #f5f3ff, #a6b5ff 55%, #d8a6ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero-subtitle { color: #b7bdd8; font-size: 1.05rem; max-width: 700px; line-height: 1.6; }
        .badge { display: inline-block; margin: .6rem .35rem 0 0; padding: .3rem .65rem; border: 1px solid rgba(159, 143, 255, .45); border-radius: 999px; color: #dcd7ff; background: rgba(112, 88, 208, .13); font-size: .76rem; font-weight: 700; }
        .workflow { margin: 1.3rem 0 1.7rem; padding: .9rem 1rem; border: 1px solid rgba(122, 137, 255, .22); border-radius: 16px; background: rgba(15, 18, 40, .65); color: #c9cdea; text-align: center; line-height: 1.8; }
        .empty-state { padding: 3rem 1.5rem; text-align: center; border: 1px dashed rgba(151, 150, 255, .42); border-radius: 20px; background: rgba(17, 20, 42, .52); }
        .empty-icon { font-size: 2rem; margin-bottom: .5rem; }
        div[data-testid="stMetric"] { background: rgba(18, 22, 47, .72); border: 1px solid rgba(137, 142, 255, .22); border-radius: 14px; padding: .8rem; }
        div[data-testid="stMetricLabel"] { color: #aeb6d8; }
        div[data-testid="stMetricValue"] { color: #f2edff; }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(17, 20, 43, .6); border-color: rgba(135, 129, 255, .25); }
        .stButton > button { border-radius: 12px; min-height: 3rem; font-weight: 700; }
        [data-testid="stSidebar"] { background: #0b0e1d; border-right: 1px solid rgba(139, 130, 255, .16); }
        @media (max-width: 700px) { .block-container { padding: 1.2rem 1rem 2rem; } .hero { padding-top: .8rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_resources() -> tuple[object, np.ndarray, list[str], int]:
    """Cache the trained Keras model and processed sequences for the UI session."""
    return load_generation_artifacts(GENRE)


@st.cache_data(show_spinner=False)
def load_artifact_summary() -> tuple[int, int]:
    """Read lightweight sidebar details without loading the neural network."""
    artifact_path = PROCESSED_DATA_DIR / f"{GENRE}_preprocessed.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return int(artifact["metadata"]["sequence_length"]), len(artifact["id_to_event"])


def generate_composition(settings: dict[str, int | float | None]) -> dict[str, object]:
    """Reuse the Phase 5 and 6 backend functions for one UI generation request."""
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
    temperature_label = str(settings["temperature"]).replace(".", "p")
    tempo_label = str(settings["tempo"]).replace(".", "p")
    min_duration = settings["min_duration"]
    duration_label = (
        f"_mindur{str(min_duration).replace('.', 'p')}" if min_duration is not None else ""
    )
    output_path = OUTPUTS_DIR / (
        f"{GENRE}_studio_{settings['length']}_t{temperature_label}_tempo{tempo_label}"
        f"{duration_label}_seed{random_seed}_{timestamp}.mid"
    )
    skipped_events = export_midi(
        generated_tokens,
        output_path,
        min_duration=min_duration,
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


def initialize_session_state() -> None:
    """Keep the latest composition visible during Streamlit widget reruns."""
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("latest_midi_path", None)
    st.session_state.setdefault("latest_settings", None)
    st.session_state.setdefault("latest_metrics", None)
    st.session_state.setdefault("generation_history", [])


def render_sidebar() -> None:
    """Show compact project and model context."""
    model_ready = (MODELS_DIR / f"{GENRE}_lstm.keras").exists()
    artifacts_ready = all(
        path.exists()
        for path in (
            PROCESSED_DATA_DIR / f"{GENRE}_preprocessed.json",
            PROCESSED_DATA_DIR / f"{GENRE}_sequences.npz",
        )
    )

    with st.sidebar:
        st.markdown("## NeuraTune")
        st.caption("AI Music Generation Studio")
        if model_ready and artifacts_ready:
            st.success("Trained model ready")
        else:
            st.error("Model or processed artifacts missing")

        try:
            sequence_length, vocabulary_size = load_artifact_summary()
        except Exception:
            sequence_length, vocabulary_size = 50, 31668

        st.markdown("---")
        st.markdown("**Model**  \\n+        LSTM")
        st.markdown("**Training Dataset**  \\n+        MAESTRO")
        st.markdown(f"**Sequence Length**  \\n+        {sequence_length} events")
        st.markdown(f"**Vocabulary**  \\n+        {vocabulary_size:,} events")
        st.markdown("---")
        st.caption(
            "Generated compositions are produced by an experimental neural network and may contain unusual musical patterns."
        )


def render_header() -> None:
    """Render the product identity and an explanatory generation workflow."""
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">NEURAL COMPOSITION LAB</div>
          <div class="hero-title">NeuraTune</div>
          <h3>AI Music Generation Studio</h3>
          <p class="hero-subtitle">Compose original classical piano sequences with a neural network trained on MIDI musical patterns.</p>
          <span class="badge">LSTM</span><span class="badge">TensorFlow</span>
          <span class="badge">music21</span><span class="badge">MAESTRO</span>
        </section>
        <div class="workflow"><strong>50-event seed</strong> &nbsp;→&nbsp; <strong>LSTM</strong>
        &nbsp;→&nbsp; Probability distribution &nbsp;→&nbsp; Temperature sampling &nbsp;→&nbsp;
        Next event &nbsp;→&nbsp; Repeat &nbsp;→&nbsp; MIDI</div>
        """,
        unsafe_allow_html=True,
    )


def render_controls() -> tuple[dict[str, int | float | None], bool]:
    """Render the generation controls and return the selected settings."""
    st.subheader("Compose a new sequence")
    with st.container(border=True):
        first_column, second_column = st.columns(2)
        with first_column:
            length = st.slider("Generation length", 50, 500, 200, 10)
            temperature = st.slider("Creativity / temperature", 0.3, 1.5, 1.0, 0.1)
            st.metric("Current temperature", f"{temperature:.1f}")
            st.caption("Lower is more predictable. 1.0 is balanced. Higher is more diverse.")
        with second_column:
            seed = st.number_input(
                "Random seed",
                min_value=0,
                value=42,
                step=1,
                help="The same seed and settings produce reproducible generation.",
            )
            tempo = st.slider("Tempo", 60, 180, 100, 1)
            duration_choice = st.selectbox(
                "Minimum duration",
                ["Original", "0.25", "0.5", "1.0"],
                index=2,
                help="This only changes MIDI playback reconstruction, not the AI prediction.",
            )
            st.caption("Minimum duration affects playback reconstruction only.")

        generate_clicked = st.button("Generate Composition", type="primary", use_container_width=True)

    settings: dict[str, int | float | None] = {
        "length": length,
        "temperature": temperature,
        "seed": int(seed),
        "tempo": tempo,
        "min_duration": None if duration_choice == "Original" else float(duration_choice),
    }
    return settings, generate_clicked


def render_distribution_chart(
    labels: list[str], values: list[int], title: str, color: str
) -> None:
    """Render a small dark Matplotlib bar chart without extra dependencies."""
    figure, axis = plt.subplots(figsize=(8, 3.2))
    figure.patch.set_facecolor("#11152a")
    axis.set_facecolor("#11152a")
    axis.bar(labels, values, color=color, edgecolor="#dcd8ff", linewidth=0.4)
    axis.set_title(title, color="#f4f1ff", loc="left", pad=12)
    axis.tick_params(colors="#c5c9e3")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#596083")
    axis.grid(axis="y", color="#394060", alpha=0.35)
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_result() -> None:
    """Render the current composition, its download action, and MIDI analysis."""
    result = st.session_state.latest_result
    if not result:
        st.markdown(
            """
            <div class="empty-state"><div class="empty-icon">✦</div>
            <h3>Your composition will appear here</h3>
            <p>Configure the studio controls above, then let the neural network compose a new MIDI sequence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    midi_path = Path(str(result["midi_path"]))
    if not midi_path.exists():
        st.error("The previous MIDI file is no longer available. Generate a new composition.")
        return

    st.markdown("---")
    st.subheader("Your AI Composition")
    metrics = result.get("metrics")
    settings = result["settings"]

    if metrics is None:
        st.warning("The MIDI was created, but its analysis could not be completed.")
    else:
        metric_columns = st.columns(5)
        metric_columns[0].metric("Generated Events", result["generated_events"])
        metric_columns[1].metric("Temperature", settings["temperature"])
        metric_columns[2].metric("Tempo", f"{settings['tempo']} BPM")
        metric_columns[3].metric("Duration", f"{metrics['total_duration']:.1f} ql")
        metric_columns[4].metric("Unique Pitches", metrics["unique_pitches"])

        detail_columns = st.columns(3)
        detail_columns[0].metric(
            "Pitch Range", f"{metrics['lowest_pitch']} – {metrics['highest_pitch']}"
        )
        detail_columns[1].metric("Random Seed", settings["seed"])
        minimum_duration = settings["min_duration"]
        detail_columns[2].metric(
            "Minimum Duration", minimum_duration if minimum_duration is not None else "Original"
        )

    st.download_button(
        "Download MIDI",
        data=midi_path.read_bytes(),
        file_name=midi_path.name,
        mime="audio/midi",
        type="primary",
        use_container_width=True,
    )
    st.caption("Open the downloaded MIDI in a DAW, MIDI player, or MuseScore. Browser audio playback is not included.")

    if metrics is not None:
        overview_tab, pitch_tab, rhythm_tab, model_tab = st.tabs(
            ["Overview", "Pitch Analysis", "Rhythm Analysis", "About the Model"]
        )
        with overview_tab:
            overview_columns = st.columns(3)
            overview_columns[0].metric("Total Notes", metrics["total_notes"])
            overview_columns[1].metric("Note Objects", metrics["note_count"])
            overview_columns[2].metric("Chord Objects", metrics["chord_count"])
            st.write(
                f"Average note duration: **{metrics['average_note_duration']:.2f}** quarter lengths  \\n+                Shortest: **{metrics['shortest_note_duration']:.2f}** · Longest: **{metrics['longest_note_duration']:.2f}**"
            )
        with pitch_tab:
            distribution = metrics["pitch_class_distribution"]
            render_distribution_chart(
                PITCH_CLASS_NAMES,
                [distribution.get(pitch_class, 0) for pitch_class in PITCH_CLASS_NAMES],
                "Pitch-class distribution",
                "#8d7aff",
            )
        with rhythm_tab:
            duration_distribution = metrics["duration_distribution"]
            sorted_durations = sorted(duration_distribution, key=float)
            render_distribution_chart(
                sorted_durations,
                [duration_distribution[duration] for duration in sorted_durations],
                "Duration distribution (quarter lengths)",
                "#59c8ff",
            )
        with model_tab:
            st.markdown(
                """
                **Dataset** · MAESTRO classical piano MIDI

                **Model** · Embedding → LSTM → Dropout → Softmax

                **Input** · Previous 50 musical events

                **Output** · A probability distribution for the next event
                **Generation** · Autoregressive temperature sampling

                Temperature controls diversity: lower values favor likely musical events, while higher values explore less likely alternatives.
                """
            )

    if result.get("skipped_events"):
        st.info(f"Skipped {result['skipped_events']} malformed event tokens during MIDI reconstruction.")
    if result.get("evaluation_error"):
        st.warning("MIDI analysis was unavailable for this composition.")

    if st.session_state.generation_history:
        with st.expander("Current session history"):
            for item in st.session_state.generation_history:
                st.caption(
                    f"{item['name']} · {item['events']} events · T={item['temperature']} · {item['tempo']} BPM"
                )


def main() -> None:
    """Run the Streamlit studio."""
    st.set_page_config(page_title="NeuraTune · AI Music Studio", page_icon="🎹", layout="wide")
    apply_theme()
    initialize_session_state()
    render_sidebar()
    render_header()
    settings, generate_clicked = render_controls()

    if generate_clicked:
        with st.spinner("Neural network is composing..."):
            try:
                result = generate_composition(settings)
            except FileNotFoundError:
                st.error("The trained model or processed artifacts are missing. Complete the backend phases first.")
            except Exception:
                st.error("Generation could not be completed. Check that the trained model and processed artifacts are valid.")
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
                st.success("Composition generated successfully.")

    render_result()


if __name__ == "__main__":
    main()
