import streamlit as st
import requests
import plotly.graph_objects as go

BACKEND_URL = "http://localhost:8000"

EMOTION_EMOJIS = {
    "sadness" : "😢",
    "joy"     : "😊",
    "love"    : "❤️",
    "anger"   : "😠",
    "fear"    : "😨",
    "surprise": "😲",
}

EMOTION_COLORS = {
    "sadness" : "#4C72B0",
    "joy"     : "#FFD700",
    "love"    : "#FF69B4",
    "anger"   : "#C44E52",
    "fear"    : "#8172B2",
    "surprise": "#55A868",
}

SAMPLE_TEXTS = [
    "I feel so happy and excited about this!",
    "I am devastated and heartbroken right now.",
    "This is absolutely terrifying, I can't breathe.",
    "I love you so much, you mean everything to me.",
    "I am furious about what just happened!",
    "I can't believe it, I'm completely shocked!",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
)

st.title("🎭 Emotion Detector")
st.caption("Real-time emotion classification from English text")

# ── Check backend health ──────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def get_available_models():
    try:
        r = requests.get(f"{BACKEND_URL}/models", timeout=3)
        if r.status_code == 200:
            models = r.json()["models"]
            return [k for k, v in models.items() if v["loaded"]]
    except Exception:
        pass
    return []

available_models = get_available_models()

if not available_models:
    st.error("Cannot reach the backend. Make sure it is running:\n\n`uvicorn backend.app:app --port 8000 --reload`")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_labels = {
        "baseline"  : "Baseline (TF-IDF + LR)",
        "bilstm"    : "BiLSTM",
        "distilbert": "DistilBERT",
    }
    model_choice = st.selectbox(
        "Model",
        options=available_models,
        format_func=lambda k: model_labels.get(k, k),
    )

    show_highlights = st.checkbox("Show word highlights", value=True)

    st.divider()
    st.markdown("**Sample texts**")
    for sample in SAMPLE_TEXTS:
        if st.button(sample[:45] + "...", key=sample, use_container_width=True):
            st.session_state["input_text"] = sample

# ── Text input ────────────────────────────────────────────────────────────────
default_text = st.session_state.get("input_text", "")
text = st.text_area(
    "Enter text to analyse",
    value=default_text,
    height=120,
    placeholder="Type something like: I feel so happy today!",
)

predict_btn = st.button("Detect Emotion", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        endpoint = "/predict/explain" if show_highlights else "/predict"
        with st.spinner("Analysing..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json={"text": text, "model": model_choice},
                    timeout=15,
                )
                r.raise_for_status()
                result = r.json()
            except requests.exceptions.ConnectionError:
                st.error("Backend is not reachable. Start it with `uvicorn backend.app:app --port 8000 --reload`")
                st.stop()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        emotion    = result["emotion"]
        confidence = result["confidence"]
        emoji      = EMOTION_EMOJIS.get(emotion, "")
        probs      = result["probabilities"]

        # Result card
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"<div style='text-align:center; font-size:72px'>{emoji}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align:center; font-size:28px; font-weight:bold'>{emotion.capitalize()}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align:center; color:gray'>Confidence: {confidence*100:.1f}%</div>",
                unsafe_allow_html=True,
            )

        # Confidence bar chart
        st.divider()
        st.subheader("Confidence across all emotions")

        emotions = list(probs.keys())
        values   = [probs[e] * 100 for e in emotions]
        colors   = [EMOTION_COLORS.get(e, "#888") for e in emotions]

        fig = go.Figure(go.Bar(
            x=emotions,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(title="Confidence (%)", range=[0, 110]),
            xaxis=dict(title="Emotion"),
            showlegend=False,
            margin=dict(t=20, b=20),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Word highlights
        if show_highlights and "highlights" in result:
            st.divider()
            st.subheader("Word importance")
            st.caption("Green = supports the predicted emotion  |  Red = contradicts it")

            color_map = {
                "strong_positive"  : "#2ecc71",
                "moderate_positive": "#a8e6cf",
                "negative"         : "#e74c3c",
                "neutral"          : "#bdc3c7",
            }

            html_parts = []
            for h in result["highlights"]:
                bg    = color_map.get(h["color"], "#bdc3c7")
                score = h["score"]
                html_parts.append(
                    f"<span style='background:{bg}; padding:3px 6px; "
                    f"border-radius:4px; margin:2px; display:inline-block; "
                    f"font-size:16px' title='score: {score:.3f}'>{h['word']}</span>"
                )

            st.markdown(
                "<div style='line-height:2.2'>" + " ".join(html_parts) + "</div>",
                unsafe_allow_html=True,
            )

        # Latency
        st.caption(f"Model: `{result['model']}` | Latency: {result['latency_ms']} ms")
