import streamlit as st
import streamlit.components.v1 as components
import io
import json
import base64
import hmac
import re
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# Configure the Streamlit page
st.set_page_config(page_title="Secure Portal", page_icon="🔒", layout="centered")

SAMPLE_RATE = 24000


def check_password():
    """Returns `True` if the user has entered the correct password."""
    if st.session_state.get("password_correct", False):
        return True

    # Completely plain login screen
    st.title("🔒 Secure Login")
    st.write("Please authenticate to access this programme.")

    with st.form("login_form"):
        password_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if hmac.compare_digest(password_input, st.secrets["password"]):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Password incorrect. Please try again.")

    return False


# Stop execution if the password is not correct
if not check_password():
    st.stop()

# ==========================================
# POST-LOGIN: HSE BRANDING & APP LOGIC BELOW
# ==========================================

# Inject Custom HSE CSS Styling only after login
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Main Background & Font */
    .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Primary Buttons (Login & Generate) */
    .stButton > button {
        background: linear-gradient(135deg, #004d42 0%, #006858 50%, #00917a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #006858 0%, #00917a 50%, #00bfa5 100%);
        box-shadow: 0 10px 20px -8px rgba(0, 104, 88, 0.4);
        transform: translateY(-2px);
        color: white;
    }

    /* General Inputs */
    .stTextInput > div > div > input, .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #cbd5e1 !important;
    }

    /* FLOATING WHITE BOX FOR TEXT AREA */
    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 15px 35px -5px rgba(0, 104, 88, 0.15), 0 5px 15px -5px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        color: #1e293b !important;
        font-size: 1.05rem !important;
        line-height: 1.5 !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #00bfa5 !important;
        box-shadow: 0 15px 35px -5px rgba(0, 191, 165, 0.25), 0 5px 15px -5px rgba(0, 0, 0, 0.05) !important;
    }

    /* Custom Headers */
    h1, h2, h3 {
        color: #004d42 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        letter-spacing: -0.02em;
    }
    </style>
""", unsafe_allow_html=True)

# HSE Header
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 2rem; background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
        <img src="https://www.esther.ie/wp-content/uploads/2022/05/HSE-Logo-Green-NEW-no-background.png" width="80">
        <div>
            <h2 style="margin: 0; font-size: 1.5rem;">AI Text-to-Speech</h2>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem; font-weight: 500;">Specialist App Portal</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_pipeline():
    # Load the pipeline without triggering external model downloads
    return KPipeline(lang_code='b')


with st.spinner("Loading secure offline AI engine..."):
    pipeline = load_pipeline()

# Kokoro GB Voices
voices = {
    "👩🏼 Emma (GB)": "bf_emma",
    "👩🏽 Isabella (GB)": "bf_isabella",
    "👩🏻 Alice (GB)": "bf_alice",
    "👩🏼 Lily (GB)": "bf_lily",
    "👨🏼 George (GB)": "bm_george",
    "👨🏽 Fable (GB)": "bm_fable",
    "👨🏻 Lewis (GB)": "bm_lewis",
    "👨🏼 Daniel (GB)": "bm_daniel"
}


# ---------- Helpers ----------

def preprocess_for_tts(sentence: str) -> str:
    """Expand abbreviations so the synthesiser reads them naturally."""
    return re.sub(r'(?i)\bhse\b', 'H S E', sentence)


def split_sentences(text: str):
    """Simple sentence splitter that keeps the displayed text intact.

    Splits after . ! ? followed by whitespace. Text without terminal
    punctuation is treated as a single sentence.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def to_numpy(audio) -> np.ndarray:
    """Kokoro may return a torch tensor; coerce to a 1-D float32 numpy array."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def synthesise(text: str, voice_id: str, speed: float, gap_seconds: float = 0.12):
    """Synthesise sentence by sentence.

    Returns (full_audio, segments) where segments is a list of
    {text, start, end} in seconds, aligned to the concatenated audio.
    A short silence is inserted between sentences for more natural pacing.
    """
    sentences = split_sentences(text)
    gap = np.zeros(int(SAMPLE_RATE * gap_seconds), dtype=np.float32)

    audio_parts = []
    segments = []
    cursor = 0.0  # running position in seconds

    for idx, sentence in enumerate(sentences):
        processed = preprocess_for_tts(sentence)
        generator = pipeline(processed, voice=voice_id, speed=speed)

        chunk_audio = [to_numpy(audio) for _, _, audio in generator]
        if not chunk_audio:
            continue

        seg_audio = np.concatenate(chunk_audio)
        duration = len(seg_audio) / SAMPLE_RATE

        segments.append({
            "text": sentence,
            "start": round(cursor, 4),
            "end": round(cursor + duration, 4),
        })
        cursor += duration
        audio_parts.append(seg_audio)

        # Insert a small gap after every sentence except the last.
        if idx < len(sentences) - 1:
            audio_parts.append(gap)
            cursor += gap_seconds

    if not audio_parts:
        return None, []

    return np.concatenate(audio_parts), segments


def render_player(wav_bytes: bytes, segments: list):
    """Render a custom audio player that highlights the current sentence
    and lets the user click any sentence to play from there."""
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    seg_json = json.dumps(segments)

    html = """
    <div id="kokoro-card" style="
        font-family: 'Inter', system-ui, sans-serif;
        background: #ffffff;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 15px 35px -5px rgba(0, 104, 88, 0.15), 0 5px 15px -5px rgba(0,0,0,0.05);
        border: 1px solid rgba(226, 232, 240, 0.8);">

        <audio id="kokoro-player" controls preload="auto"
               style="width: 100%; margin-bottom: 14px;"
               src="data:audio/wav;base64, __B64__"></audio>

        <div id="kokoro-transcript" style="
            max-height: 220px;
            overflow-y: auto;
            line-height: 1.9;
            font-size: 1.05rem;
            color: #334155;
            padding-right: 6px;"></div>

        <p style="margin: 12px 0 0; color: #94a3b8; font-size: 0.78rem;">
            Click any sentence to play from there.
        </p>
    </div>

    <style>
        #kokoro-transcript .seg {
            cursor: pointer;
            padding: 1px 3px;
            border-radius: 5px;
            transition: background-color 0.15s ease, color 0.15s ease;
        }
        #kokoro-transcript .seg:hover {
            background-color: rgba(0, 191, 165, 0.12);
        }
        #kokoro-transcript .seg.active {
            background-color: #00917a;
            color: #ffffff;
        }
        #kokoro-transcript::-webkit-scrollbar { width: 8px; }
        #kokoro-transcript::-webkit-scrollbar-thumb {
            background: #cbd5e1; border-radius: 4px;
        }
    </style>

    <script>
        const segments = __SEGMENTS__;
        const player = document.getElementById('kokoro-player');
        const container = document.getElementById('kokoro-transcript');

        segments.forEach((seg, i) => {
            const span = document.createElement('span');
            span.className = 'seg';
            span.dataset.idx = i;
            span.textContent = seg.text + ' ';
            span.addEventListener('click', () => {
                player.currentTime = seg.start + 0.001;
                player.play();
            });
            container.appendChild(span);
        });

        const spans = container.querySelectorAll('.seg');
        let activeIdx = -1;

        function setActive(idx) {
            if (idx === activeIdx) return;
            if (activeIdx >= 0 && spans[activeIdx]) spans[activeIdx].classList.remove('active');
            if (idx >= 0 && spans[idx]) {
                spans[idx].classList.add('active');
                spans[idx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }
            activeIdx = idx;
        }

        player.addEventListener('timeupdate', () => {
            const t = player.currentTime;
            const idx = segments.findIndex(s => t >= s.start && t < s.end);
            if (idx === -1) return;   // in an inter-sentence gap: keep current
            setActive(idx);
        });

        player.addEventListener('ended', () => setActive(-1));
    </script>
    """

    html = (html
            .replace("__B64__", b64)
            .replace("__SEGMENTS__", seg_json))

    # Height = audio bar + transcript window + hint + padding.
    components.html(html, height=360)


# ---------- User Interface ----------

st.markdown("### Configuration")

voice_names = list(voices.keys())
selected_voice = st.selectbox(
    "Select Synthesiser Voice:",
    voice_names,
    index=voice_names.index("👩🏼 Lily (GB)")  # default to Lily
)

speed = st.slider(
    "Pace (lower = slower, more natural)",
    min_value=0.7, max_value=1.2, value=0.9, step=0.05
)

text_input = st.text_area(
    "Enter the text you wish to convert:",
    "Welcome to the HSE Capital and Estates digital infrastructure hub.",
    height=150
)

# Generation Button
if st.button("Synthesise Audio"):
    if text_input.strip() == "":
        st.warning("Please enter some text to synthesise.")
    else:
        with st.spinner("Generating audio securely..."):
            try:
                voice_id = voices[selected_voice]
                full_audio, segments = synthesise(text_input, voice_id, speed)

                if full_audio is None or len(segments) == 0:
                    raise ValueError("No audio was generated by the model.")

                # Encode WAV once and stash everything for re-renders.
                buffer = io.BytesIO()
                sf.write(buffer, full_audio, samplerate=SAMPLE_RATE, format='WAV')
                st.session_state["tts_result"] = {
                    "wav": buffer.getvalue(),
                    "segments": segments,
                }
            except Exception as e:
                st.error(f"An error occurred during synthesis: {e}")

# Display the player if we have a result (persists across reruns).
result = st.session_state.get("tts_result")
if result:
    st.success("✅ Audio synthesised securely and offline.")
    render_player(result["wav"], result["segments"])
    st.download_button(
        "Download WAV",
        data=result["wav"],
        file_name="hse_tts.wav",
        mime="audio/wav",
    )

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2rem;">
        <p style="font-weight: 900; color: #1e293b; margin-bottom: 0.2rem;">Digital Solutions Developed by Dave Maher</p>
        <p style="text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 1rem;">HSE Estates Infrastructure Intelligence</p>
        <p>&copy; 2026 Health Service Executive. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
