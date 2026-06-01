import streamlit as st
import io
import gc
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
MAX_CHARS = 2000


def check_password():
    """Returns `True` if the user has entered the correct password."""
    if st.session_state.get("password_correct", False):
        return True

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
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem; background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
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

def render_embed(html: str, height: int):
    """Embed sandboxed HTML/JS. Prefer st.iframe (current API);
    fall back to the deprecated components.html on older Streamlit."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def preprocess_for_tts(sentence: str) -> str:
    """Expand abbreviations so the synthesiser reads them naturally."""
    return re.sub(r'(?i)\bhse\b', 'H S E', sentence)


def split_sentences(text: str):
    """Simple sentence splitter. Splits after . ! ? followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def to_numpy(audio) -> np.ndarray:
    """Kokoro may return a torch tensor; coerce to a 1-D float32 numpy array."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def estimate_words(sentence: str, base: float, duration: float):
    """Fallback: spread a sentence's duration across its words by length."""
    toks = sentence.split()
    if not toks:
        return []
    weights = [max(len(w), 1) for w in toks]
    total = sum(weights)
    out = []
    t = base
    for w, weight in zip(toks, weights):
        d = duration * (weight / total)
        out.append({"t": w, "ws": " ", "start": round(t, 4), "end": round(t + d, 4)})
        t += d
    return out


def synthesise(text: str, voice_id: str, speed: float, gap_seconds: float = 0.12):
    """Synthesise sentence by sentence and collect word-level timings.

    Returns (full_audio, words) where words is a list of
    {t, ws, start, end} aligned to the concatenated audio. Uses Kokoro's
    native token timestamps where available, otherwise estimates them.
    """
    sentences = split_sentences(text)
    gap = np.zeros(int(SAMPLE_RATE * gap_seconds), dtype=np.float32)

    audio_parts = []
    words = []
    cursor = 0.0  # global position in seconds

    for s_idx, sentence in enumerate(sentences):
        processed = preprocess_for_tts(sentence)
        generator = pipeline(processed, voice=voice_id, speed=speed)

        sentence_audio = []
        sentence_words = []
        chunk_offset = 0.0  # position within this sentence

        for result in generator:
            audio = result.audio if hasattr(result, "audio") else result[2]
            audio = to_numpy(audio)
            duration = len(audio) / SAMPLE_RATE

            for tok in (getattr(result, "tokens", None) or []):
                text_t = getattr(tok, "text", None)
                if not text_t:
                    continue
                ws = getattr(tok, "whitespace", "") or ""
                a = getattr(tok, "start_ts", None)
                b = getattr(tok, "end_ts", None)
                if a is not None and b is not None:
                    sentence_words.append({
                        "t": text_t, "ws": ws,
                        "start": round(cursor + chunk_offset + a, 4),
                        "end": round(cursor + chunk_offset + b, 4),
                    })
                else:
                    sentence_words.append({"t": text_t, "ws": ws,
                                           "start": None, "end": None})

            chunk_offset += duration
            sentence_audio.append(audio)

        if not sentence_audio:
            continue

        seg = np.concatenate(sentence_audio)
        seg_dur = len(seg) / SAMPLE_RATE

        # If this version returned no usable timestamps, estimate them.
        if not any(w["start"] is not None for w in sentence_words):
            sentence_words = estimate_words(sentence, cursor, seg_dur)

        words.extend(sentence_words)
        cursor += seg_dur
        audio_parts.append(seg)

        if s_idx < len(sentences) - 1:
            audio_parts.append(gap)
            cursor += gap_seconds

    if not audio_parts:
        return None, []

    return np.concatenate(audio_parts), words


def render_player(wav_bytes: bytes, words: list):
    """Custom player: highlights each word as it is spoken and lets the
    user click any word to play from there."""
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    words_json = json.dumps(words)

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
            padding-right: 6px;
            white-space: pre-wrap;"></div>

        <p style="margin: 12px 0 0; color: #94a3b8; font-size: 0.78rem;">
            Click any word to play from there.
        </p>
    </div>

    <style>
        #kokoro-transcript .wd {
            cursor: pointer;
            padding: 0 2px;
            border-radius: 4px;
            transition: background-color 0.12s ease, color 0.12s ease;
        }
        #kokoro-transcript .wd:hover { background-color: rgba(0, 191, 165, 0.12); }
        #kokoro-transcript .wd.active { background-color: #00917a; color: #ffffff; }
        #kokoro-transcript::-webkit-scrollbar { width: 8px; }
        #kokoro-transcript::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    </style>

    <script>
        const words = __WORDS__;
        const player = document.getElementById('kokoro-player');
        const container = document.getElementById('kokoro-transcript');
        const spans = [];

        words.forEach((w, i) => {
            const span = document.createElement('span');
            span.textContent = w.t;
            if (w.start !== null) {
                span.className = 'wd';
                span.addEventListener('click', () => {
                    player.currentTime = w.start + 0.001;
                    player.play();
                });
            }
            container.appendChild(span);
            if (w.ws) container.appendChild(document.createTextNode(w.ws));
            spans.push(span);
        });

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
            let idx = -1;
            for (let i = 0; i < words.length; i++) {
                const w = words[i];
                if (w.start !== null && t >= w.start && t < w.end) { idx = i; break; }
            }
            if (idx === -1) return;   // between words: hold current highlight
            setActive(idx);
        });

        player.addEventListener('ended', () => setActive(-1));
    </script>
    """

    html = (html
            .replace("__B64__", b64)
            .replace("__WORDS__", words_json))

    render_embed(html, height=380)


# ---------- User Interface ----------

# Settings tucked behind a gear icon, aligned top-right.
_, gear_col = st.columns([5, 1])
with gear_col:
    with st.popover("⚙️", use_container_width=True, help="Settings"):
        st.markdown("**Settings**")
        voice_names = list(voices.keys())
        selected_voice = st.selectbox(
            "Synthesiser voice",
            voice_names,
            index=voice_names.index("👩🏼 Lily (GB)")
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
    elif len(text_input) > MAX_CHARS:
        st.warning(f"Please keep text under {MAX_CHARS} characters (currently {len(text_input)}).")
    else:
        with st.spinner("Generating audio securely..."):
            try:
                voice_id = voices[selected_voice]
                full_audio, words = synthesise(text_input, voice_id, speed)

                if full_audio is None or len(words) == 0:
                    raise ValueError("No audio was generated by the model.")

                buffer = io.BytesIO()
                sf.write(buffer, full_audio, samplerate=SAMPLE_RATE, format='WAV')
                st.session_state["tts_result"] = {
                    "wav": buffer.getvalue(),
                    "words": words,
                }

                # Free large arrays and reclaim memory.
                del full_audio
                del buffer
                gc.collect()

            except Exception as e:
                st.error(f"An error occurred during synthesis: {e}")

# Display the player if we have a result (persists across reruns).
result = st.session_state.get("tts_result")
if result:
    st.success("✅ Audio synthesised securely and offline.")
    render_player(result["wav"], result["words"])
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
