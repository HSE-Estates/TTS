import streamlit as st
import torch
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset
import io
import hmac

# Configure the Streamlit page
st.set_page_config(page_title="HSE Capital & Estates | TTS", page_icon="🗣️", layout="centered")

# --- Custom HSE CSS Styling ---
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
    
    /* Text Inputs and Text Areas */
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > textarea:focus {
        border-color: #00bfa5;
        box-shadow: 0 0 0 1px #00bfa5;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #004d42 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        letter-spacing: -0.02em;
    }
    
    /* Info/Warning Boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user has entered the correct password."""
    if st.session_state.get("password_correct", False):
        return True

    # HSE Branded Login Screen
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem 0;'>
            <img src="https://www.esther.ie/wp-content/uploads/2022/05/HSE-Logo-Green-NEW-no-background.png" width="140" style="margin-bottom: 1rem;">
            <h1>HSE Estates & Capital</h1>
            <p style="color: #006858; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; font-size: 0.8rem;">Digital Infrastructure Hub</p>
            <p style="color: #64748b; font-size: 1.1rem;">Secure Login Required</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Login Form with Submit Button
    with st.form("login_form"):
        password_input = st.text_input("Enter your access credential:", type="password")
        submitted = st.form_submit_button("Secure Login")
        
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

# --- Main Programme Content ---

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

# Load Models
@st.cache_resource
def load_models():
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    return synthesiser, embeddings_dataset

with st.spinner("Initialising secure speech models..."):
    synthesiser, embeddings_dataset = load_models()

voices = {
    "Voice 1 (Male)": 7306,
    "Voice 2 (Female)": 2271,
    "Voice 3 (Male)": 6799,
    "Voice 4 (Female)": 1138
}

# User Interface
st.markdown("### Configuration")
selected_voice = st.selectbox("Select Synthesiser Voice:", list(voices.keys()))

text_input = st.text_area(
    "Enter the text you wish to convert:", 
    "Welcome to the HSE Capital and Estates digital infrastructure hub.",
    height=120
)

# Generation Form/Button
if st.button("Synthesise Audio"):
    if text_input.strip() == "":
        st.warning("Please enter some text to synthesise.")
    else:
        with st.spinner("Processing audio array..."):
            try:
                speaker_index = voices[selected_voice]
                speaker_embedding = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)
                
                speech = synthesiser(text_input, forward_params={"speaker_embeddings": speaker_embedding})
                
                buffer = io.BytesIO()
                sf.write(buffer, speech["audio"], samplerate=speech["sampling_rate"], format='WAV')
                buffer.seek(0)
                
                st.success("✅ Audio synthesised successfully.")
                st.audio(buffer, format="audio/wav")
                
            except Exception as e:
                st.error(f"An error occurred during synthesis: {e}")

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
