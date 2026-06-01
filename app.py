import streamlit as st
import io
import hmac
import re
import asyncio
import edge_tts
import tempfile
import os

# Configure the Streamlit page
st.set_page_config(page_title="Secure Portal", page_icon="🔒", layout="centered")

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

# Microsoft Edge GB Voices
voices = {
    "👩🏼 Emma (GB)": "en-GB-EmmaNeural",
    "👩🏽 Isabella (GB)": "en-GB-IsabellaNeural",
    "👩🏻 Alice (GB)": "en-GB-AliceNeural",
    "👩🏼 Lily (GB)": "en-GB-LilyNeural",
    "👨🏼 George (GB)": "en-GB-GeorgeNeural",
    "👨🏽 Fable (GB)": "en-GB-FableNeural",
    "👨🏻 Lewis (GB)": "en-GB-LewisNeural",
    "👨🏼 Daniel (GB)": "en-GB-DanielNeural"
}

# User Interface
st.markdown("### Configuration")
selected_voice = st.selectbox("Select Synthesiser Voice:", list(voices.keys()))

text_input = st.text_area(
    "Enter the text you wish to convert:", 
    "Welcome to the HSE Capital and Estates digital infrastructure hub.",
    height=150
)

# Robust async function to write safely to a temp file
async def generate_audio(text, voice_name, output_path):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_path)

# Generation Form/Button
if st.button("Synthesise Audio"):
    if text_input.strip() == "":
        st.warning("Please enter some text to synthesise.")
    else:
        with st.spinner("Connecting to Microsoft Neural Voice API..."):
            try:
                # 1. PRE-PROCESS TEXT: Use spaces instead of full stops to avoid silence bugs
                processed_text = re.sub(r'(?i)\bhse\b', 'H S E', text_input)
                
                # 2. Setup the Voice ID and temporary file path
                voice_id = voices[selected_voice]
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_path = fp.name
                
                # 3. Generate Audio 
                asyncio.run(generate_audio(processed_text, voice_id, temp_path))
                
                # 4. Read the safely saved audio
                with open(temp_path, "rb") as f:
                    audio_data = f.read()
                    
                # Clean up the temp file
                os.remove(temp_path)
                
                if not audio_data:
                    raise ValueError("Empty audio received. Microsoft Azure may be blocking this IP address.")
                
                # 5. Display audio player
                st.success("✅ Audio synthesised successfully.")
                st.audio(audio_data, format="audio/mp3")
                
            except Exception as e:
                st.error(f"An error occurred during synthesis: {e}")
                st.info("💡 If this persists, Microsoft Azure is likely blocking Streamlit Cloud's public IP address. Running this code locally on your own machine will solve it instantly.")

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
