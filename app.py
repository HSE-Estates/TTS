import streamlit as st
import io
import hmac
import re
from gtts import gTTS

# Configure Page
st.set_page_config(page_title="HSE Estates Intelligence", page_icon="🔒", layout="centered")

def check_password():
    if st.session_state.get("password_correct", False):
        return True
    
    st.title("🔒 Secure Access")
    with st.form("login"):
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Authenticate"):
            if hmac.compare_digest(pwd, st.secrets.get("password", "default")):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Incorrect Password")
    return False

if not check_password():
    st.stop()

# --- CSS Styling ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
    .stButton > button { background: linear-gradient(135deg, #004d42 0%, #006858 50%, #00917a 100%); color: white; border-radius: 8px; width: 100%; font-weight: 700; border: none; }
    div[data-testid="stTextArea"] textarea { background-color: #ffffff !important; border-radius: 16px !important; padding: 18px !important; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); border: 1px solid #e2e8f0 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""<div style="background: white; padding: 20px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px;"><h2 style="color: #004d42;">HSE AI Speech Intelligence</h2></div>""", unsafe_allow_html=True)

text_input = st.text_area("Enter text to convert:", "Welcome to the HSE Capital and Estates digital infrastructure hub.")

if st.button("Synthesise Audio"):
    if text_input.strip():
        with st.spinner("Processing..."):
            # Ensure 'HSE' is spoken as H S E
            clean_text = re.sub(r'(?i)\bhse\b', 'H S E', text_input)
            
            # Use Google TTS with UK English TLD
            tts = gTTS(text=clean_text, lang='en', tld='co.uk')
            
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            st.success("✅ Audio ready.")
            st.audio(audio_buffer, format="audio/mp3")

st.markdown("<p style='text-align: center; color: #64748b; margin-top: 50px;'>Digital Solutions by Dave Maher</p>", unsafe_allow_html=True)
