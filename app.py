import streamlit as st
import torch
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset
import io
import hmac

# Configure the Streamlit page
st.set_page_config(page_title="Hugging Face TTS", page_icon="🗣️", layout="centered")

def check_password():
    """Returns `True` if the user has entered the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password in session state
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.title("🔒 Login Required")
    st.text_input(
        "Please enter the password to access this programme:",
        type="password",
        on_change=password_entered,
        key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect. Please try again.")
    return False

# Stop execution if the password is not correct
if not check_password():
    st.stop()

st.title("🗣️ Text-to-Speech Web Programme")
st.write("Generate clear speech from text using Hugging Face's `SpeechT5` model.")

# Use st.cache_resource so the model and dataset are only loaded once per session
@st.cache_resource
def load_models():
    # Initialise the text-to-speech pipeline
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    # Load speaker embeddings dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    return synthesiser, embeddings_dataset

# Show a loading spinner whilst the model downloads/loads
with st.spinner("Loading models... This may take a minute on the first run."):
    synthesiser, embeddings_dataset = load_models()

# Define a selection of voices by mapping them to specific dataset indices
voices = {
    "Voice 1 (Male)": 7306,
    "Voice 2 (Female)": 2271,
    "Voice 3 (Male)": 6799,
    "Voice 4 (Female)": 1138
}

# User Interface Elements
selected_voice = st.selectbox("Select a Voice:", list(voices.keys()))

text_input = st.text_area(
    "Enter the text you want to convert:", 
    "Hello there! This is a test of the text-to-speech programme.",
    height=150
)

# Button to trigger generation
if st.button("Generate Audio", type="primary"):
    if text_input.strip() == "":
        st.warning("Please enter some text to synthesise.")
    else:
        with st.spinner("Synthesising audio..."):
            try:
                # 1. Get the specific speaker embedding for the selected voice
                speaker_index = voices[selected_voice]
                speaker_embedding = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)
                
                # 2. Generate the speech audio array
                speech = synthesiser(text_input, forward_params={"speaker_embeddings": speaker_embedding})
                
                # 3. Save the audio array into an in-memory byte buffer instead of a file
                buffer = io.BytesIO()
                sf.write(buffer, speech["audio"], samplerate=speech["sampling_rate"], format='WAV')
                buffer.seek(0)
                
                # 4. Display the audio player in Streamlit
                st.success("Audio generated successfully!")
                st.audio(buffer, format="audio/wav")
                
            except Exception as e:
                st.error(f"An error occurred during synthesis: {e}")
