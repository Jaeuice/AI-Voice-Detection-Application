import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tempfile
import soundfile as sf
import os

from pyannote.audio import Pipeline

# Load model
model = joblib.load('model/model.pkl')

# diarization pipeline (Hugging Face token needed)
@st.cache_resource
def load_pipeline():
    # Use environment variable for Hugging Face token
    # Set HF_TOKEN environment variable or provide token when running
    hf_token = os.getenv("HF_TOKEN", "your_huggingface_token_here")
    return Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

pipeline = load_pipeline()

# extract features from audios
def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)
    return features

# Streamlit UI
st.title("Multi-speaker Voice Classifier – AI vs Human 🤖🧑")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmp_path = tmpfile.name

    diarization = pipeline(tmp_path)

    audio, sr = librosa.load(tmp_path, sr=None)
    full_duration = librosa.get_duration(y=audio, sr=sr)

    st.write("🔍 Detected segments and predictions:")

    fig, ax = plt.subplots(figsize=(10, 2))

    seen_labels = set()

    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start = turn.start
        end = turn.end
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        segment = audio[start_sample:end_sample]
        features = extract_features(segment, sr)
        prediction = model.predict(features)[0]
        label = "AI" if prediction == 1 else "Human"

        st.write(f"🗣️ Speaker {speaker}: {label} voice ({start:.2f}s - {end:.2f}s)")

        color = "red" if label == "AI" else "green"
        legend_label = label if label not in seen_labels else ""
        ax.axvspan(start, end, color=color, alpha=0.4, label=legend_label)
        seen_labels.add(label)

    ax.set_xlim([0, full_duration])
    ax.set_xlabel("Time (s)")
    ax.set_title("Speaker segments: AI vs Human")
    ax.legend()
    st.pyplot(fig)

