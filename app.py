
import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load model + encoder
model = load_model("emotion_model_enhanced.h5")

with open("label_encoder_enhanced.pkl", "rb") as f:
    le = pickle.load(f)

st.title("🎤 Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

def _compute_features(y, sr=22050):
    hop_length = int(0.010 * sr)
    n_fft = int(0.025 * sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,
                                hop_length=hop_length, n_fft=n_fft).T

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                         hop_length=hop_length, n_fft=n_fft)
    mel_db = librosa.power_to_db(mel, ref=np.max).T

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12,
                                         hop_length=hop_length, n_fft=n_fft).T

    # normalization
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
    chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-6)

    return mfcc, mel_db, chroma

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    y, sr = librosa.load("temp.wav", sr=22050, duration=4.0)

    if len(y) < sr * 4:
        y = np.pad(y, (0, int(sr*4) - len(y)))

    mfcc, mel, chroma = _compute_features(y, sr)

    mfcc = pad_sequences([mfcc], maxlen=300)
    mel = pad_sequences([mel], maxlen=300)
    chroma = pad_sequences([chroma], maxlen=300)

    pred = model.predict([mfcc, mel, chroma])
    emotion = le.inverse_transform([np.argmax(pred)])[0]

    st.success(f"Predicted Emotion: {emotion}")