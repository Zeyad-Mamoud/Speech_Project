import os
import pickle
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


TARGET_SR = 22_050
DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * DURATION)
MAX_LEN = 300


def newest_asset(pattern):
    matches = sorted(
        Path(".").glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        st.error(f"Missing required file matching: {pattern}")
        st.stop()
    return matches[0]


@st.cache_resource
def load_artifacts():
    model_path = newest_asset("emotion_model_enhanced_z*.h5")
    encoder_path = newest_asset("label_encoder_enhanced_z*.pkl")
    scalers_path = newest_asset("feature_scalers_enhanced_z*.pkl")

    loaded_model = load_model(model_path, compile=False)

    with open(encoder_path, "rb") as f:
        loaded_label_encoder = pickle.load(f)

    with open(scalers_path, "rb") as f:
        loaded_feature_scalers = pickle.load(f)

    required_scalers = {"mfcc", "mel", "chroma"}
    missing_scalers = required_scalers.difference(loaded_feature_scalers)
    if missing_scalers:
        st.error(f"Scaler file is missing: {', '.join(sorted(missing_scalers))}")
        st.stop()

    output_classes = loaded_model.output_shape[-1]
    encoder_classes = len(loaded_label_encoder.classes_)
    if output_classes != encoder_classes:
        st.error(
            "Model and label encoder do not match: "
            f"model has {output_classes} outputs, "
            f"encoder has {encoder_classes} classes."
        )
        st.stop()

    return loaded_model, loaded_label_encoder, loaded_feature_scalers, {
        "model": model_path.name,
        "label_encoder": encoder_path.name,
        "feature_scalers": scalers_path.name,
    }


model, label_encoder, feature_scalers, artifact_names = load_artifacts()

st.title("Speech Emotion Recognition")
st.caption(
    "Loaded "
    f"{artifact_names['model']}, "
    f"{artifact_names['label_encoder']}, "
    f"{artifact_names['feature_scalers']}"
)


def load_audio(path, sr=TARGET_SR, duration=DURATION):
    y, sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y.astype(np.float32), sr


def fix_audio_length(y, target_samples=TARGET_SAMPLES):
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode="constant")
    else:
        y = y[:target_samples]
    return y.astype(np.float32)


def preprocess_audio(path, sr=TARGET_SR, duration=DURATION, trim=True):
    y, sr = load_audio(path, sr=sr, duration=duration)
    y = fix_audio_length(y, int(sr * duration))
    return y, sr


def compute_features(y, sr=TARGET_SR):
    hop_length = int(0.010 * sr)
    n_fft = int(0.025 * sr)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        hop_length=hop_length,
        n_fft=n_fft,
    ).T

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_chroma=12,
        hop_length=hop_length,
        n_fft=n_fft,
    ).T

    return mfcc.astype(np.float32), mel_db.astype(np.float32), chroma.astype(np.float32)


def pad_feature_list(feature_list, max_len=MAX_LEN):
    return pad_sequences(
        feature_list,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        truncating="post",
        value=0.0,
    )


def transform_3d(array, scaler):
    n_features = array.shape[-1]
    return scaler.transform(array.reshape(-1, n_features)).reshape(array.shape).astype(np.float32)


def predict_emotion(audio_path):
    y, sr = preprocess_audio(audio_path)
    mfcc, mel, chroma = compute_features(y, sr)

    mfcc = pad_feature_list([mfcc])
    mel = pad_feature_list([mel])
    chroma = pad_feature_list([chroma])

    mfcc = transform_3d(mfcc, feature_scalers["mfcc"])
    mel = transform_3d(mel, feature_scalers["mel"])
    chroma = transform_3d(chroma, feature_scalers["chroma"])

    probabilities = model.predict([mfcc, mel, chroma], verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_emotion, float(probabilities[predicted_index]), pd.Series(
        probabilities,
        index=label_encoder.classes_,
    ).sort_values(ascending=False)


tab1, tab2 = st.tabs(["Single File", "Batch Processing (Folder)"])

with tab1:
    uploaded_file = st.file_uploader("Upload WAV or FLAC file", type=["wav", "flac"])

    if uploaded_file is not None:
        temp_path = "temp_uploaded_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            emotion, confidence, probabilities = predict_emotion(temp_path)
            st.success(f"Predicted Emotion: **{emotion}** ({confidence:.2%})")
            st.dataframe(
                probabilities.rename("Probability").to_frame(),
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab2:
    st.subheader("Batch Process WAV/FLAC Files from Folder")

    folder_path = st.text_input(
        "Enter folder path containing WAV or FLAC files:",
        placeholder="e.g., C:/path/to/folder",
        help="All .wav and .flac files in this folder will be processed",
    )

    if folder_path and st.button("Process Folder", key="process_folder"):
        if os.path.isdir(folder_path):
            audio_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".wav", ".flac"))
            ]

            if audio_files:
                st.info(f"Found {len(audio_files)} audio file(s). Processing...")

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, audio_file in enumerate(audio_files):
                    file_path = os.path.join(folder_path, audio_file)
                    status_text.text(f"Processing: {audio_file}")

                    try:
                        emotion, confidence, _ = predict_emotion(file_path)
                        results.append({
                            "File Name": audio_file,
                            "Predicted Emotion": emotion,
                            "Confidence": confidence,
                            "Status": "OK",
                        })
                    except Exception as exc:
                        results.append({
                            "File Name": audio_file,
                            "Predicted Emotion": "",
                            "Confidence": np.nan,
                            "Status": f"Error: {exc}",
                        })

                    progress_bar.progress((idx + 1) / len(audio_files))

                status_text.empty()
                progress_bar.empty()

                results_df = pd.DataFrame(results)
                st.subheader("Results")
                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="emotion_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No WAV or FLAC files found in the specified folder")
        else:
            st.error("Invalid folder path. Please check the path and try again")
