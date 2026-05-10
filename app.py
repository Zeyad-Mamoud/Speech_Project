
import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd

# load model + encoder
model = load_model("emotion_model_enhanced.h5")

with open("label_encoder_enhanced.pkl", "rb") as f:
    le = pickle.load(f)

st.title("Speech Emotion Recognition")

# Create tabs for single file and batch processing
tab1, tab2 = st.tabs(["Single File", "Batch Processing (Folder)"])

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

def predict_emotion(audio_path):
    """Process a single audio file and return emotion prediction"""
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=4.0)
        
        if len(y) < sr * 4:
            y = np.pad(y, (0, int(sr*4) - len(y)))
        
        mfcc, mel, chroma = _compute_features(y, sr)
        
        mfcc = pad_sequences([mfcc], maxlen=300)
        mel = pad_sequences([mel], maxlen=300)
        chroma = pad_sequences([chroma], maxlen=300)
        
        pred = model.predict([mfcc, mel, chroma], verbose=0)
        emotion = le.inverse_transform([np.argmax(pred)])[0]
        
        return emotion
    except Exception as e:
        return f"Error: {str(e)}"

with tab1:
    uploaded_file = st.file_uploader("Upload WAV or FLAC file", type=["wav", "flac"])
    
    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        
        emotion = predict_emotion("temp.wav")
        st.success(f"Predicted Emotion: **{emotion}**")

with tab2:
    st.subheader("Batch Process WAV/FLAC Files from Folder")
    
    folder_path = st.text_input(
        "Enter folder path containing WAV or FLAC files:",
        placeholder="e.g., C:/path/to/folder",
        help="All .wav and .flac files in this folder will be processed"
    )
    
    if folder_path and st.button("Process Folder", key="process_folder"):
        if os.path.isdir(folder_path):
            wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.flac'))]
            
            if wav_files:
                st.info(f"Found {len(wav_files)} audio file(s). Processing...")
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, wav_file in enumerate(wav_files):
                    file_path = os.path.join(folder_path, wav_file)
                    status_text.text(f"Processing: {wav_file}")
                    
                    emotion = predict_emotion(file_path)
                    results.append({
                        "File Name": wav_file,
                        "Predicted Emotion": emotion
                    })
                    
                    progress_bar.progress((idx + 1) / len(wav_files))
                
                status_text.empty()
                progress_bar.empty()
                
                # Display results as table
                df = pd.DataFrame(results)
                st.subheader("Results")
                st.dataframe(df, use_container_width=True)
                
                # Option to download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="emotion_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No WAV files found in the specified folder")
        else:
            st.error("Invalid folder path. Please check the path and try again")
