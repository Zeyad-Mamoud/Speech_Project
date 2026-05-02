# Speech Emotion Recognition

Streamlit app for speech emotion recognition from WAV audio files.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app loads `emotion_model.h5` and `label_encoder.pkl`, then predicts the emotion for an uploaded WAV file.
