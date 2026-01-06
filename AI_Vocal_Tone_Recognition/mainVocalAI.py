
import sounddevice as sd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from joblib import load
import requests  # for sending predictions to Flask server

# ------------------------------
# Configuration
# ------------------------------
FLASK_SERVER_URL = "http://127.0.0.1:5000/voice-emotion"  # change to your Flask endpoint
RECORD_DURATION = 3  # seconds
SR = 22050  # sample rate

# ------------------------------
# Load your trained model
# ------------------------------
model = load("AI_Vocal_Tone_Recognition/saved_model/model.joblib")
scaler = load("AI_Vocal_Tone_Recognition/saved_model/scaler.joblib")
label_encoder = load("AI_Vocal_Tone_Recognition/saved_model/label_encoder.joblib")

# ------------------------------
# Feature extraction function
# ------------------------------
def extract_features(y, sr=SR, n_mfcc=40):
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features = np.hstack([mfcc_mean, mfcc_std, zcr, rms, spec_centroid])
    return features

# ------------------------------
# Record audio from microphone
# ------------------------------
def record_audio(duration=RECORD_DURATION, sr=SR):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.flatten()
    return audio

# ------------------------------
# Predict emotion
# ------------------------------
def predict_emotion(audio):
    features = extract_features(audio)
    features_scaled = scaler.transform([features])
    pred_label = model.predict(features_scaled)
    emotion = label_encoder.inverse_transform(pred_label)[0]
    return emotion

# ------------------------------
# Send to Flask server
# ------------------------------
def send_to_server(emotion):
    data = {"emotion": emotion}
    try:
        response = requests.post(FLASK_SERVER_URL, json=data)
        if response.status_code == 200:
            print(f"✅ Sent emotion to server: {emotion}")
        else:
            print(f"⚠️ Server returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send data: {e}")

# ------------------------------
# Main loop
# ------------------------------
if __name__ == "__main__":
    while True:
        audio = record_audio()
        emotion = predict_emotion(audio)
        print(f"Predicted Emotion: {emotion}")
        send_to_server(emotion)