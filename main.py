import sounddevice as sd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from joblib import load  # To load your saved model, scaler, label encoder

# ------------------------------
# Load your trained model
# ------------------------------
model = load("saved_model/model.joblib")        # RandomForestClassifier
scaler = load("saved_model/scaler.joblib")      # StandardScaler
label_encoder = load("saved_model/label_encoder.joblib")   # LabelEncoder

# ------------------------------
# Feature extraction function
# ------------------------------
def extract_features(y, sr=22050, n_mfcc=40):
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
def record_audio(duration=3, sr=22050):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.flatten()  # convert to 1D array
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
# Main loop
# ------------------------------
if __name__ == "__main__":
    while True:
        audio = record_audio(duration=3)  # 3-second recording
        emotion = predict_emotion(audio)
        print(f"Predicted Emotion: {emotion}")