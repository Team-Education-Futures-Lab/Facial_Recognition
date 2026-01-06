# main.py
import os, glob, librosa, numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

# ------------------------------
# 1. Emotion mapping from filename
# ------------------------------
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

# ------------------------------
# 2. Feature extraction
# ------------------------------
def extract_features(file_path, sr=22050, n_mfcc=40):
    y, _ = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y)  # remove silence
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features = np.hstack([mfcc_mean, mfcc_std, zcr, rms, spec_centroid])
    return features

# ------------------------------
# 3. Load dataset
# ------------------------------
def load_dataset(base_dir):
    X, y = [], []
    wav_files = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"⚠️ No audio files found in {base_dir}")
    for file in tqdm(wav_files, desc="Loading files"):
        fname = os.path.basename(file)
        emotion_id = fname.split("-")[2]
        if emotion_id in emotion_map:
            label = emotion_map[emotion_id]
            feat = extract_features(file)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)

# ------------------------------
# 4. Train / Evaluate model
# ------------------------------
def train_model(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler, le

# ------------------------------
# 5. Main execution
# ------------------------------
if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "data", "audio_speech_actors_01-24")
    X, y = load_dataset(base_path)
    if len(X) == 0:
        print("⚠️ No features extracted. Check your dataset path and files.")
        exit(1)

    model, scaler, label_encoder = train_model(X, y)

    # ------------------------------
    # 6. Save model, scaler, and encoder for later use
    # ------------------------------
    os.makedirs("saved_model", exist_ok=True)
    dump(model, "saved_model/model.joblib")
    dump(scaler, "saved_model/scaler.joblib")
    dump(label_encoder, "saved_model/label_encoder.joblib")

    print("\n✅ Training complete. Model saved in 'saved_model/' and ready for live inference.")