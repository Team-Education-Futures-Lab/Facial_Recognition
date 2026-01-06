from flask import Flask, jsonify, request
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
import threading

app = Flask(__name__)

# Load face emotion model
model = load_model("FacialRecognition/final_model.h5")
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Global variables
latest_face_emotion = "neutral"
latest_voice_emotion = "neutral"
previous_emotion = None

# ------------------------------
# Flask Routes
# ------------------------------
@app.route("/voice-emotion", methods=["POST"])
def receive_voice_emotion():
    global latest_voice_emotion
    data = request.get_json()
    latest_voice_emotion = data.get("emotion", "neutral")
    print(f"📡 Received voice emotion: {latest_voice_emotion}")
    return jsonify({"status": "ok"})

@app.route("/get_emotion", methods=["GET"])
def get_emotion():
    return jsonify({
        "face_emotion": latest_face_emotion,
        "voice_emotion": latest_voice_emotion
    })

@app.route('/gesture_receiver', methods=['POST'])
def receive_gesture():
    data = request.get_json()
    print("Received gesture:", data)
    return jsonify({"status": "ok"}), 200

# ------------------------------
# Face Emotion Thread
# ------------------------------
def emotion_recognition():
    global latest_face_emotion, previous_emotion
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        detected_emotion = latest_face_emotion

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)
            preds = model.predict(face)
            detected_emotion = emotion_labels[np.argmax(preds)]

        if detected_emotion != previous_emotion:
            latest_face_emotion = detected_emotion
            previous_emotion = detected_emotion
            print(f"📡 Emotion changed → {latest_face_emotion}")

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, latest_face_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Face Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    threading.Thread(target=emotion_recognition, daemon=True).start()
    app.run(host="127.0.0.1", port=5000)