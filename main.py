from flask import Flask, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
import threading

app = Flask(__name__)

# Load model
model = load_model("final_model.h5")
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Global variables
latest_emotion = "neutral"
previous_emotion = None  # for change detection

def emotion_recognition():
    global latest_emotion, previous_emotion

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_emotion = latest_emotion  # default fallback

        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224,224))
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)

            preds = model.predict(face)
            detected_emotion = emotion_labels[np.argmax(preds)]

        # Update only when changed
        if detected_emotion != previous_emotion:
            latest_emotion = detected_emotion
            previous_emotion = detected_emotion
            print(f"📡 Emotion changed → {latest_emotion}")  # confirm in terminal

        # Optional: show window for debugging
        if len(faces) > 0:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, latest_emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow("Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask route
@app.route("/get_emotion", methods=["GET"])
def get_emotion():
    print(f"📡 Sending emotion: {latest_emotion}")
    return jsonify({"emotion": latest_emotion})

if __name__ == "__main__":
    # Run camera + ML in background thread
    t = threading.Thread(target=emotion_recognition, daemon=True)
    t.start()

    # Run Flask API
    app.run(host="127.0.0.1", port=5000)