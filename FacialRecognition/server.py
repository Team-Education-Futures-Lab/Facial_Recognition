from flask import Flask, jsonify, request
import threading
import time

app = Flask(__name__)

# ------------------------------
# Global state
# ------------------------------
latest_face_emotion = "neutral"
latest_voice_emotion = "none"
previous_face_emotion = None

latest_gesture = {
    "hand_sign": "none",
    "finger_gesture": "none"
}

latest_eye_data = {
    "eye_direction": "center",
    "blink_state": "open"
}

# ------------------------------
# Routes
# ------------------------------
@app.route("/get_emotion", methods=["GET"])
def get_emotion():
    """
    Returns the combined state for Unity.
    """
    return jsonify({
        "face_emotion": latest_face_emotion,
        "voice_emotion": latest_voice_emotion,
        "hand_sign": latest_gesture["hand_sign"],
        "finger_gesture": latest_gesture["finger_gesture"],
        "eye_direction": latest_eye_data["eye_direction"],
        "blink_state": latest_eye_data["blink_state"]
    })


@app.route("/gesture_receiver", methods=["POST"])
@app.route("/unity_endpoint", methods=["POST"])
def receive_gesture():
    global latest_gesture, latest_face_emotion
    data = request.get_json()

    # Update gesture data
    latest_gesture["hand_sign"] = data.get("hand_sign", "none")
    latest_gesture["finger_gesture"] = data.get("finger_gesture", "none")

    # Update face emotion if sent
    if "face_emotion" in data:
        latest_face_emotion = data["face_emotion"]

    print(f"📡 Received gesture data → Face={latest_face_emotion}, Hand={latest_gesture['hand_sign']}, Finger={latest_gesture['finger_gesture']}")
    return jsonify({"status": "ok"}), 200


@app.route("/voice-emotion", methods=["POST"])
def receive_voice_emotion():
    """
    Receives emotion data from the voice tone recognizer.
    Expects: {"emotion": "happy"}
    """
    global latest_voice_emotion
    data = request.get_json()
    latest_voice_emotion = data.get("emotion", "none")

    print(f"🎤 Voice emotion received → {latest_voice_emotion}")
    return jsonify({"status": "ok"}), 200


@app.route("/eye-tracking", methods=["POST"])
def receive_eye_tracking():
    """
    Optional endpoint for eye tracking.
    Expects: {"eye_direction": "left", "blink_state": "open"}
    """
    global latest_eye_data
    data = request.get_json()
    latest_eye_data["eye_direction"] = data.get("eye_direction", "center")
    latest_eye_data["blink_state"] = data.get("blink_state", "open")

    print(f"👁️ Eye tracking → {latest_eye_data}")
    return jsonify({"status": "ok"}), 200


# ------------------------------
# Simulated emotion changes (optional)
# ------------------------------
def simulate_emotion_updates():
    global latest_face_emotion, previous_face_emotion
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    idx = 0
    while True:
        latest_face_emotion = emotions[idx % len(emotions)]
        if latest_face_emotion != previous_face_emotion:
            previous_face_emotion = latest_face_emotion
            print(f"📡 Simulated face emotion → {latest_face_emotion}")
        idx += 1
        time.sleep(5)


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # Uncomment the next line if you want simulated face emotion changes
    # threading.Thread(target=simulate_emotion_updates, daemon=True).start()

    app.run(host="127.0.0.1", port=5000)