import cv2
import numpy as np
import mediapipe as mp
import requests
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
from collections import Counter, deque
import itertools

# ------------------------------
# Configuration
# ------------------------------
FLASK_SERVER_URL = "http://127.0.0.1:5000/gesture_receiver"  # Flask endpoint
HISTORY_LENGTH = 16

# ------------------------------
# Load models
# ------------------------------
face_model = load_model("FacialRecognition/final_model.h5")
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

from model import KeyPointClassifier, PointHistoryClassifier
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

with open('FacialRecognition/model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_labels = [row[0] for row in f]

with open('FacialRecognition/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in f]

# ------------------------------
# Helper functions
# ------------------------------
def send_to_server(data):
    try:
        response = requests.post(FLASK_SERVER_URL, json=data, timeout=1)
        if response.status_code == 200:
            print(f"✅ Sent data: {data}")
        else:
            print(f"⚠️ Server returned {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send data: {e}")

def preprocess_landmarks(landmark_list):
    temp = [[p[0]-landmark_list[0][0], p[1]-landmark_list[0][1]] for p in landmark_list]
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat)) or 1
    return [x/max_val for x in flat]

# ------------------------------
# Video capture & detection
# ------------------------------
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

point_history = deque(maxlen=HISTORY_LENGTH)
finger_gesture_history = deque(maxlen=HISTORY_LENGTH)
latest_face_emotion = "neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    debug_image = frame.copy()

    # ---- Face emotion ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = mp_face.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
        face_input = np.expand_dims(face, axis=0)
        face_input = preprocess_input(face_input)
        preds = face_model.predict(face_input)
        latest_face_emotion = emotion_labels[np.argmax(preds)]
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_image, latest_face_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # ---- Hand gesture ----
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    hand_sign_name = "none"
    finger_gesture_name = "none"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = [[int(lm.x*frame.shape[1]), int(lm.y*frame.shape[0])] for lm in hand_landmarks.landmark]

            # Static hand sign
            pre_processed = preprocess_landmarks(landmark_list)
            hand_sign_id = keypoint_classifier(pre_processed)
            hand_sign_name = keypoint_classifier_labels[hand_sign_id]

            # Track index finger for dynamic gesture
            if hand_sign_name in ["point", "open", "close"]:
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            # Dynamic gesture
            if len(point_history) == HISTORY_LENGTH:
                base_x, base_y = point_history[0]
                pre_history = list(itertools.chain.from_iterable(
                    [[(p[0]-base_x)/frame.shape[1], (p[1]-base_y)/frame.shape[0]] for p in point_history]
                ))
                finger_gesture_id = point_history_classifier(pre_history)
                finger_gesture_history.append(finger_gesture_id)
                most_common_id = Counter(finger_gesture_history).most_common(1)[0][0]
                finger_gesture_name = point_history_labels[most_common_id]
            else:
                finger_gesture_history.append(0)
                finger_gesture_name = point_history_labels[0]

    # ---- Send combined data ----
    send_to_server({
        "face_emotion": latest_face_emotion,
        "hand_sign": hand_sign_name,
        "finger_gesture": finger_gesture_name
    })

    # ---- Display ----
    cv2.imshow("Face & Hand Recognition", debug_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()