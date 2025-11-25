# collect_data.py

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

import config

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


# ------------------------------
# Helper: Extract Features
# ------------------------------
def extract_features(landmarks):
    """Extract (x,y) coordinates of 21 hand landmarks into a feature vector"""
    features = []
    for lm in landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
    return np.array(features)


# ------------------------------
# Main Data Collection
# ------------------------------
def collect_data(gesture_name, samples=200):
    print(f"[INFO] Starting data collection for: {gesture_name}")

    cap = cv2.VideoCapture(0)
    data = []
    count = 0

    while cap.isOpened() and count < samples:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = extract_features(hand_landmarks)
                data.append(features)

                count += 1
                cv2.putText(frame, f"{gesture_name}: {count}/{samples}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Gesture Data", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save data
    if len(data) > 0:
        if os.path.exists(config.DATA_FILE):
            with open(config.DATA_FILE, "rb") as f:
                all_data = pickle.load(f)
        else:
            all_data = {}

        all_data[gesture_name] = all_data.get(gesture_name, []) + data

        with open(config.DATA_FILE, "wb") as f:
            pickle.dump(all_data, f)

        print(f"[INFO] Saved {len(data)} samples for '{gesture_name}' in {config.DATA_FILE}")


if __name__ == "__main__":
    # 🕹️ Game gestures
    gestures = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "start", "stop","double_tap"]

    # 🔊 Volume gestures
    gestures += ["volume_up", "volume_down", "mute"]

    for gesture in gestures:
        collect_data(gesture, samples=100)  # Collect 100 samples each
