# gesture_control.py

import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import threading

from adb_commands import send_command, connect_device
from analytics import AnalyticsLogger
import config

# ------------------------------
# Load Model
# ------------------------------
with open(config.MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Analytics logger
logger = AnalyticsLogger()


# ------------------------------
# Helper: Extract Features
# ------------------------------
def extract_features(landmarks):
    """Extract (x,y) coordinates of 21 hand landmarks into a feature vector"""
    features = []
    for lm in landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
    return np.array(features).reshape(1, -1)


# ------------------------------
# Map gestures to game actions
# ------------------------------
def perform_action(gesture, device):
    if gesture == "swipe_left":
        threading.Thread(target=send_command, args=("LEFT", device)).start()
        print("[ACTION] Move Left")
    elif gesture == "swipe_right":
        threading.Thread(target=send_command, args=("RIGHT", device)).start()
        print("[ACTION] Move Right")
    elif gesture == "swipe_up":
        threading.Thread(target=send_command, args=("JUMP", device)).start()
        print("[ACTION] Jump")
    elif gesture == "swipe_down":
        threading.Thread(target=send_command, args=("DUCK", device)).start()
        print("[ACTION] Duck")
    elif gesture == "stop":
        print("[ACTION] Stop detected – no movement")
        # Optional: could pause game or just ignore
    elif gesture == "start":
        print("[ACTION] Start detected – begin game")
        # Optional: send start command
    else:
        print(f"[INFO] Ignored gesture: {gesture}")


# ------------------------------
# Real-time Gesture Recognition
# ------------------------------
def gesture_loop(device):
    cap = cv2.VideoCapture(0)  # webcam
    prev_gesture = None
    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand skeleton
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract features
                features = extract_features(hand_landmarks)

                # Predict gesture
                gesture = model.predict(features)[0]

                # Display gesture on frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Only act if gesture changed and cooldown passed
                if gesture != prev_gesture and (time.time() - last_time) > 0.5:
                    perform_action(gesture, device)   # 👈 now sends with device
                    logger.log_gesture(gesture)
                    prev_gesture = gesture
                    last_time = time.time()

        cv2.imshow("Gesture Control", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.log_session()  # Save analytics at the end


if __name__ == "__main__":
    device = connect_device()   # Get connected BlueStacks device
    if not device:
        print("[ERROR] No device found. Exiting...")
        exit(1)

    print("🚀 Starting Gesture Control...")
    gesture_loop(device)

