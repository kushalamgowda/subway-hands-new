import cv2
import mediapipe as mp
import pickle
import os

DATA_FILE = "data/gesture_data.pkl"
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "stop", "start"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def extract_features(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
    return features

def collect_data():
    data = {"features": [], "labels": []}

    cap = cv2.VideoCapture(0)
    for gesture in GESTURES:
        input(f"\n👉 Collecting data for: {gesture}\nPress ENTER when ready...")
        collected = 0

        while collected < 100:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    features = extract_features(handLms)
                    data["features"].append(features)
                    data["labels"].append(gesture)
                    collected += 1
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"{gesture}: {collected}/100", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    os.makedirs("data", exist_ok=True)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\n✅ Data collection complete. Saved at {DATA_FILE}")

if __name__ == "__main__":
    collect_data()
