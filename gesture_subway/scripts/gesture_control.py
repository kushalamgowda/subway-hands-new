# gesture_control.py
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import subprocess
import os

from utils import volume_up, volume_down, mute_toggle
import config

# ------------------------------
# Paths
# ------------------------------
BLUESTACKS_EXE = config.BLUESTACKS_PATH
ADB_EXE = config.ADB_PATH
SUBWAY_SURFER_PACKAGE = config.SUBWAY_SURFER_PACKAGE
SUBWAY_SURFER_ACTIVITY = config.SUBWAY_SURFER_ACTIVITY

# ------------------------------
# Load Model
# ------------------------------
with open(config.MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ------------------------------
# Helper Functions
# ------------------------------
def run_adb(args):
    """Run adb command using full path."""
    cmd = [ADB_EXE] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def get_device():
    """Return the first connected device or None."""
    output = run_adb(["devices"])
    lines = output.splitlines()
    for line in lines[1:]:
        if line.strip() and "device" in line:
            return line.split()[0]
    return None

def wait_for_device(timeout=120, poll_interval=3):
    start = time.time()
    while time.time() - start < timeout:
        device = get_device()
        if device:
            return device
        print(f"[INFO] Waiting for device... retrying in {poll_interval}s")
        time.sleep(poll_interval)
    return None

def launch_bluestacks(wait_after_launch=15):
    if not os.path.exists(BLUESTACKS_EXE):
        print(f"[ERROR] BlueStacks not found at {BLUESTACKS_EXE}")
        return
    print(f"[INFO] Launching BlueStacks...")
    subprocess.Popen([BLUESTACKS_EXE])
    print(f"[INFO] Waiting {wait_after_launch}s for BlueStacks to boot...")
    time.sleep(wait_after_launch)

def launch_subway_surfer(device):
    try:
        subprocess.run([ADB_EXE, "-s", device, "shell",
                        "monkey", "-p", SUBWAY_SURFER_PACKAGE,
                        "-c", "android.intent.category.LAUNCHER", "1"],
                       check=True)
        print("[INFO] Subway Surfer launched!")
    except Exception as e:
        print(f"[ERROR] Could not launch Subway Surfer: {e}")

def extract_features(landmarks):
    features = []
    for lm in landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
    return np.array(features).reshape(1, -1)

def perform_action(gesture, device):
    if gesture == "swipe_left":
        run_adb(["-s", device, "shell", "input", "swipe", "600", "1000", "200", "1000"])
    elif gesture == "swipe_right":
        run_adb(["-s", device, "shell", "input", "swipe", "400", "1000", "800", "1000"])
    elif gesture == "swipe_up":
        run_adb(["-s", device, "shell", "input", "swipe", "500", "1000", "500", "500"])
    elif gesture == "swipe_down":
        run_adb(["-s", device, "shell", "input", "swipe", "500", "1000", "500", "1500"])
    elif gesture == "volume_up":
        volume_up()
    elif gesture == "volume_down":
        volume_down()
    elif gesture == "mute":
        mute_toggle()

# ------------------------------
# Gesture Loop
# ------------------------------
def gesture_loop(device):
    # Use CAP_DSHOW to fix black screen issue on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    prev_gesture = None
    last_time = time.time()

    print("[INFO] Gesture control started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from camera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = extract_features(hand_landmarks)
                gesture = model.predict(features)[0]

                cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if gesture != prev_gesture and (time.time() - last_time) > 0.5:
                    perform_action(gesture, device)
                    prev_gesture = gesture
                    last_time = time.time()

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    launch_bluestacks(wait_after_launch=config.BLUESTACKS_BOOT_WAIT)

    # Start adb server
    run_adb(["start-server"])

    device = wait_for_device(timeout=config.BLUESTACKS_CONNECT_TIMEOUT)
    if not device:
        print("[ERROR] No device detected. Exiting...")
        exit(1)

    launch_subway_surfer(device)
    print("🚀 Starting Gesture Control (Game + Volume)...")
    gesture_loop(device)
