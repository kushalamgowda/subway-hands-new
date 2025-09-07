# gesture_control.py

import os
import subprocess
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import threading

from adb_commands import send_command, connect_device
from analytics import AnalyticsLogger
from utils import volume_up, volume_down, mute_toggle   # ✅ Volume helpers
import config

# ------------------------------
# BlueStacks helper functions
# ------------------------------
def is_bluestacks_running():
    """Return True if a BlueStacks process appears in the Windows task list."""
    try:
        out = subprocess.check_output(["tasklist"], text=True, stderr=subprocess.DEVNULL)
        keys = ["hd-player.exe", "hdplayer.exe", "bluestacks.exe", "BlueStacks.exe", "HD-Player.exe"]
        out_l = out.lower()
        for k in keys:
            if k.lower() in out_l:
                return True
    except Exception:
        # If tasklist fails for some reason, treat it as not running (we'll try to launch)
        return False
    return False


def find_bluestacks_executable():
    """Try common install paths or config.BLUETSTACKS_PATH and return the first existing path, else None."""
    candidates = []
    # allow override from config
    try:
        if getattr(config, "BLUESTACKS_PATH", None):
            candidates.append(config.BLUETSTACKS_PATH)
    except Exception:
        pass

    # common installation locations
    candidates += [
        r"C:\Program Files\BlueStacks_nxt\HD-Player.exe",
        r"C:\Program Files\BlueStacks\HD-Player.exe",
        r"C:\Program Files (x86)\BlueStacks\HD-Player.exe",
        r"C:\Program Files (x86)\BlueStacks_nxt\HD-Player.exe",
    ]

    for p in candidates:
        if p and os.path.exists(p):
            return p

    # fall back to 'where' command for HD-Player.exe
    try:
        where_out = subprocess.check_output(["where", "HD-Player.exe"], text=True, stderr=subprocess.DEVNULL).strip().splitlines()
        for line in where_out:
            if os.path.exists(line):
                return line
    except Exception:
        pass

    return None


def launch_bluestacks(wait_after_launch=15):
    """
    Launch BlueStacks if not already running.
    wait_after_launch: seconds to wait after starting the process before attempting ADB.
    Returns True if launch was attempted or BlueStacks already running, False if executable not found.
    """
    if is_bluestacks_running():
        print("[INFO] BlueStacks already running.")
        return True

    exe = find_bluestacks_executable()
    if not exe:
        print("[ERROR] Could not find BlueStacks executable. Set config.BLUETSTACKS_PATH to its path if needed.")
        return False

    try:
        print(f"[INFO] Launching BlueStacks from: {exe}")
        # start without blocking; Windows will handle the UI
        subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[INFO] Waiting {wait_after_launch}s for BlueStacks to boot...")
        time.sleep(wait_after_launch)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to start BlueStacks: {e}")
        return False


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
# Map gestures to actions (Game + Volume)
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

    elif gesture == "start":
        print("[ACTION] Start detected – begin game")

    # 🔊 Volume Control Gestures
    elif gesture == "volume_up":
        volume_up()
        print("[ACTION] Volume Up")

    elif gesture == "volume_down":
        volume_down()
        print("[ACTION] Volume Down")

    elif gesture == "mute":
        mute_toggle()
        print("[ACTION] Mute Toggle")

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
                    perform_action(gesture, device)
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
    # 1) Try to auto-launch BlueStacks
    launch_bluestacks(wait_after_launch=getattr(config, "BLUESTACKS_BOOT_WAIT", 15))

    # 2) Best-effort: start adb server so devices can be seen
    try:
        subprocess.run(["adb", "start-server"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # 3) Poll connect_device() until we have a device or timeout
    timeout = getattr(config, "BLUESTACKS_CONNECT_TIMEOUT", 120)  # seconds
    poll_interval = getattr(config, "BLUESTACKS_POLL_INTERVAL", 3)  # seconds
    print(f"[INFO] Waiting up to {timeout}s for BlueStacks/ADB to become available...")

    device = None
    start_time = time.time()
    while time.time() - start_time < timeout:
        device = connect_device()
        if device:
            break
        print(f"[INFO] No device yet. Retrying in {poll_interval}s...")
        time.sleep(poll_interval)

    if not device:
        print(f"[ERROR] No device found after {timeout} seconds. Exiting...")
        exit(1)

    print("🚀 Starting Gesture Control (Game + Volume)...")
    gesture_loop(device)
