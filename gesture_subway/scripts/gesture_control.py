# gesture_control.py
"""
Gesture control runner for Subway Surfer (BlueStacks)
Features:
 - Loads trained model from config.MODEL_FILE
 - Uses MediaPipe hands for real-time inference
 - Smooths predictions using a sliding window and majority vote
 - Sends adb input commands via config.ADB_PATH (or prints them in --dry-run)
 - CLI flags: --dry-run, --no-bluestacks, --camera
"""

import argparse
import os
import sys
import time
import pickle
import subprocess
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np

# project imports (ensure project root is on sys.path if running from scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from analytics import AnalyticsLogger

# ------------------------------
# CLI and runtime flags (set in main)
# ------------------------------
DRY_RUN = False
NO_BLUETSTACKS = False
CAMERA_INDEX = 0

# ------------------------------
# Analytics setup
# ------------------------------
ANALYTICS_PATH = os.path.join("data", "analytics_log.json")
os.makedirs(os.path.dirname(ANALYTICS_PATH) or ".", exist_ok=True)
analytics = AnalyticsLogger(log_file=ANALYTICS_PATH)

# ------------------------------
# ADB / BlueStacks config
# ------------------------------
BLUESTACKS_EXE = config.BLUESTACKS_PATH
ADB_EXE = config.ADB_PATH
SUBWAY_SURFER_PACKAGE = config.SUBWAY_SURFER_PACKAGE
SUBWAY_SURFER_ACTIVITY = config.SUBWAY_SURFER_ACTIVITY

# ------------------------------
# Model load
# ------------------------------
try:
    with open(config.MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded from {config.MODEL_FILE}")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at {config.MODEL_FILE}")
    raise
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    raise

# ------------------------------
# MediaPipe setup
# ------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# If you want to tune these, change the constants or pass them via config
WINDOW_SIZE = getattr(config, "WINDOW_SIZE", 7)
CONF_THRESHOLD = getattr(config, "CONF_THRESHOLD", 0.75)
ACTION_COOLDOWN = getattr(config, "ACTION_COOLDOWN", 0.5)

# ------------------------------
# Helper: run adb
# ------------------------------
def run_adb(args):
    """Run adb command using full path and return stdout/stderr (string)."""
    cmd = [ADB_EXE] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"[ADB-ERR] cmd={' '.join(cmd)} rc={result.returncode} stderr={result.stderr.strip()}")
        return result.stdout.strip()
    except Exception as e:
        print(f"[ADB-EXC] {e} while running {' '.join(cmd)}")
        return ""

def get_device():
    """Return the first connected device id or None."""
    output = run_adb(["devices"])
    if not output:
        return None
    lines = output.splitlines()
    # first line normally "List of devices attached"
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1] == "device":
            return parts[0]
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

def launch_bluestacks(wait_after_launch=25):
    if not os.path.exists(BLUESTACKS_EXE):
        print(f"[ERROR] BlueStacks not found at {BLUESTACKS_EXE}")
        return None
    try:
        print("[INFO] Launching BlueStacks...")
        proc = subprocess.Popen([BLUESTACKS_EXE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[INFO] Waiting {wait_after_launch}s for BlueStacks to boot...")
        time.sleep(wait_after_launch)
        return proc
    except Exception as e:
        print(f"[ERROR] Failed to launch BlueStacks: {e}")
        return None

def launch_subway_surfer(device):
    try:
        subprocess.run([ADB_EXE, "-s", device, "shell", "monkey", "-p", SUBWAY_SURFER_PACKAGE,
                        "-c", "android.intent.category.LAUNCHER", "1"], check=True)
        print("[INFO] Subway Surfer launched!")
    except Exception as e:
        print(f"[ERROR] Could not launch Subway Surfer: {e}")

# ------------------------------
# Feature extraction: matches collect_data normalized features (wrist-centered)
# ------------------------------
def normalize_landmarks(landmarks):
    lm = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=float)
    center = lm[0].copy()
    lm -= center
    dists = np.linalg.norm(lm, axis=1)
    max_dist = np.max(dists)
    if max_dist > 1e-6:
        lm /= max_dist
    return lm.flatten()

def extract_features(landmarks):
    return normalize_landmarks(landmarks).reshape(1, -1)

# ------------------------------
# Map gesture -> adb args
# ------------------------------
def build_adb_args_for_gesture(gesture, device):
    prefix = ["-s", device, "shell"]
    if gesture == "swipe_left":
        return [prefix + ["input", "swipe", "600", "1000", "200", "1000", "200"]]
    elif gesture == "swipe_right":
        return [prefix + ["input", "swipe", "400", "1000", "800", "1000", "200"]]
    elif gesture == "swipe_up":
        return [prefix + ["input", "swipe", "500", "1200", "500", "400", "350"]]
    elif gesture == "swipe_down":
        return [prefix + ["input", "swipe", "500", "1000", "500", "1500", "200"]]
    elif gesture == "stop":
        # multiple candidate tap coords (tries each in order)
        return [
            prefix + ["input", "tap", "1000", "150"],
            prefix + ["input", "tap", "980", "120"],
            prefix + ["input", "tap", "1050", "170"],
            prefix + ["input", "tap", "1100", "80"],
            prefix + ["input", "tap", "70", "60"]
        ]
    elif gesture == "start":
        return [prefix + ["input", "tap", "600", "1200"]]
    else:
        return None

def perform_action(gesture, device):
    """Log + execute (or print in dry-run) the adb action(s) for a gesture."""
    if not device:
        print("[WARN] No device provided to perform_action.")
        return

    cmd_list = build_adb_args_for_gesture(gesture, device)
    if not cmd_list:
        print(f"[INFO] No mapped action for gesture '{gesture}'")
        return

    # analytics log (non-fatal)
    try:
        analytics.log_gesture(gesture)
    except Exception as e:
        print(f"[ANALYTICS-ERR] Could not log gesture: {e}")

    # If multiple candidate commands, try each (good fallback for pause)
    for args in cmd_list:
        # Print command for debugging
        print(f"[ADB-CMD] adb {' '.join(args)}")
        if DRY_RUN:
            print(f"[DRY-RUN] would run: adb {' '.join(args)}")
        else:
            out = run_adb(args)
            if out:
                print(f"[ADB-OUT] {out}")
        # small delay between attempts
        time.sleep(0.18)

# ------------------------------
# Gesture loop (real-time): smoothing + confidence gating
# ------------------------------
def gesture_loop(device, camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(camera_index)
    smoothed_queue = deque(maxlen=WINDOW_SIZE)
    last_action_time = 0
    frame_no = 0

    print("[INFO] Gesture control started. Press 'q' in window or Ctrl+C to quit.")

    # create hands instance here, will be closed on exit
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                frame_no += 1
                if not ret:
                    print("[WARNING] Failed to read frame from camera.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                pred_label = None
                pred_conf = 0.0

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    try:
                        features = extract_features(hand_landmarks)
                    except Exception as e:
                        print(f"[WARN] Feature extraction failed: {e}")
                        features = None

                    if features is not None:
                        # prediction with confidence if available
                        try:
                            probs = model.predict_proba(features)[0]
                            best_idx = int(np.argmax(probs))
                            pred_conf = float(probs[best_idx])
                            pred_label = model.classes_[best_idx] if hasattr(model, "classes_") else str(best_idx)
                        except Exception:
                            # fallback
                            pred_label = model.predict(features)[0]
                            pred_conf = 1.0

                        # confidence gating
                        if pred_conf >= CONF_THRESHOLD:
                            smoothed_queue.append(pred_label)
                        else:
                            smoothed_queue.append(None)

                        # majority vote ignoring None
                        votes = [v for v in smoothed_queue if v is not None]
                        stable_label = None
                        if votes:
                            most_common, cnt = Counter(votes).most_common(1)[0]
                            if cnt >= (WINDOW_SIZE // 2) + 1:
                                stable_label = most_common

                        cv2.putText(frame, f"Frame:{frame_no} Pred:{pred_label} Conf:{pred_conf:.2f} Smoothed:{stable_label}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # record to analytics (non-fatal)
                        try:
                            analytics.data.setdefault("predictions", []).append({
                                "time": time.time(),
                                "frame": frame_no,
                                "predicted": str(pred_label),
                                "confidence": float(pred_conf),
                                "stable_label": str(stable_label) if stable_label is not None else None
                            })
                        except Exception as e:
                            print(f"[ANALYTICS-ERR] Could not record prediction: {e}")

                        # Fire action if stable and cooldown passed
                        now = time.time()
                        if stable_label and (now - last_action_time) > ACTION_COOLDOWN:
                            print(f"[ACTION] frame={frame_no} label={stable_label} conf={pred_conf:.2f}")
                            perform_action(stable_label, device)
                            last_action_time = now

                else:
                    # no hands
                    smoothed_queue.append(None)

                cv2.imshow("Gesture Control (press 'q' to quit)", frame)
                # small sleep and key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] 'q' pressed — exiting gesture loop.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt received — stopping gesture loop.")
            raise
        finally:
            cap.release()
            cv2.destroyAllWindows()

# ------------------------------
# Main: CLI handling and orchestration
# ------------------------------
def main():
    global DRY_RUN, NO_BLUETSTACKS, CAMERA_INDEX

    parser = argparse.ArgumentParser(description="Gesture control (live).")
    parser.add_argument("--dry-run", action="store_true", help="Print ADB commands instead of executing them")
    parser.add_argument("--no-bluestacks", action="store_true", help="Do not launch BlueStacks (use if it's already running)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use (default 0)")
    args = parser.parse_args()

    DRY_RUN = args.dry_run
    NO_BLUETSTACKS = args.no_bluestacks
    CAMERA_INDEX = args.camera

    print(f"[INFO] Flags: DRY_RUN={DRY_RUN}, NO_BLUETSTACKS={NO_BLUETSTACKS}, CAMERA_INDEX={CAMERA_INDEX}")

    bluestacks_process = None
    if not NO_BLUETSTACKS:
        bluestacks_process = launch_bluestacks(wait_after_launch=config.BLUESTACKS_BOOT_WAIT)
        if bluestacks_process is None:
            print("[ERROR] BlueStacks launch failed. Exiting...")
            sys.exit(1)
    else:
        print("[INFO] Skipping BlueStacks launch (--no-bluestacks).")

    # Start adb server
    run_adb(["start-server"])

    device = wait_for_device(timeout=config.BLUESTACKS_CONNECT_TIMEOUT)
    if not device:
        print("[ERROR] No device detected. Exiting...")
        if bluestacks_process and bluestacks_process.poll() is None:
            bluestacks_process.terminate()
        sys.exit(1)

    # Launch game
    launch_subway_surfer(device)
    print("🚀 Starting Gesture Control (Subway Surf Game)...")

    try:
        gesture_loop(device, camera_index=CAMERA_INDEX)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed. Exiting gesture control...")
    finally:
        try:
            analytics.log_session()
        except Exception as e:
            print(f"[ANALYTICS-ERR] Could not save session log: {e}")

        # Ensure BlueStacks closed if we launched it
        if bluestacks_process and bluestacks_process.poll() is None:
            print("[INFO] Closing BlueStacks...")
            bluestacks_process.terminate()
            bluestacks_process.wait()

        print("[INFO] All resources released. Goodbye!")

if __name__ == "__main__":
    main()
