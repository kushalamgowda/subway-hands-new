import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import subprocess
import os
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

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

# ===========================================================
# 🔹 THREAD-SAFE CAMERA GRABBER (for smoother frame updates)
# ===========================================================
class CameraGrabber(threading.Thread):
    def __init__(self, src=0, width=None, height=None):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame
        self.cap.release()

    def read(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False


# ===========================================================
# 🔹 NON-BLOCKING ADB COMMANDS
# ===========================================================
_executor = ThreadPoolExecutor(max_workers=2)

def _run_subprocess_cmd(cmd_list):
    try:
        subprocess.run(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
    except Exception:
        pass

def send_command_async(cmd_list):
    _executor.submit(_run_subprocess_cmd, cmd_list)


# ===========================================================
# 🔹 DEBOUNCE FOR GESTURE ACTIONS
# ===========================================================
class ActionDebouncer:
    def __init__(self, min_interval=0.15):
        self.min_interval = min_interval
        self._last_time = {}

    def allow(self, action_name):
        now = time.time()
        last = self._last_time.get(action_name, 0)
        if now - last >= self.min_interval:
            self._last_time[action_name] = now
            return True
        return False


# ===========================================================
# 🔹 ADB HELPER FUNCTIONS
# ===========================================================
def run_adb(args):
    cmd = [ADB_EXE] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def get_device():
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
        return None
    print(f"[INFO] Launching BlueStacks...")
    process = subprocess.Popen([BLUESTACKS_EXE])
    print(f"[INFO] Waiting {wait_after_launch}s for BlueStacks to boot...")
    time.sleep(wait_after_launch)
    return process

def launch_subway_surfer(device):
    try:
        subprocess.run([ADB_EXE, "-s", device, "shell",
                        "monkey", "-p", SUBWAY_SURFER_PACKAGE,
                        "-c", "android.intent.category.LAUNCHER", "1"],
                       check=True)
        print("[INFO] Subway Surfer launched!")
    except Exception as e:
        print(f"[ERROR] Could not launch Subway Surfer: {e}")


# ===========================================================
# 🔹 OPTIMIZED GESTURE LOOP
# ===========================================================
def gesture_loop_optimized(device_id, model, hands, show_preview=False):
    cam = CameraGrabber(src=0, width=480, height=320)
    cam.start()

    debouncer = ActionDebouncer(min_interval=0.12)
    prev_gesture = None

    print("[INFO] Optimized gesture loop started. Press Ctrl+C to exit.")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            small = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            gesture = None

            if result.multi_hand_landmarks:
                for lm_set in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(small, lm_set, mp_hands.HAND_CONNECTIONS)
                lm = result.multi_hand_landmarks[0].landmark
                feat = np.array([[p.x for p in lm] + [p.y for p in lm]])
                try:
                    pred = model.predict(feat)
                    gesture = pred[0]
                except Exception:
                    gesture = None

            if gesture is None:
                prev_gesture = None
                time.sleep(0.002)
                continue

            # ACTION MAPPING (use debounce + async ADB)
            if gesture != prev_gesture:
                if gesture == "swipe_left" and debouncer.allow("LEFT"):
                    send_command_async([ADB_EXE, "-s", device_id, "shell", "input", "swipe", "600", "1000", "200", "1000"])
                elif gesture == "swipe_right" and debouncer.allow("RIGHT"):
                    send_command_async([ADB_EXE, "-s", device_id, "shell", "input", "swipe", "400", "1000", "800", "1000"])
                elif gesture == "swipe_up" and debouncer.allow("UP"):
                    send_command_async([ADB_EXE, "-s", device_id, "shell", "input", "swipe", "500", "1000", "500", "500"])
                elif gesture == "swipe_down" and debouncer.allow("DOWN"):
                    send_command_async([ADB_EXE, "-s", device_id, "shell", "input", "swipe", "500", "1000", "500", "1500"])
                elif gesture == "volume_up" and debouncer.allow("VOL_UP"):
                    volume_up()
                elif gesture == "volume_down" and debouncer.allow("VOL_DOWN"):
                    volume_down()
                elif gesture == "mute" and debouncer.allow("MUTE"):
                    mute_toggle()
                prev_gesture = gesture

            if show_preview:
                cv2.putText(small, f"Gesture: {gesture}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Gesture Control", small)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Exiting gesture loop...")
    finally:
        cam.stop()
        _executor.shutdown(wait=False)
        if show_preview:
            cv2.destroyAllWindows()
        print("[INFO] All resources released.")


# ===========================================================
# 🔹 MAIN ENTRY POINT
# ===========================================================
if __name__ == "__main__":
    bluestacks_process = launch_bluestacks(wait_after_launch=config.BLUESTACKS_BOOT_WAIT)
    if bluestacks_process is None:
        print("[ERROR] BlueStacks launch failed. Exiting...")
        sys.exit(1)

    run_adb(["start-server"])

    device = wait_for_device(timeout=config.BLUESTACKS_CONNECT_TIMEOUT)
    if not device:
        print("[ERROR] No device detected. Exiting...")
        bluestacks_process.terminate()
        sys.exit(1)

    launch_subway_surfer(device)
    print("🚀 Starting Gesture Control (Optimized)...")

    try:
        gesture_loop_optimized(device, model, hands, show_preview=True)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed. Stopping...")
    finally:
        if bluestacks_process.poll() is None:
            print("[INFO] Closing BlueStacks...")
            bluestacks_process.terminate()
            bluestacks_process.wait()
        print("[INFO] Goodbye 👋")

