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

# project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config
from analytics import AnalyticsLogger

# ------------------------------
# CLI flags
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
# BlueStacks / ADB
# ------------------------------
BLUESTACKS_EXE = config.BLUESTACKS_PATH
ADB_EXE = config.ADB_PATH
SUBWAY_SURFER_PACKAGE = config.SUBWAY_SURFER_PACKAGE
SUBWAY_SURFER_ACTIVITY = config.SUBWAY_SURFER_ACTIVITY

# ------------------------------
# Load Model
# ------------------------------
try:
    with open(config.MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded from {config.MODEL_FILE}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    raise

# ------------------------------
# MediaPipe setup
# ------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

WINDOW_SIZE = getattr(config, "WINDOW_SIZE", 7)
CONF_THRESHOLD = getattr(config, "CONF_THRESHOLD", 0.75)
ACTION_COOLDOWN = getattr(config, "ACTION_COOLDOWN", 0.5)

# ------------------------------
# ADB helper functions
# ------------------------------
def run_adb(args):
    cmd = [ADB_EXE] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"[ADB-EXC] {e}")
        return ""

def get_device():
    output = run_adb(["devices"])
    lines = output.splitlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1] == "device":
            return parts[0]
    return None

def wait_for_device(timeout=120, poll=3):
    start = time.time()
    while time.time() - start < timeout:
        d = get_device()
        if d:
            return d
        print(f"[INFO] Waiting for device...")
        time.sleep(poll)
    return None

def launch_bluestacks(wait_after_launch=25):
    if not os.path.exists(BLUESTACKS_EXE):
        print(f"[ERROR] BlueStacks not found: {BLUESTACKS_EXE}")
        return None
    print("[INFO] Launching BlueStacks...")
    try:
        p = subprocess.Popen([BLUESTACKS_EXE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(wait_after_launch)
        return p
    except Exception as e:
        print(f"[ERROR] Could not launch BlueStacks: {e}")
        return None

def launch_subway_surfer(device):
    try:
        subprocess.run([
            ADB_EXE, "-s", device, "shell",
            "monkey", "-p", SUBWAY_SURFER_PACKAGE,
            "-c", "android.intent.category.LAUNCHER", "1"
        ], check=True)
        print("[INFO] Subway Surfer launched!")
    except Exception as e:
        print(f"[ERROR] Could not launch game: {e}")

# ------------------------------
# Feature extraction
# ------------------------------
def normalize_landmarks(landmarks):
    arr = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=float)
    center = arr[0]
    arr -= center
    d = np.linalg.norm(arr, axis=1)
    max_d = np.max(d)
    if max_d > 1e-6:
        arr /= max_d
    return arr.flatten()

def extract_features(landmarks):
    return normalize_landmarks(landmarks).reshape(1, -1)

# ------------------------------
# Gesture → ADB mapping (NO volume)
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
        return [
            prefix + ["input", "tap", "1000", "150"],
            prefix + ["input", "tap", "980", "120"],
            prefix + ["input", "tap", "1050", "170"],
            prefix + ["input", "tap", "1100", "80"],
            prefix + ["input", "tap", "70", "60"],
        ]

    elif gesture == "start":
        return [prefix + ["input", "tap", "600", "1200"]]

    return None
