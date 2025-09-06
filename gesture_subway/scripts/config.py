# config.py
# ------------------------------
# Project Configuration
# ------------------------------

import os

# Base directory (gesture_subway/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
DATA_FILE = os.path.join(BASE_DIR, "data", "gesture_data.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "models", "gesture_model.pkl")
LOG_FILE = os.path.join(BASE_DIR, "logs", "analytics.log")

# Make sure folders exist
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# Training configuration
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "stop"]
SAMPLES_PER_GESTURE = 100   # How many samples to collect for each gesture
TEST_SIZE = 0.2             # Train/Test split
RANDOM_STATE = 42           # For reproducibility

# ML Model settings
N_ESTIMATORS = 100          # Number of trees in RandomForest
MAX_DEPTH = None            # No max depth (let it expand)

# ADB (Android Debug Bridge) commands mapping
ADB_COMMANDS = {
    "swipe_left": "adb shell input keyevent 21",   # LEFT arrow
    "swipe_right": "adb shell input keyevent 22",  # RIGHT arrow
    "swipe_up": "adb shell input keyevent 19",     # UP arrow
    "swipe_down": "adb shell input keyevent 20",   # DOWN arrow
    "stop": "adb shell input keyevent 66"          # ENTER/OK
}

# Analytics Settings
ENABLE_LOGGING = True
LOGGING_LEVEL = "INFO"
