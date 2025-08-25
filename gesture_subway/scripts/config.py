# config.py

# ------------------------------
# Project Configuration
# ------------------------------

# File paths
DATA_FILE = "data/gesture_data.pkl"
MODEL_FILE = "models/gesture_model.pkl"
LOG_FILE = "logs/analytics.log"

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
