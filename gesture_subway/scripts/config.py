import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
DATA_FILE = os.path.join(BASE_DIR, "data", "gesture_data.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "models", "gesture_model.pkl")
TEST_FILE = os.path.join(BASE_DIR, "data", "test.pkl")
LOG_FILE = os.path.join(BASE_DIR, "logs", "analytics.log")

# Ensure folders exist
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# Gesture settings
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "stop", "start"]
SAMPLES_PER_GESTURE = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ML model settings
N_ESTIMATORS = 100
MAX_DEPTH = None

# ADB main path
ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"

# --------------------------------------------
# ADB COMMANDS — list templates (recommended)
# --------------------------------------------
ADB_COMMANDS_LIST = {
    # Launch Subway Surfer
    "launch": [
        ADB_PATH, "-s", "{device}", "shell", "am", "start",
        "-n", "{package}/{activity}"
    ],

    # Game controls (swipes)
    "swipe_left":  [ADB_PATH, "-s", "{device}", "shell", "input", "swipe", "600", "1000", "200", "1000"],
    "swipe_right": [ADB_PATH, "-s", "{device}", "shell", "input", "swipe", "400", "1000", "800", "1000"],
    "swipe_up":    [ADB_PATH, "-s", "{device}", "shell", "input", "swipe", "500", "1000", "500", "500"],
    "swipe_down":  [ADB_PATH, "-s", "{device}", "shell", "input", "swipe", "500", "1000", "500", "1500"],

    # Pause the game (mapped to gesture: stop)
    "pause":       [ADB_PATH, "-s", "{device}", "shell", "input", "tap", "1000", "150"],

    # Resume the game (mapped to gesture: start)
    "resume":      [ADB_PATH, "-s", "{device}", "shell", "input", "tap", "600", "1200"],

    # Optional: send HOME button
    "home":        [ADB_PATH, "-s", "{device}", "shell", "input", "keyevent", "3"]
}

# --------------------------------------------
# Optional: string-based templates (shell=True)
# --------------------------------------------
ADB_COMMANDS = {
    "launch":  rf'{ADB_PATH} -s {{device}} shell am start -n {{package}}/{{activity}}',

    "swipe_left":  rf'{ADB_PATH} -s {{device}} shell input swipe 600 1000 200 1000',
    "swipe_right": rf'{ADB_PATH} -s {{device}} shell input swipe 400 1000 800 1000',
    "swipe_up":    rf'{ADB_PATH} -s {{device}} shell input swipe 500 1000 500 500',
    "swipe_down":  rf'{ADB_PATH} -s {{device}} shell input swipe 500 1000 500 1500',

    "pause":  rf'{ADB_PATH} -s {{device}} shell input tap 1000 150',
    "resume": rf'{ADB_PATH} -s {{device}} shell input tap 600 1200',

    "home":   rf'{ADB_PATH} -s {{device}} shell input keyevent 3'
}

# Logging
ENABLE_LOGGING = True
LOGGING_LEVEL = "INFO"

# BlueStacks settings
BLUESTACKS_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"
BLUESTACKS_BOOT_WAIT = 25
BLUESTACKS_CONNECT_TIMEOUT = 120
BLUESTACKS_POLL_INTERVAL = 3

# Subway Surfer (for launch)
SUBWAY_SURFER_PACKAGE = "com.kiloo.subwaysurf"
SUBWAY_SURFER_ACTIVITY = "com.kiloo.subwaysurf.SplashActivity"
