import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
DATA_FILE = os.path.join(BASE_DIR, "data", "gesture_data.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "models", "gesture_model.pkl")
TEST_FILE = os.path.join(BASE_DIR, "data", "test.pkl")   # <-- Added
LOG_FILE = os.path.join(BASE_DIR, "logs", "analytics.log")

# Ensure folders exist
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# Gesture settings
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "stop"]
SAMPLES_PER_GESTURE = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ML model
N_ESTIMATORS = 100
MAX_DEPTH = None

# ADB main path
ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"

# ADB commands
ADB_COMMANDS = {
    "swipe_left": rf'"{ADB_PATH}" shell input keyevent 21',
    "swipe_right": rf'"{ADB_PATH}" shell input keyevent 22',
    "swipe_up": rf'"{ADB_PATH}" shell input keyevent 19',
    "swipe_down": rf'"{ADB_PATH}" shell input keyevent 20',
    "stop": rf'"{ADB_PATH}" shell input keyevent 66'
}

# Analytics
ENABLE_LOGGING = True
LOGGING_LEVEL = "INFO"

# BlueStacks settings
BLUESTACKS_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"
BLUESTACKS_BOOT_WAIT = 25
BLUESTACKS_CONNECT_TIMEOUT = 120
BLUESTACKS_POLL_INTERVAL = 3

# Subway Surfer
SUBWAY_SURFER_PACKAGE = "com.kiloo.subwaysurf"
SUBWAY_SURFER_ACTIVITY = "com.kiloo.subwaysurf.SplashActivity"
