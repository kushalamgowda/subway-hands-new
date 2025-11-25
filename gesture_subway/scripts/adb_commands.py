import os
import subprocess
import time

# ------------------------------
# Paths
# ------------------------------
ADB_EXE = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
BLUESTACKS_EXE = r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"

# ------------------------------
# Helper: Run adb commands
# ------------------------------
def run_adb(args):
    """Run adb command using full path"""
    cmd = [ADB_EXE] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"[ERROR] Could not run adb command: {e}")
        return None

# ------------------------------
# Device detection
# ------------------------------
def get_default_device():
    """Return first connected device"""
    output = run_adb(["devices"])
    lines = output.splitlines()
    for line in lines[1:]:
        if line.strip() and "device" in line:
            return line.split()[0]
    return None

# ------------------------------
# Send commands to device
# ------------------------------
def send_command(action, device=None):
    if not device:
        print("[ERROR] No device connected!")
        return
    
    prefix = ["-s", device, "shell"]

    if action in ["UP", "JUMP"]:
        run_adb(prefix + ["input", "swipe", "500", "1000", "500", "500"])
    elif action in ["DOWN", "DUCK"]:
        run_adb(prefix + ["input", "swipe", "500", "1000", "500", "1500"])
    elif action == "LEFT":
        run_adb(prefix + ["input", "swipe", "600", "1000", "200", "1000"])
    elif action == "RIGHT":
        run_adb(prefix + ["input", "swipe", "400", "1000", "800", "1000"])
    elif action == "PAUSE":
        run_adb(prefix + ["input", "tap", "1000", "150"])
    elif action == "PLAY":
        run_adb(prefix + ["input", "tap", "600", "1200"])
    elif action == "DOUBLE_TAP":
        print("[ACTION] Double Tap → Activating Surfboard 🏄")

        run_adb(prefix + ["input", "tap", "500", "1000"])
        time.sleep(0.15)  
        run_adb(prefix + ["input", "tap", "500", "1000"])
    elif action == "STOP":
        print("[ACTION] Stop detected – no command sent")
    else:
        print(f"[WARNING] Unknown action: {action}")


# ------------------------------
# Launch BlueStacks
# ------------------------------
def launch_bluestacks(wait_after_launch=15):
    if not os.path.exists(BLUESTACKS_EXE):
        print(f"[ERROR] BlueStacks not found at {BLUESTACKS_EXE}")
        return
    print(f"[INFO] Launching BlueStacks...")
    subprocess.Popen([BLUESTACKS_EXE])
    print(f"[INFO] Waiting {wait_after_launch}s for BlueStacks to boot...")
    time.sleep(wait_after_launch)

# ------------------------------
# Launch Subway Surfer
# ------------------------------
def launch_subway_surfer(device, package="com.kiloo.subwaysurf", activity="com.kiloo.subwaysurf.SplashActivity"):
    """Launch Subway Surfer inside BlueStacks"""
    try:
        run_adb(["-s", device, "shell", "am", "start", "-n", f"{package}/{activity}"])
        print("[INFO] Subway Surfer launched!")
    except Exception as e:
        print(f"[ERROR] Could not launch Subway Surfer: {e}")
