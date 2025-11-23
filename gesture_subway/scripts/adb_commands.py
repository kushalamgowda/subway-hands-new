# adb_commands.py (patched)
import os
import subprocess
import time
import logging

# try to use central config if available
try:
    import config
    ADB_EXE = getattr(config, "ADB_PATH", r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe")
    BLUESTACKS_EXE = getattr(config, "BLUESTACKS_PATH", r"C:\Program Files\BlueStacks_nxt\HD-Player.exe")
except Exception:
    ADB_EXE = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
    BLUESTACKS_EXE = r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"

# logging setup (caller can reconfigure)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ------------------------------
# Helper: Run adb commands
# ------------------------------
def run_adb(args, timeout=10):
    """Run adb command using full ADB_EXE path. Returns stdout string ('' on error)."""
    cmd = [ADB_EXE] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logging.debug(f"ADB stderr: {result.stderr.strip()}")
            logging.error(f"ADB command failed (rc={result.returncode}): {' '.join(cmd)}")
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logging.error("ADB command timed out: %s", " ".join(cmd))
    except FileNotFoundError:
        logging.error("ADB executable not found at: %s", ADB_EXE)
    except Exception as e:
        logging.exception("Could not run adb command: %s", e)
    return ""

# ------------------------------
# Device detection
# ------------------------------
def get_default_device():
    """Return first connected device id, or None."""
    output = run_adb(["devices"])
    if not output:
        return None
    lines = output.splitlines()
    # skip header if present
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        # line example: "127.0.0.1:5555\tdevice"
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            return parts[0]
    return None

def connect_device(retries=3, delay=1):
    """Start adb server and return a connected device (retries)."""
    run_adb(["start-server"])
    for i in range(retries):
        device = get_default_device()
        if device:
            logging.info("Device connected: %s", device)
            return device
        logging.info("No device detected. Retrying in %ss (%d/%d)...", delay, i+1, retries)
        time.sleep(delay)
    logging.error("No device detected after retries. Start BlueStacks/emulator first.")
    return None

# ------------------------------
# Send commands to device
# ------------------------------
_ACTION_MAP = {
    "UP":    ["input", "swipe", "500", "1000", "500", "500", "200"],
    "JUMP":  ["input", "swipe", "500", "1000", "500", "500", "200"],
    "DOWN":  ["input", "swipe", "500", "1000", "500", "1500", "200"],
    "DUCK":  ["input", "swipe", "500", "1000", "500", "1500", "200"],
    "LEFT":  ["input", "swipe", "600", "1000", "200", "1000", "200"],
    "RIGHT": ["input", "swipe", "400", "1000", "800", "1000", "200"],
    "PAUSE": ["input", "tap", "1000", "150"],
    "PLAY":  ["input", "tap", "600", "1200"],
    # STOP intentionally does nothing
}

def send_command(action, device=None, dry_run=False):
    """
    Send a mapped input action to the device.
    Returns True on success (or dry_run), False on failure.
    """
    if not device:
        logging.error("No device connected!")
        return False

    action = action.upper()
    if action == "STOP":
        logging.info("STOP detected — no adb command sent.")
        return True

    if action not in _ACTION_MAP:
        logging.warning("Unknown action: %s", action)
        return False

    args = ["-s", device, "shell"] + _ACTION_MAP[action]

    logging.info("ADB CMD: %s %s", ADB_EXE, " ".join(args))
    if dry_run:
        logging.info("Dry-run: command not executed.")
        return True

    out = run_adb(args)
    if out == "":
        # run_adb already logged error if needed
        return False
    return True

# ------------------------------
# Launch BlueStacks
# ------------------------------
def launch_bluestacks(wait_after_launch=15):
    """Launch BlueStacks and wait. Returns subprocess.Popen object or None."""
    if not os.path.exists(BLUESTACKS_EXE):
        logging.error("BlueStacks not found at %s", BLUESTACKS_EXE)
        return None
    logging.info("Launching BlueStacks: %s", BLUESTACKS_EXE)
    try:
        proc = subprocess.Popen([BLUESTACKS_EXE])
        logging.info("Waiting %ds for BlueStacks to boot...", wait_after_launch)
        time.sleep(wait_after_launch)
        return proc
    except Exception as e:
        logging.exception("Failed to launch BlueStacks: %s", e)
        return None

# ------------------------------
# Launch Subway Surfer
# ------------------------------
def launch_subway_surfer(device, package="com.kiloo.subwaysurf", activity="com.kiloo.subwaysurf.SplashActivity"):
    """Launch Subway Surfer inside BlueStacks; returns True on success."""
    if not device:
        logging.error("No device specified to launch Subway Surfer.")
        return False
    cmd = ["-s", device, "shell", "am", "start", "-n", f"{package}/{activity}"]
    out = run_adb(cmd)
    if out == "":
        logging.error("Failed to launch Subway Surfer.")
        return False
    logging.info("Subway Surfer launch attempted.")
    return True
