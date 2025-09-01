# scripts/adb_commands.py
import os
import time
import subprocess
import subprocess

# Full path to HD-Adb.exe
ADB_PATH = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"

def run_adb_command(args):
    try:
        result = subprocess.run([ADB_PATH] + args, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        print(f"[ERROR] Could not run adb command: {e}")
        return None

def get_default_device():
    """Detects the first connected device/emulator."""
    try:
        result = subprocess.check_output(["adb", "devices"], text=True)
        lines = result.strip().split("\n")[1:]  # Skip "List of devices attached"
        for line in lines:
            if line.strip() and "device" in line:
                return line.split()[0]  # Return device ID (e.g., emulator-5554)
    except Exception as e:
        print(f"[ERROR] Could not detect device: {e}")
    return None

def connect_device():
    devices = run_adb_command(["devices"])
    if devices and "emulator-5554" in devices:
        print("[INFO] Device connected:", devices)
        return True
    else:
        print("[ERROR] No device detected. Please start BlueStacks first.")
        return False

def send_command(action, device=None):
    """Send swipe/tap commands to the connected device."""
    if not device:
        print("[ERROR] No device connected!")
        return
    
    prefix = f"adb -s {device} shell"
    
    if action in ["UP", "JUMP"]:
        os.system(f"{prefix} input swipe 500 1000 500 500")
        print("[ACTION] Jump / Swipe Up")
    elif action in ["DOWN", "DUCK"]:
        os.system(f"{prefix} input swipe 500 1000 500 1500")
        print("[ACTION] Duck / Swipe Down")
    elif action == "LEFT":
        os.system(f"{prefix} input swipe 600 1000 200 1000")
        print("[ACTION] Swipe Left")
    elif action == "RIGHT":
        os.system(f"{prefix} input swipe 400 1000 800 1000")
        print("[ACTION] Swipe Right")
    elif action == "PAUSE":
        os.system(f"{prefix} input tap 1000 150")
        print("[ACTION] Pause")
    elif action == "PLAY":
        os.system(f"{prefix} input tap 600 1200")
        print("[ACTION] Play")
    elif action == "STOP":
        print("[ACTION] Stop detected – no command sent")
    else:
        print(f"[WARNING] Unknown action: {action}")
    
    time.sleep(0.1)
