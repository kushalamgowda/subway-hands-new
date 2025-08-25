# scripts/adb_commands.py
import os
import time

# Optional: default device IP for BlueStacks
DEVICE_IP = "127.0.0.1:5555"

def connect_device(device_ip=DEVICE_IP):
    """Connects to the Android device using adb."""
    print(f"[INFO] Connecting to {device_ip} ...")
    os.system(f"adb connect {device_ip}")
    time.sleep(1)

def send_command(action):
    """
    Send swipe/tap commands to the connected device (BlueStacks / Android Emulator).
    
    Parameters:
    action (str): One of 'UP', 'DOWN', 'LEFT', 'RIGHT', 'JUMP', 'DUCK', 'PAUSE', 'PLAY'.
    """
    if action in ["UP", "JUMP"]:
        # Swipe from bottom to top
        os.system("adb shell input swipe 500 1000 500 500")
        print("[ACTION] Jump / Swipe Up")
    elif action in ["DOWN", "DUCK"]:
        # Swipe from top to bottom
        os.system("adb shell input swipe 500 1000 500 1500")
        print("[ACTION] Duck / Swipe Down")
    elif action == "LEFT":
        # Swipe from right to left
        os.system("adb shell input swipe 600 1000 200 1000")
        print("[ACTION] Swipe Left")
    elif action == "RIGHT":
        # Swipe from left to right
        os.system("adb shell input swipe 400 1000 800 1000")
        print("[ACTION] Swipe Right")
    elif action == "PAUSE":
        os.system("adb shell input tap 1000 150")  # Adjust coordinates
        print("[ACTION] Pause")
    elif action == "PLAY":
        os.system("adb shell input tap 600 1200")  # Adjust coordinates
        print("[ACTION] Play")
    else:
        print(f"[WARNING] Unknown action: {action}")

    # Small delay to avoid sending commands too quickly
    time.sleep(0.1)
