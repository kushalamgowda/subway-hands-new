# utils.py
import cv2
import mediapipe as mp
import numpy as np

# For system volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# ------------------------------
# Feature Extraction (Hand Landmarks → Features)
# ------------------------------
def extract_features(landmarks):
    """Extract (x,y) coordinates of 21 hand landmarks into a feature vector"""
    features = []
    for lm in landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
    return np.array(features).reshape(1, -1)


# ------------------------------
# Volume Control Functions
# ------------------------------
def get_volume_interface():
    """Initialize system volume control"""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )
    return cast(interface, POINTER(IAudioEndpointVolume))


def volume_up(step=0.1):
    """Increase system volume"""
    volume = get_volume_interface()
    current = volume.GetMasterVolumeLevelScalar()
    new_vol = min(current + step, 1.0)
    volume.SetMasterVolumeLevelScalar(new_vol, None)
    print(f"[VOLUME] Increased to {round(new_vol*100)}%")


def volume_down(step=0.1):
    """Decrease system volume"""
    volume = get_volume_interface()
    current = volume.GetMasterVolumeLevelScalar()
    new_vol = max(current - step, 0.0)
    volume.SetMasterVolumeLevelScalar(new_vol, None)
    print(f"[VOLUME] Decreased to {round(new_vol*100)}%")


def mute_toggle():
    """Toggle mute/unmute"""
    volume = get_volume_interface()
    mute_state = volume.GetMute()
    volume.SetMute(not mute_state, None)
    print("[VOLUME] Muted" if not mute_state else "[VOLUME] Unmuted")
