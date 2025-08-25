# utils.py

import cv2
import numpy as np
import time
import os

def preprocess_frame(frame, size=(64, 64)):
    """
    Preprocess a frame for ML model input:
    - Convert to grayscale
    - Resize
    - Normalize pixel values (0-1)
    - Flatten to 1D array
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    normalized = resized / 255.0
    flattened = normalized.flatten()
    return flattened


def create_dir_if_not_exists(path):
    """
    Create a directory if it does not already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def draw_text(frame, text, position=(50, 50), color=(0, 255, 0), scale=1, thickness=2):
    """
    Draw text on the given frame.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def get_timestamp():
    """
    Return current timestamp string.
    """
    return time.strftime("%Y%m%d_%H%M%S")


def smooth_predictions(predictions, window_size=5):
    """
    Smooth gesture predictions using a sliding window majority vote.
    Helps reduce jitter in real-time gesture recognition.
    """
    if len(predictions) < window_size:
        return predictions[-1]  # return latest if not enough history
    window = predictions[-window_size:]
    return max(set(window), key=window.count)
