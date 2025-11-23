# utils.py (cleaned – no volume controls)
import numpy as np


def extract_features(landmarks, normalize=True):
    """
    Convert mediapipe landmarks to a (1, 42) numpy array of float32.
    - landmarks: mediapipe hand landmarks or list/array of (x, y)
    - normalize: center on wrist & scale (matches training)
    Returns: np.ndarray shape (1, 42)
    """
    # Try mediapipe format: landmarks.landmark
    if hasattr(landmarks, "landmark"):
        arr = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=np.float32)
    else:
        # list/array case
        arr = np.asarray(landmarks, dtype=np.float32)
        if arr.ndim == 1 and arr.size == 42:
            arr = arr.reshape(21, 2)
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            arr = arr[:, :2]
        else:
            raise ValueError(f"Invalid landmarks shape: {arr.shape}")

    if arr.shape[0] != 21:
        raise ValueError(f"Expected 21 landmarks, got {arr.shape[0]}")

    if normalize:
        center = arr[0].copy()        # wrist index 0
        arr = arr - center
        dists = np.linalg.norm(arr, axis=1)
        max_dist = np.max(dists)
        if max_dist > 1e-6:
            arr = arr / max_dist

    flat = arr.flatten().astype(np.float32).reshape(1, -1)
    return flat
