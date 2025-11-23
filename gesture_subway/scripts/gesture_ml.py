# gesture_ml.py (patched)
import os
import pickle
import logging
from typing import List, Tuple, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

# ------------------------------
# Logging
# ------------------------------
LOG_LEVEL = getattr(config, "LOGGING_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="[%(levelname)s] %(message)s")

# ------------------------------
# Utility: Feature handling
# ------------------------------
def extract_features_from_landmarks(landmarks: List[Tuple[float, float]]) -> np.ndarray:
    """
    Convert list/array of (x,y) landmarks to a flat numpy feature vector.
    Accepts:
      - list/tuple of (x,y)
      - numpy array shape (N,2) or (2N,)
    Returns a 1D numpy array of floats.
    """
    if landmarks is None:
        raise ValueError("landmarks is None")

    # If numpy array
    if isinstance(landmarks, np.ndarray):
        arr = landmarks
        if arr.ndim == 2 and arr.shape[1] >= 2:
            flat = arr[:, :2].flatten()
        elif arr.ndim == 1:
            flat = arr
        else:
            raise ValueError(f"Invalid numpy landmark shape: {arr.shape}")
        return np.asarray(flat, dtype=float)

    # Otherwise expect list/iterable of (x,y)
    flat = []
    for lm in landmarks:
        if not (isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 2):
            raise ValueError(f"Invalid landmark format: {lm}")
        flat.append(float(lm[0]))
        flat.append(float(lm[1]))
    return np.asarray(flat, dtype=float)


# ------------------------------
# Save / Load Model
# ------------------------------
def save_model(model, filename: str = None) -> None:
    """Save model (pipeline) to disk using pickle (atomic-ish)."""
    filename = filename if filename is not None else config.MODEL_FILE
    os.makedirs(os.path.dirname(os.path.abspath(filename)) or ".", exist_ok=True)
    tmp = filename + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, filename)
        logging.info("Model saved to %s", filename)
    except Exception as e:
        logging.exception("Failed to save model to %s: %s", filename, e)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def load_model(filename: str = None):
    """Load model (pipeline) from disk. Returns object or raises."""
    filename = filename if filename is not None else config.MODEL_FILE
    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded from %s", filename)
        return model
    except FileNotFoundError:
        logging.error("Model file not found at %s", filename)
        raise
    except Exception as e:
        logging.exception("Failed to load model from %s: %s", filename, e)
        raise RuntimeError(f"Failed to load model from {filename}: {e}")


# ------------------------------
# Train Model
# ------------------------------
def train_model(X: np.ndarray, y: np.ndarray,
                test_size: Optional[float] = None,
                random_state: Optional[int] = None,
                n_estimators: Optional[int] = None) -> Pipeline:
    """
    Trains a RandomForest inside a Pipeline (scaling + RF).

    Behavior:
      - If test_size is None: uses config.TEST_SIZE
      - If test_size == 0: do NOT perform an internal split; train on X,y directly
      - Otherwise: perform train_test_split(X, y, test_size=test_size)

    X: 2D numpy array (n_samples, n_features)
    y: 1D array-like labels
    Returns: trained sklearn Pipeline
    """
    if X is None or y is None:
        raise ValueError("X and y must be provided")
    if len(X) == 0:
        raise ValueError("Empty X provided")

    # resolve defaults from config
    test_size = test_size if test_size is not None else getattr(config, "TEST_SIZE", 0.2)
    random_state = random_state if random_state is not None else getattr(config, "RANDOM_STATE", 42)
    n_estimators = n_estimators if n_estimators is not None else getattr(config, "N_ESTIMATORS", 100)

    logging.info("train_model called with test_size=%s, random_state=%s, n_estimators=%s", test_size, random_state, n_estimators)

    # If caller requested no internal split (test_size == 0), train on X,y directly
    if float(test_size) == 0.0:
        logging.info("test_size==0. Training on entire provided dataset (no internal split).")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        logging.info("Splitting data (test_size=%s, random_state=%s)", test_size, random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logging.info("Building pipeline (StandardScaler -> RandomForest n_estimators=%s)...", n_estimators)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
    ])

    logging.info("Training RandomForest model...")
    pipeline.fit(X_train, y_train)

    # If we have an internal test set, evaluate and log
    if X_test is not None and y_test is not None:
        try:
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            logging.info("Validation Accuracy: %.4f", acc)
            logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))
        except Exception as e:
            logging.exception("Validation evaluation failed: %s", e)

    return pipeline


# ------------------------------
# Predict helper
# ------------------------------
def predict_from_landmarks(model, landmarks: List[Tuple[float, float]], expected_feature_len: Optional[int] = None):
    """
    Given a loaded model (pipeline) and landmarks list, returns predicted label (and probs if available).
    expected_feature_len: optional integer (e.g., 42) to validate incoming landmark length.
    Returns: (pred_label, probs_or_None)
    """
    features = extract_features_from_landmarks(landmarks)
    if expected_feature_len is not None and features.size != expected_feature_len:
        raise ValueError(f"Feature length mismatch: expected {expected_feature_len}, got {features.size}")

    X = features.reshape(1, -1)
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        logging.exception("Model prediction failed: %s", e)
        raise

    probs = None
    # Pipeline has predict_proba if the final estimator supports it
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)[0]
        except Exception:
            probs = None

    # Ensure predicted label is serializable/string for downstream modules
    if isinstance(pred, (np.integer, int, float)):
        pred_label = str(pred)
    else:
        pred_label = pred

    return pred_label, probs


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    logging.info("Running standalone training test...")

    # Dummy dataset (100 samples, 42 features -> 21 landmarks)
    X_dummy = np.random.rand(100, 42)
    y_dummy = np.random.choice(["swipe_left", "swipe_right", "swipe_up"], size=100)

    pipeline = train_model(X_dummy, y_dummy,
                           test_size=getattr(config, "TEST_SIZE", 0.2),
                           random_state=getattr(config, "RANDOM_STATE", 42),
                           n_estimators=getattr(config, "N_ESTIMATORS", 100))
    save_model(pipeline)

    # test a dummy landmark input (21 landmarks -> 42 features)
    sample_landmarks = [(float(np.random.rand()), float(np.random.rand())) for _ in range(21)]
    pred_label, pred_probs = predict_from_landmarks(pipeline, sample_landmarks, expected_feature_len=42)
    logging.info("Sample prediction: %s, probs: %s", pred_label, pred_probs)
