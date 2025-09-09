# scripts/test_model.py

import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# Import our ML helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_ml
import config


def test_model():
    # Load test dataset
    print("[INFO] Loading test dataset...")
    try:
        with open(config.TEST_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Test dataset not found at {config.TEST_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load test dataset: {e}")
        sys.exit(1)

    X = np.array(data["features"])
    y = np.array(data["labels"])

    # Load trained model
    print("[INFO] Loading trained model...")
    try:
        model = gesture_ml.load_model(config.MODEL_FILE)
    except FileNotFoundError:
        print(f"[ERROR] Model file not found at {config.MODEL_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Predict on dataset
    print("[INFO] Testing model...")
    y_pred = model.predict(X)

    # Evaluate performance
    acc = accuracy_score(y, y_pred)
    print(f"[INFO] Accuracy on test set: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))


if __name__ == "__main__":
    test_model()
