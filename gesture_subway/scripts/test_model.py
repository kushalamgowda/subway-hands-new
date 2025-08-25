# scripts/test_model.py

import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Import our ML helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_ml
import config


def test_model():
    # Load dataset
    print("[INFO] Loading dataset...")
    with open(config.DATA_FILE, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["features"])
    y = np.array(data["labels"])

    # Load trained model
    model = gesture_ml.load_model(config.MODEL_FILE)

    # Predict on dataset
    print("[INFO] Testing model...")
    y_pred = model.predict(X)

    # Evaluate performance
    acc = accuracy_score(y, y_pred)
    print(f"[INFO] Accuracy on dataset: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y, y_pred))


if __name__ == "__main__":
    test_model()
