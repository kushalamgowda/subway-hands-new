# scripts/train_model.py

import pickle
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import helper modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_ml
import config


def train_model():
    # Load dataset
    print("[INFO] Loading dataset...")
    try:
        with open(config.DATA_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found at {config.DATA_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    X, y = [], []

    # ✅ Collect features + labels safely
    for gesture_name, samples in data.items():
        for s in samples:
            arr = np.array(s).ravel()   # flatten into 1D vector

            # ✅ Skip invalid samples (hand not detected properly)
            if arr.shape[0] != 42:  
                continue  

            X.append(arr)
            y.append(gesture_name)

    # ✅ Convert safely to arrays
    X = np.array(X)
    y = np.array(y)

    print("[INFO] Final dataset shape:", X.shape, y.shape)
    print("[INFO] Unique labels:", set(y))

    # Split into train/test sets
    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("[INFO] Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    gesture_ml.save_model(model, config.MODEL_FILE)

    # Save test data separately
    test_data = {"features": X_test, "labels": y_test}
    with open(config.TEST_FILE, "wb") as f:
        pickle.dump(test_data, f)

    # Evaluate on test set
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Test Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train_model()
