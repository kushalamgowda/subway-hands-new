# scripts/train_model.py

import pickle
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import helper modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_ml
import config


def train_model():
    # Load dataset
    print("[INFO] Loading dataset...")
    with open(config.DATA_FILE, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["features"])
    y = np.array(data["labels"])

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

    # Evaluate
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Test Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_model()
