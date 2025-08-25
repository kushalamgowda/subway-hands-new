# gesture_ml.py

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import config


# ------------------------------
# Save Model
# ------------------------------
def save_model(model, filename=config.MODEL_FILE):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved to {filename}")


# ------------------------------
# Load Model
# ------------------------------
def load_model(filename=config.MODEL_FILE):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded from {filename}")
    return model


# ------------------------------
# Train Model
# ------------------------------
def train_model(X, y):
    """
    Trains a RandomForest model on gesture dataset.
    X: features (landmarks)
    y: labels (gestures)
    """
    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model


# ------------------------------
# Extract Features from Landmarks
# ------------------------------
def extract_features_from_landmarks(landmarks):
    """
    landmarks: list of (x, y) points from MediaPipe
    Returns: numpy feature vector
    """
    features = []
    for lm in landmarks:
        features.append(lm[0])  # x
        features.append(lm[1])  # y
    return np.array(features)


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # Example dummy data (for testing only)
    # In real usage, use scripts/train_model.py
    print("[DEBUG] Running standalone training test...")

    # Dummy dataset (100 samples, 42 features, 3 classes)
    X_dummy = np.random.rand(100, 42)
    y_dummy = np.random.choice(["LEFT", "RIGHT", "JUMP"], size=100)

    model = train_model(X_dummy, y_dummy)
    save_model(model)
