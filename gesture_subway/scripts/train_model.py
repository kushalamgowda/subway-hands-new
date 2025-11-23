# train_model.py

import argparse
import os
import pickle
import sys
from pprint import pprint

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# make project root importable (assumes this file is in a subfolder of project)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import gesture_ml
import config


def load_dataset_from_collect_file(path):
    """
    Load dataset saved by collect_data.py which stores a dict:
      { gesture_name: np.array(shape=(n_samples, 42)), ... }
    Returns (X, y) as numpy arrays.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)

    X_list = []
    y_list = []

    # raw might contain arrays or lists; handle both
    for gesture, arr in raw.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            # single sample flattened — convert to (1, n_features)
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            continue
        X_list.append(arr)
        y_list += [gesture] * arr.shape[0]

    if len(X_list) == 0:
        raise ValueError("No data found in dataset file.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=object)
    return X, y


def main(args):
    # 1) Load data
    print(f"[INFO] Loading dataset from {config.DATA_FILE} ...")
    try:
        X, y = load_dataset_from_collect_file(config.DATA_FILE)
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found at {config.DATA_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    print(f"[INFO] Dataset loaded: X.shape={X.shape}, y.shape={y.shape}")
    # show class counts
    unique, counts = np.unique(y, return_counts=True)
    print("[INFO] Class distribution:")
    pprint(dict(zip(unique, counts)))

    # 2) Decide whether stratify is safe
    stratify = y if np.all(counts >= 2) else None
    if stratify is None:
        print("[WARN] Some classes have fewer than 2 samples — skipping stratified split.")

    # 3) Split into train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    print(f"[INFO] Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 4) Train via your gesture_ml.train_model (which returns a pipeline)
    print("[INFO] Training model (using gesture_ml.train_model)...")
    pipeline = gesture_ml.train_model(X_train, y_train, test_size=0.0, random_state=args.seed, n_estimators=args.n_estimators)
    # Note: we passed test_size=0.0 so gesture_ml.train_model doesn't split again;
    # If gesture_ml.train_model complains about test_size=0.0, fallback to training directly below.

    # 5) Save model
    try:
        gesture_ml.save_model(pipeline, filename=config.MODEL_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")

    # 6) Evaluate on held-out test set
    print("[INFO] Evaluating on held-out test set...")
    try:
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        print(f"[ERROR] Model prediction failed: {e}")
        sys.exit(1)

    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 7) Save test set for later inspection
    try:
        test_data = {"features": X_test, "labels": y_test}
        with open(config.TEST_FILE, "wb") as f:
            pickle.dump(test_data, f)
        print(f"[INFO] Saved test set to {config.TEST_FILE}")
    except Exception as e:
        print(f"[WARN] Could not save test set: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture model from collected data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for final hold-out test set")
    parser.add_argument("--n-estimators", type=int, default=200, help="RandomForest n_estimators")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
