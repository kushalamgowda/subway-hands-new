# test_model.py (patched)
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
import json

# Import our ML helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_ml
import config


def safe_load_pickle(path):
    print(f"[INFO] Loading: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def pretty_print_confusion(cm, classes):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "     " + " ".join([f"{c:>8}" for c in classes])
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:>5} " + " ".join([f"{int(v):8d}" for v in row]))


def test_model(save_report=True):
    # Load test dataset
    print("[INFO] Loading test dataset...")
    try:
        data = safe_load_pickle(config.TEST_FILE)
    except FileNotFoundError:
        print(f"[ERROR] Test dataset not found at {config.TEST_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load test dataset: {e}")
        sys.exit(1)

    # Expecting dict with "features" and "labels"
    if not isinstance(data, dict) or "features" not in data or "labels" not in data:
        print(f"[ERROR] Test file format unexpected. Expected dict with 'features' and 'labels'. Got: {type(data)}")
        sys.exit(1)

    X = np.array(data["features"])
    y = np.array(data["labels"])

    print(f"[INFO] Test data shape: X={X.shape}, y={y.shape}")

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

    # Sanity check: model expected features vs test features
    expected = None
    # 1) try pipeline-level attribute
    expected = getattr(model, "n_features_in_", None)
    # 2) fallback to final estimator (if pipeline)
    if expected is None and hasattr(model, "named_steps"):
        # attempt to find last estimator
        try:
            # find last named step's estimator
            last_step = list(model.named_steps.keys())[-1]
            final_est = model.named_steps[last_step]
            expected = getattr(final_est, "n_features_in_", None)
        except Exception:
            expected = None

    actual = X.reshape(len(X), -1).shape[1]
    if expected is not None and int(expected) != int(actual):
        print(f"[ERROR] Feature dimension mismatch: model expects {expected} but test X has {actual}")
        print("[HINT] Ensure training preprocessing == inference/test preprocessing.")
        sys.exit(1)
    else:
        print(f"[INFO] Feature dimension check passed (model_expected={expected}, test_actual={actual})")

    # Predict on dataset
    print("[INFO] Testing model...")
    try:
        y_pred = model.predict(X)
    except Exception as e:
        print(f"[ERROR] model.predict failed: {e}")
        sys.exit(1)

    # Evaluate performance
    acc = accuracy_score(y, y_pred)
    print(f"[INFO] Accuracy on test set: {acc:.4f}")

    print("\nClassification Report:\n")
    report = classification_report(y, y_pred, output_dict=True)
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    # Print a nicer confusion matrix
    classes = []
    cls_attr = getattr(model, "classes_", None)
    if cls_attr is None and hasattr(model, "named_steps"):
        # fallback to final estimator classes_
        try:
            last_step = list(model.named_steps.keys())[-1]
            final_est = model.named_steps[last_step]
            cls_attr = getattr(final_est, "classes_", None)
        except Exception:
            cls_attr = None
    if cls_attr is not None:
        classes = list(map(str, cls_attr))
    else:
        # default numeric class names
        classes = [str(i) for i in range(cm.shape[0])]

    pretty_print_confusion(cm, classes)

    # Try to print some sample probabilities for the first 5 test samples (if available)
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X[:5])
            print("\nSample predicted probabilities (first 5 samples):")
            for i, p in enumerate(probs):
                top_idx = int(np.argmax(p))
                top_class = classes[top_idx] if len(classes) > top_idx else top_idx
                print(f" sample {i}: top={top_class} conf={float(p[top_idx]):.3f} probs={np.round(p,3).tolist()}")
        except Exception as e:
            print(f"[WARN] predict_proba failed on samples: {e}")
    else:
        print("[INFO] model.predict_proba not available for this model.")

    # Optionally save a report JSON for later inspection
    if save_report:
        out = {
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "classes": classes
        }
        out_path = os.path.splitext(config.TEST_FILE)[0] + "_report.json"
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            # convert any numpy types in report (float64 etc.) by using json dump with default conversion
            with open(out_path, "w") as of:
                json.dump(out, of, indent=2)
            print(f"[INFO] Saved test report to {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save report: {e}")


if __name__ == "__main__":
    test_model()
