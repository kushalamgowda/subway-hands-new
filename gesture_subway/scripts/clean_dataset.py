# scripts/clean_dataset.py
import pickle
import config
import os

def remove_features(obj):
    """Recursively remove all 'features' keys from dictionaries/lists."""
    if isinstance(obj, dict):
        if "features" in obj:
            del obj["features"]
        for key in list(obj.keys()):
            obj[key] = remove_features(obj[key])
    elif isinstance(obj, list):
        return [remove_features(item) for item in obj]
    return obj

# clean one pickle file
def clean_file(path):
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return
    
    with open(path, "rb") as f:
        data = pickle.load(f)

    cleaned = remove_features(data)

    with open(path, "wb") as f:
        pickle.dump(cleaned, f)

    print(f"[INFO] Cleaned dataset saved: {path}")


# Clean main dataset
clean_file(config.DATA_FILE)

# OPTIONAL: also clean test.pkl if you have it
test_path = os.path.join(config.BASE_DIR, "test.pkl")
if os.path.exists(test_path):
    clean_file(test_path)
