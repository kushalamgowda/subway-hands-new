# scripts/clean_dataset.py
import pickle
import config

with open(config.DATA_FILE, "rb") as f:
    data = pickle.load(f)

# remove unwanted label
if "features" in data:
    del data["features"]

with open(config.DATA_FILE, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Cleaned dataset saved without 'features'")
