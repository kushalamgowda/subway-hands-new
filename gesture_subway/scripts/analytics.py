# analytics.py
import json
import time
from collections import defaultdict

class AnalyticsLogger:
    def __init__(self, log_file="data/analytics_log.json"):
        self.log_file = log_file
        self.data = defaultdict(list)
        self.start_time = time.time()

    def log_gesture(self, gesture_name):
        """Logs gesture action with timestamp."""
        entry = {
            "gesture": gesture_name,
            "timestamp": time.time()
        }
        self.data["gestures"].append(entry)
        print(f"[ANALYTICS] Logged gesture: {gesture_name}")

    def log_accuracy(self, predicted, actual):
        """Logs model accuracy comparison."""
        entry = {
            "predicted": predicted,
            "actual": actual,
            "timestamp": time.time()
        }
        self.data["accuracy"].append(entry)
        print(f"[ANALYTICS] Predicted: {predicted}, Actual: {actual}")

    def log_session(self):
        """Logs overall session duration and gesture counts."""
        session_duration = time.time() - self.start_time
        summary = {
            "session_duration_sec": round(session_duration, 2),
            "gesture_counts": self.count_gestures(),
            "total_gestures": len(self.data.get("gestures", []))
        }
        self.data["session_summary"].append(summary)
        print(f"[ANALYTICS] Session Summary: {summary}")
        self.save_log()

    def count_gestures(self):
        """Counts how many times each gesture was used."""
        counts = defaultdict(int)
        for g in self.data.get("gestures", []):
            counts[g["gesture"]] += 1
        return dict(counts)

    def save_log(self):
        """Saves analytics data to JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=4)
        print(f"[ANALYTICS] Saved log to {self.log_file}")
