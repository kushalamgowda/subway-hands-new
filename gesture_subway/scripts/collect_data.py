# collect_data.py (interactive - single prompt for gesture number only)
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import sys

import config

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# --------------------------------------------
# Feature extraction
# --------------------------------------------
def normalize_landmarks(landmarks):
    lm = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=float)
    center = lm[0].copy()
    lm -= center
    dists = np.linalg.norm(lm, axis=1)
    max_dist = np.max(dists)
    if max_dist > 1e-6:
        lm /= max_dist
    return lm.flatten()

def extract_features(landmarks):
    return normalize_landmarks(landmarks)

# --------------------------------------------
# Collect data function
# --------------------------------------------
def collect_data(gesture_name, samples=None, camera_index=0, show_window=True):
    """
    Collect `samples` normalized landmark feature vectors for `gesture_name`.
    If samples is None, uses config.SAMPLES_PER_GESTURE.
    Press 'q' in the camera window to stop collection early (returns to menu).
    Press Ctrl+C to exit the whole program.
    """
    samples = samples if samples is not None else getattr(config, "SAMPLES_PER_GESTURE", 100)
    print(f"[INFO] Starting data collection for: {gesture_name} (target: {samples})")

    # open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return

    data = []
    count = 0
    start_t = time.time()

    try:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while cap.isOpened() and count < samples:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Empty frame")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    try:
                        features = extract_features(hand_landmarks)
                        if features.size == 42:
                            data.append(features.astype(np.float32))
                            count += 1
                        else:
                            print(f"[WARN] Unexpected feature size: {features.size}")
                    except Exception as e:
                        print(f"[WARN] Skipping sample due to: {e}")

                    cv2.putText(frame, f"{gesture_name}: {count}/{samples}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if show_window:
                    cv2.imshow("Collecting Gesture Data (press 'q' to stop)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] Stopped by user (q pressed). Returning to menu.")
                        break
                else:
                    # small sleep to avoid busy loop if no window
                    time.sleep(0.01)

    except KeyboardInterrupt:
        # Propagate so the caller (main loop) can exit cleanly
        print("\n[INFO] KeyboardInterrupt detected during collection. Exiting.")
        raise
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Save data safely (append to existing dict keyed by gesture_name)
    if len(data) > 0:
        all_data = {}
        if os.path.exists(config.DATA_FILE):
            try:
                with open(config.DATA_FILE, "rb") as f:
                    all_data = pickle.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load existing data file: {e}. Overwriting.")

        new_arr = np.vstack(data).astype(np.float32)

        if gesture_name in all_data and isinstance(all_data[gesture_name], (list, np.ndarray)) and len(all_data[gesture_name]) > 0:
            existing = np.array(all_data[gesture_name], dtype=np.float32)
            combined = np.vstack([existing, new_arr])
        else:
            combined = new_arr

        all_data[gesture_name] = combined

        # atomic-ish write
        tmp_path = config.DATA_FILE + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(all_data, f)
        os.replace(tmp_path, config.DATA_FILE)

        elapsed = time.time() - start_t
        print(f"[INFO] Saved {len(new_arr)} samples for '{gesture_name}' (took {elapsed:.1f}s).")
    else:
        print("[WARN] No data collected for this run.")

# --------------------------------------------
# Interactive Menu (Main)
# --------------------------------------------
if __name__ == "__main__":
    GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "stop", "start"]
    CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)

    try:
        while True:
            print("\n======================")
            print("   Gesture Collector")
            print("======================")

            for i, g in enumerate(GESTURES, start=1):
                print(f"{i}. {g}")

            print("0. Exit")

            try:
                choice = input("\nSelect gesture number (press Ctrl+C to exit): ")
            except KeyboardInterrupt:
                print("\n[INFO] KeyboardInterrupt received. Exiting.")
                break

            if choice.strip() == "":
                print("[ERROR] No input detected. Please enter a number.")
                continue

            if choice == "0":
                print("Exiting...")
                break

            try:
                choice_int = int(choice)
                if choice_int < 1 or choice_int > len(GESTURES):
                    print("[ERROR] Choice out of range.")
                    continue
                gesture = GESTURES[choice_int - 1]
            except Exception:
                print("[ERROR] Invalid choice. Enter a number corresponding to the gesture.")
                continue

            # Immediately start collection using default samples from config
            try:
                collect_data(gesture, samples=getattr(config, "SAMPLES_PER_GESTURE", 100), camera_index=CAMERA_INDEX, show_window=True)
            except KeyboardInterrupt:
                # Ctrl+C pressed during collect_data -> exit whole program cleanly
                print("\n[INFO] KeyboardInterrupt received during collection. Exiting program.")
                break

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(0)
