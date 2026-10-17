import os
import time
from collections import deque

import cv2
import joblib
import pandas as pd

from pose_detection.detector import PoseDetector
from model.classifier import train_model
from utils.angles import extract_features, FEATURE_COLUMNS


CSV_PATH = "pose_data.csv"
MODEL_PATH = "model/yoga_model.pkl"

# Tutor settings: require a stable prediction, then hold the target pose.
CONFIDENCE_THRESHOLD = 0.40
VOTE_WINDOW = 15
REQUIRED_TARGET_VOTES = 12
HOLD_SECONDS = 2.0
AUTO_RETRAIN_EVERY = 5


def load_pose_sequence(csv_path):
    """Use all dataset labels once, in a consistent alphabetical order."""
    labels = pd.read_csv(csv_path, usecols=["label"])["label"].dropna().unique()
    return sorted((str(label) for label in labels), key=str.casefold)


POSE_SEQUENCE = load_pose_sequence(CSV_PATH)
if not POSE_SEQUENCE:
    raise ValueError(f"No pose labels found in {CSV_PATH}.")

model = joblib.load(MODEL_PATH)
model_labels = {str(label) for label in getattr(model, "classes_", [])}
missing_labels = [label for label in POSE_SEQUENCE if label not in model_labels]
if missing_labels:
    raise ValueError(
        "The saved model and pose_data.csv have different pose labels. "
        "Retrain from the project folder with "
        "'.\\venv\\Scripts\\python.exe model\\classifier.py', then relaunch. "
        f"Missing from model: {', '.join(missing_labels)}"
    )

detector = PoseDetector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    detector.pose.close()
    raise RuntimeError("Could not open webcam. Check that it is connected and not in use.")

correction_count = 0
last_features = None
pose_index = 0
vote_history = deque(maxlen=VOTE_WINDOW)
hold_started_at = None


def save_correction(features, correct_label):
    row_df = pd.DataFrame([features + [correct_label]], columns=FEATURE_COLUMNS + ["label"])
    file_exists = os.path.isfile(CSV_PATH)
    row_df.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)
    print(f"Saved correction: '{correct_label}'")


def retrain_and_reload():
    global model
    print("Retraining model on updated data... this may take a moment.")
    model, accuracy = train_model(csv_path=CSV_PATH, model_path=MODEL_PATH)
    print(f"Retrained. New accuracy on held-out split: {accuracy:.2f}")


def show_text(frame, text, y, color=(255, 255, 255), scale=0.65):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


print(f"Tutor sequence: {len(POSE_SEQUENCE)} poses, alphabetical by pose name.")
print(f"First target: {POSE_SEQUENCE[0]}")
print("Controls: [c] correct prediction   [r] retrain   [s] restart sequence   [q] quit")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read a webcam frame; stopping the tutor.")
            break

        frame, results = detector.detect_pose(frame)
        detected_pose = "No Pose Detected"
        displayed_pose = detected_pose
        confidence = None
        current_features = None

        if results.pose_landmarks:
            current_features = extract_features(results.pose_landmarks.landmark)
            input_data = pd.DataFrame([current_features], columns=FEATURE_COLUMNS)
            detected_pose = str(model.predict(input_data)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                confidence = float(max(probabilities))

            displayed_pose = detected_pose
            if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
                displayed_pose = "Uncertain"

        last_features = current_features
        confident_pose = (
            detected_pose
            if current_features is not None
            and (confidence is None or confidence >= CONFIDENCE_THRESHOLD)
            else None
        )
        vote_history.append(confident_pose)

        now = time.monotonic()
        target = POSE_SEQUENCE[pose_index] if pose_index < len(POSE_SEQUENCE) else None
        target_votes = sum(pose == target for pose in vote_history) if target else 0
        stable_target = (
            target is not None
            and len(vote_history) == VOTE_WINDOW
            and target_votes >= REQUIRED_TARGET_VOTES
            and confident_pose == target
        )

        if stable_target:
            if hold_started_at is None:
                hold_started_at = now
            held_seconds = now - hold_started_at
            if held_seconds >= HOLD_SECONDS:
                pose_index += 1
                vote_history.clear()
                hold_started_at = None
                target = POSE_SEQUENCE[pose_index] if pose_index < len(POSE_SEQUENCE) else None
                if target is None:
                    print(f"Sequence complete! You completed all {len(POSE_SEQUENCE)} poses.")
                else:
                    print(f"Next pose ({pose_index + 1}/{len(POSE_SEQUENCE)}): {target}")
        else:
            hold_started_at = None
            held_seconds = 0.0

        show_text(frame, f"Detected: {displayed_pose}", 35, (0, 255, 0))
        if confidence is not None:
            show_text(frame, f"Confidence: {confidence * 100:.0f}%", 65)

        if target is not None:
            show_text(frame, f"Target ({pose_index + 1}/{len(POSE_SEQUENCE)}): {target}", 105, (0, 220, 255))
            if stable_target:
                show_text(frame, f"Hold steady: {min(held_seconds, HOLD_SECONDS):.1f}/{HOLD_SECONDS:.1f}s", 140)
            elif target_votes:
                show_text(frame, f"Pose stabilizing: {target_votes}/{REQUIRED_TARGET_VOTES}", 140)
            else:
                show_text(frame, "Move into the target pose", 140)
            next_pose = POSE_SEQUENCE[pose_index + 1] if pose_index + 1 < len(POSE_SEQUENCE) else "Sequence complete"
            show_text(frame, f"After this: {next_pose}", 175)
        else:
            show_text(frame, f"All {len(POSE_SEQUENCE)} poses completed! Press [s] to restart.", 105, (0, 255, 0))

        show_text(frame, "[c] correct  [r] retrain  [s] restart  [q] quit", frame.shape[0] - 20, scale=0.5)
        cv2.imshow("Yoga Pose Tutor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            pose_index = 0
            vote_history.clear()
            hold_started_at = None
            print(f"Sequence restarted. First target: {POSE_SEQUENCE[0]}")
        elif key == ord("c"):
            if last_features is None:
                print("No pose detected right now, can't save a correction.")
            else:
                correct_label = input(
                    f"Model said '{displayed_pose}'. Type the CORRECT pose name: "
                ).strip()
                if correct_label:
                    save_correction(last_features, correct_label)
                    correction_count += 1
                    if correction_count >= AUTO_RETRAIN_EVERY:
                        retrain_and_reload()
                        correction_count = 0
        elif key == ord("r"):
            retrain_and_reload()

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.pose.close()
