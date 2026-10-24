import cv2
import joblib
import pandas as pd
import os

from pose_detection.detector import PoseDetector
from model.classifier import train_model
from utils.angles import extract_features, FEATURE_COLUMNS

CSV_PATH = "pose_data.csv"
MODEL_PATH = "model/yoga_model.pkl"

# How many corrections to collect before auto-retraining
AUTO_RETRAIN_EVERY = 5

# Below this confidence, show "Uncertain" instead of committing to a guess.
CONFIDENCE_THRESHOLD = 0.40

# Load trained model
model = joblib.load(MODEL_PATH)

# Create detector
detector = PoseDetector()

# Open webcam
cap = cv2.VideoCapture(0)

correction_count = 0
last_features = None


def save_correction(features, correct_label):
    row = features + [correct_label]
    row_df = pd.DataFrame([row], columns=FEATURE_COLUMNS + ["label"])

    file_exists = os.path.isfile(CSV_PATH)
    row_df.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)
    print(f"Saved correction: '{correct_label}'")


def retrain_and_reload():
    global model
    print("Retraining model on updated data... this may take a moment.")
    model, accuracy = train_model(csv_path=CSV_PATH, model_path=MODEL_PATH)
    print(f"Retrained. New accuracy on held-out split: {accuracy:.2f}")


print("Controls: [c] correct wrong prediction   [r] retrain now   [q] quit")

while True:

    success, frame = cap.read()
    if not success:
        break

    frame, results = detector.detect_pose(frame)

    pose_name = "No Pose Detected"
    confidence = None

    if results.pose_landmarks:

        features = extract_features(results.pose_landmarks.landmark)
        last_features = features

        input_data = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        prediction = model.predict(input_data)
        pose_name = prediction[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            confidence = max(probs)

            if confidence < CONFIDENCE_THRESHOLD:
                pose_name = "Uncertain"
    else:
        last_features = None

    label_text = f"Pose: {pose_name}"
    if confidence is not None:
        label_text += f" ({confidence * 100:.0f}%)"

    cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "[c] correct   [r] retrain   [q] quit",
                (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Yoga Pose Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        if last_features is None:
            print("No pose detected right now, can't save a correction.")
        else:
            correct_label = input(
                f"Model said '{pose_name}'. Type the CORRECT pose name: "
            ).strip()

            if correct_label:
                save_correction(last_features, correct_label)
                correction_count += 1

                if correction_count >= AUTO_RETRAIN_EVERY:
                    retrain_and_reload()
                    correction_count = 0

    elif key == ord('r'):
        retrain_and_reload()

cap.release()
cv2.destroyAllWindows()