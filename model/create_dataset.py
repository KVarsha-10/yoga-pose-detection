import os
import cv2
import mediapipe as mp
import pandas as pd
import sys

# Allow importing from utils/ when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.angles import extract_angles, ANGLE_COLUMNS

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

DATASET_PATH = "dataset"
OUTPUT_CSV = "pose_data.csv"

data = []

for label in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    print("Processing:", label)
    image_count = 0

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            angles = extract_angles(results.pose_landmarks.landmark)
            angles.append(label)
            data.append(angles)
            image_count += 1

    print(f"  -> {image_count} usable images")

columns = ANGLE_COLUMNS + ["label"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nCSV created successfully! {len(df)} total samples across {df['label'].nunique()} poses.")
print(f"Saved to: {OUTPUT_CSV}")
