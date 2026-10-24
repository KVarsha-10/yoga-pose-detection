"""Flask based browser yoga tutor. Run with ``python app.py``."""

from __future__ import annotations

import csv
import math
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "pose_data.csv"
MODEL_PATH = ROOT / "model" / "yoga_model.pkl"
REFERENCE_DIR = ROOT / "pose_references"
CONFIDENCE_THRESHOLD = 0.30
VOTE_WINDOW = 10
REQUIRED_TARGET_VOTES = 7


def load_poses():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("pose_data.csv must contain a 'label' column")
        return sorted({r["label"].strip() for r in reader if r.get("label", "").strip()}, key=str.casefold)


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


app = Flask(__name__)
POSES = load_poses()


class Tutor:
    """Owns one camera, pose detector, and model for the active browser session."""

    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.cap = None
        self.detector = None
        self.model = None
        self.pd = None
        self.np = None
        self.cv2 = None
        self.features_fn = None
        self.feature_columns = None
        self.angle_columns = None
        self.compare_angles = None
        self.profiles = None
        self.training = None
        self.sequence = []
        self.index = 0
        self.hold_seconds = 10
        self.rest_seconds = 0
        self.hold_started = None
        self.rest_until = None
        self.paused = False
        self.active = False
        self.complete = False
        self.votes = deque(maxlen=VOTE_WINDOW)
        self.probabilities = deque(maxlen=5)
        self.frame_bytes = None
        self.frame_sequence = 0
        self.frame_updated = threading.Condition(self.lock)
        self.state = {}
        self.prompt_id = 0
        self.feedback_key = None
        self.last_features = None
        self.last_detected = "Camera is starting"
        self.last_speech_time = 0

    def prompt(self, text):
        self.prompt_id += 1
        self.state["prompt_id"] = self.prompt_id
        self.state["prompt"] = text

    def target(self):
        return self.sequence[self.index] if self.index < len(self.sequence) else None

    def public_state(self):
        with self.lock:
            now = time.monotonic()
            remaining = self.hold_seconds
            mode = "waiting"
            if self.complete:
                mode = "complete"
                remaining = 0
            elif not self.active:
                mode = "idle"
            elif self.paused:
                mode = "paused"
            elif self.rest_until is not None:
                remaining = max(0, math.ceil(self.rest_until - now))
                mode = "rest" if remaining else "waiting"
            elif self.hold_started is not None:
                remaining = max(0, math.ceil(self.hold_seconds - (now - self.hold_started)))
                mode = "holding"
            state = dict(self.state)
            state.update({
                "active": self.active, "complete": self.complete, "paused": self.paused,
                "mode": mode, "target": self.target(), "step": min(self.index + 1, len(self.sequence)),
                "total": len(self.sequence), "hold_seconds": self.hold_seconds,
                "remaining": remaining, "detected": self.last_detected,
                "frame_ready": self.frame_sequence > 0,
            })
            return state

    def start(self, sequence, hold_seconds, rest_seconds):
        self.stop()
        import cv2
        import joblib
        import numpy as np
        import pandas as pd
        from pose_detection.detector import PoseDetector
        from utils.angles import ANGLE_COLUMNS, FEATURE_COLUMNS, extract_features
        from utils.pose_feedback import build_reference_profiles, compare_angles_to_examples

        model = joblib.load(MODEL_PATH)
        missing = set(POSES) - {str(x) for x in getattr(model, "classes_", [])}
        if missing:
            raise RuntimeError("Saved model does not include every pose in pose_data.csv; retrain it first.")
        training = pd.read_csv(CSV_PATH)
        detector = PoseDetector()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            detector.pose.close()
            raise RuntimeError("Could not open webcam. Check that it is connected and not in use.")
        with self.lock:
            self.cv2, self.np, self.pd = cv2, np, pd
            self.model, self.training, self.detector, self.cap = model, training, detector, cap
            self.features_fn, self.feature_columns = extract_features, FEATURE_COLUMNS
            self.angle_columns, self.compare_angles = ANGLE_COLUMNS, compare_angles_to_examples
            self.profiles = build_reference_profiles(training, ANGLE_COLUMNS)
            self.sequence, self.index = sequence, 0
            self.hold_seconds, self.rest_seconds = hold_seconds, rest_seconds
            self.hold_started, self.rest_until = None, None
            self.paused, self.complete, self.active = False, False, True
            self.votes.clear(); self.probabilities.clear()
            self.frame_bytes = None
            self.frame_sequence = 0
            self.feedback_key = None
            self.last_detected = "Waiting for you to enter the camera view"
            self.state = {"feedback": "Stand where your full body is visible.", "confidence": None,
                          "prompt": "", "prompt_id": 0}
            self.prompt(f"Starting your sequence. First pose: {sequence[0]}. Hold for {hold_seconds} seconds.")
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="yoga-camera")
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        thread = self.thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        with self.lock:
            if self.cap is not None:
                self.cap.release(); self.cap = None
            if self.detector is not None:
                try: self.detector.pose.close()
                except Exception: pass
                self.detector = None
            self.thread = None
            self.active = False
            self.paused = False
            self.hold_started = None
            self.rest_until = None
            self.votes.clear(); self.probabilities.clear()

    def advance(self, skipped=False):
        self.index += 1
        self.hold_started = None
        self.votes.clear(); self.probabilities.clear()
        self.feedback_key = None
        target = self.target()
        if target is None:
            self.complete = True
            self.prompt("Sequence complete. Well done. Take a moment to notice how you feel.")
            self.state["feedback"] = "You completed every pose in your sequence. Well done!"
            return
        if self.rest_seconds:
            self.rest_until = time.monotonic() + self.rest_seconds
            self.prompt(f"Pose complete. Rest for {self.rest_seconds} seconds. Next pose: {target}. After your rest, hold it for {self.hold_seconds} seconds.")
            self.state["feedback"] = f"Rest break. Next: {target}."
        else:
            self.rest_until = None
            self.prompt(f"Next pose: {target}. Hold for {self.hold_seconds} seconds.")
            self.state["feedback"] = f"Now move into {target}."

    def _capture_loop(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            with self.lock:
                cap, detector = self.cap, self.detector
            if cap is None or detector is None:
                break
            ok, frame = cap.read()
            if not ok:
                with self.lock: self.state["feedback"] = "Camera frame unavailable. Check camera permissions."
                time.sleep(.1); continue
            frame = self.cv2.flip(frame, 1)
            frame, results = detector.detect_pose(frame)
            detected, confidence, features, visible = "No pose detected", None, None, False
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                key = (11, 12, 23, 24, 25, 26, 27, 28)
                visible = sum(
    float(landmarks[i].visibility)
    for i in key
) / len(key) >= .35
                if visible:
                    features = self.features_fn(landmarks)
                    row = self.pd.DataFrame([features], columns=self.feature_columns)
                    probs = self.model.predict_proba(row)[0]
                    with self.lock:
                        self.probabilities.append(probs)
                        probs = self.np.mean(self.probabilities, axis=0)
                    best = int(self.np.argmax(probs))
                    detected, confidence = str(self.model.classes_[best]), float(probs[best])
                else:
                    detected = "Move back so your full body is visible"
                    with self.lock: self.probabilities.clear()
            else:
                with self.lock: self.probabilities.clear()
            with self.lock:
                self.last_detected = detected
                self.last_features = features
                self._update_logic(detected if confidence is not None and confidence >= CONFIDENCE_THRESHOLD else None,
                                   confidence, features, visible)
                label = f"Detected: {detected}" + (f"  ·  {confidence:.0%}" if confidence is not None else "")
                self.cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (22, 44, 38), -1)
                self.cv2.putText(frame, label[:90], (16, 28), self.cv2.FONT_HERSHEY_SIMPLEX,
                                 .65, (235, 245, 235), 2)
                ok, jpeg = self.cv2.imencode(".jpg", frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    self.frame_bytes = jpeg.tobytes()
                    self.frame_sequence += 1
                    self.frame_updated.notify_all()
            time.sleep(max(0, .035 - (time.monotonic() - started)))

    def _update_logic(self, prediction, confidence, features, visible):
        if not self.active or self.complete or self.paused:
            return
        now, target = time.monotonic(), self.target()
        self.votes.append(prediction)
        self.state["confidence"] = confidence
        if self.rest_until is not None:
            left = max(0, math.ceil(self.rest_until - now))
            if left:
                self.state["feedback"] = f"Rest break · {left} seconds. Next pose: {target}."
                self.votes.clear()
                return
            self.rest_until = None
            self.votes.clear()
            self.prompt(f"Rest complete. Now do {target}. Hold for {self.hold_seconds} seconds.")
        stable = (len(self.votes) == VOTE_WINDOW and sum(x == target for x in self.votes) >= REQUIRED_TARGET_VOTES
                  and prediction == target)
        if prediction == target and features is not None:
            feedback = self.compare_angles(target, features, self.feature_columns, self.profiles)
            self.state["feedback"] = "Compared with dataset examples: " + feedback["text"]
            speech = feedback.get("speech")
            key = re.sub(r"\b\d+(?:\.\d+)?\b", "", speech or "").strip()

        if (speech and key != self.feedback_key and time.monotonic() - self.last_speech_time > 4):
            self.prompt(speech)
            self.feedback_key = key
            self.last_speech_time = time.monotonic()
        else:
            self.feedback_key = None
            self.state["feedback"] = ("Move back until your shoulders, hips, knees, and ankles are visible."
                                       if not visible else f"Move into {target} to begin the hold.")
        if stable:
            if self.hold_started is None:
                self.hold_started = now
                self.prompt(f"Good. You are in {target}. Hold for {self.hold_seconds} seconds.")
            elapsed = now - self.hold_started
            left = max(0, math.ceil(self.hold_seconds - elapsed))
            self.state["remaining"] = left
            if left in (5, 3, 2, 1) and getattr(self, "last_countdown", None) != left:
                self.prompt(f"{left} seconds remaining."); self.last_countdown = left
            if elapsed >= self.hold_seconds:
                self.advance()
        else:
            self.hold_started = None
            self.state["remaining"] = self.hold_seconds


tutor = Tutor()


@app.get("/")
def index():
    return render_template("index.html", poses=POSES)


@app.get("/api/state")
def api_state():
    return jsonify(tutor.public_state())


@app.post("/api/start")
def api_start():
    data = request.get_json(silent=True) or {}
    sequence = data.get("sequence")
    if not isinstance(sequence, list) or not sequence:
        return jsonify(error="Choose at least one pose."), 400
    if any(p not in POSES for p in sequence):
        return jsonify(error="The sequence includes an unknown pose."), 400
    try:
        hold = int(data.get("hold_seconds", 10)); rest = int(data.get("rest_seconds", 0))
        if not 1 <= hold <= 120 or not 0 <= rest <= 120: raise ValueError
    except (TypeError, ValueError):
        return jsonify(error="Hold and rest must be whole seconds (hold 1–120, rest 0–120)."), 400
    try:
        tutor.start(sequence, hold, rest)
        return jsonify(tutor.public_state())
    except Exception as exc:
        tutor.stop()
        return jsonify(error=str(exc)), 500


@app.post("/api/stop")
def api_stop():
    tutor.stop()
    with tutor.lock:
        tutor.complete = False
        tutor.state = {"feedback": "Session ended.", "prompt": "", "prompt_id": tutor.prompt_id}
    return jsonify(ok=True)


@app.post("/api/pause")
def api_pause():
    data = request.get_json(silent=True) or {}
    with tutor.lock:
        if not tutor.active: return jsonify(error="No active session."), 400
        tutor.paused = bool(data.get("paused"))
        tutor.hold_started = None
        tutor.votes.clear(); tutor.probabilities.clear()
        tutor.state["feedback"] = "Practice paused. Resume when ready." if tutor.paused else "Practice resumed. Find your pose again."
        tutor.prompt("Practice paused." if tutor.paused else "Practice resumed. Find your pose again.")
    return jsonify(tutor.public_state())


@app.post("/api/skip")
def api_skip():
    with tutor.lock:
        if tutor.active and not tutor.complete: tutor.advance(skipped=True)
    return jsonify(tutor.public_state())


@app.post("/api/correct")
def api_correct():
    data = request.get_json(silent=True) or {}
    label = data.get("label")
    with tutor.lock:
        if label not in POSES: return jsonify(error="Choose a pose from the list."), 400
        if tutor.last_features is None: return jsonify(error="No full-body pose is visible yet."), 400
        row = tutor.pd.DataFrame([tutor.last_features + [label]], columns=tutor.feature_columns + ["label"])
        row.to_csv(CSV_PATH, mode="a", header=False, index=False)
        tutor.state["feedback"] = f"Saved correction as {label}. Retrain the model to include corrections."
    return jsonify(ok=True, feedback=tutor.state["feedback"])


@app.get("/video_feed")
def video_feed():
    def frames():
        last_sequence = 0
        while True:
            with tutor.frame_updated:
                tutor.frame_updated.wait_for(
                    lambda: tutor.frame_sequence > last_sequence or not tutor.active,
                    timeout=2,
                )
                frame = tutor.frame_bytes
                sequence = tutor.frame_sequence
                active = tutor.active
            if frame and sequence > last_sequence:
                last_sequence = sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
            elif not active:
                break
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-cache"})

DATASET_DIR = ROOT / "dataset"
@app.get("/reference/<path:pose_name>")
def reference(pose_name):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Prefer curated images stored with the documented slugged filenames.
    for extension in sorted(extensions):
        reference_image = REFERENCE_DIR / f"{slug(pose_name)}{extension}"
        if reference_image.is_file():
            return send_file(reference_image)

    # Support extracted datasets with class-named folders, regardless of case.
    if DATASET_DIR.is_dir():
        folders = [
            folder for folder in DATASET_DIR.rglob("*")
            if folder.is_dir() and folder.name.casefold() == pose_name.casefold()
        ]
        images = sorted(
            image
            for folder in folders
            for image in folder.rglob("*")
            if image.is_file() and image.suffix.casefold() in extensions
        )
        if images:
            return send_file(images[0])

    return "", 404


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )

