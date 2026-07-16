import numpy as np

# MediaPipe Pose landmark indices we need
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Each entry: (name, point_a, vertex_point, point_c)
# Angle is measured AT the vertex point
ANGLE_DEFINITIONS = [
    ("left_elbow_angle", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    ("right_elbow_angle", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ("left_shoulder_angle", LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    ("right_shoulder_angle", RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    ("left_hip_angle", LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    ("right_hip_angle", RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    ("left_knee_angle", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    ("right_knee_angle", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
]

ANGLE_COLUMNS = [name for name, _, _, _ in ANGLE_DEFINITIONS]


def _calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) at point b, formed by points a-b-c.
    a, b, c are (x, y) tuples.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Avoid division by zero if landmarks overlap
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # guard against float rounding errors

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def extract_angles(landmarks):
    """
    Takes a MediaPipe pose_landmarks.landmark list and returns a list of
    joint angles (in degrees) in the fixed order defined by ANGLE_COLUMNS.

    This is position- and scale-invariant: the same pose gives the same
    angles regardless of where the person stands in the frame or how far
    they are from the camera.
    """
    points = [(lm.x, lm.y) for lm in landmarks]

    angles = []
    for _, idx_a, idx_b, idx_c in ANGLE_DEFINITIONS:
        angle = _calculate_angle(points[idx_a], points[idx_b], points[idx_c])
        angles.append(angle)

    return angles
