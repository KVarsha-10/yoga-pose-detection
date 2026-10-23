"""Approximate joint-angle comparison against labeled training examples."""


ANGLE_LABELS = {
    "left_elbow_angle": ("Left elbow", "elbow"),
    "right_elbow_angle": ("Right elbow", "elbow"),
    "left_shoulder_angle": ("Left shoulder", "shoulder"),
    "right_shoulder_angle": ("Right shoulder", "shoulder"),
    "left_hip_angle": ("Left hip", "hip"),
    "right_hip_angle": ("Right hip", "hip"),
    "left_knee_angle": ("Left knee", "knee"),
    "right_knee_angle": ("Right knee", "knee"),
}


def build_reference_profiles(dataframe, angle_columns):
    """Create robust per-pose angle ranges from the static training rows."""
    profiles = {}
    for label, pose_rows in dataframe.groupby("label"):
        pose_profile = {}
        for column in angle_columns:
            values = pose_rows[column].dropna()
            if values.empty:
                continue
            pose_profile[column] = {
                "median": float(values.median()),
                "low": float(values.quantile(0.20)),
                "high": float(values.quantile(0.80)),
            }
        profiles[str(label)] = pose_profile
    return profiles


def compare_angles_to_examples(pose_name, features, feature_columns, profiles):
    """Return one actionable angle cue for the largest clear deviation."""
    pose_profile = profiles.get(pose_name)
    if not pose_profile:
        return {"text": "No angle examples are available for this pose.", "speech": None}

    deviations = []
    for column, (display_name, joint_type) in ANGLE_LABELS.items():
        reference = pose_profile.get(column)
        if reference is None:
            continue
        value = float(features[feature_columns.index(column)])
        low, high = reference["low"], reference["high"]
        if low <= value <= high:
            continue

        outside_by = low - value if value < low else value - high
        spread = max(high - low, 8.0)
        median_difference = value - reference["median"]

        if joint_type in ("elbow", "knee"):
            if median_difference < 0:
                cue = f"{display_name} is about {abs(median_difference):.0f} degrees more bent than examples; gently straighten it."
            else:
                cue = f"{display_name} is about {median_difference:.0f} degrees straighter than examples; bend it slightly."
        elif joint_type == "hip":
            if median_difference < 0:
                cue = f"{display_name} is about {abs(median_difference):.0f} degrees more folded than examples; ease the fold slightly."
            else:
                cue = f"{display_name} is about {median_difference:.0f} degrees more open than examples; fold a little more."
        else:
            direction = "lower" if median_difference < 0 else "higher"
            cue = (
                f"{display_name} angle is about {abs(median_difference):.0f} degrees {direction} "
                "than examples; adjust your arm gently toward the reference."
            )

        deviations.append((outside_by / spread, cue))

    if not deviations:
        return {
            "text": "Joint angles are within the range of the training examples.",
            "speech": None,
        }

    _, cue = max(deviations, key=lambda item: item[0])
    return {"text": cue, "speech": cue}
