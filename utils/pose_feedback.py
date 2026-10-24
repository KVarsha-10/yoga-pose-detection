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


def compare_angles_to_examples(
    pose_name,
    features,
    feature_columns,
    profiles
):
    pose_profile = profiles.get(pose_name)

    if not pose_profile:
        return {
            "text": "No angle examples are available.",
            "speech": None
        }

    deviations = []

    for column, (_, joint_type) in ANGLE_LABELS.items():

        reference = pose_profile.get(column)

        if reference is None:
            continue

        value = float(
            features[
                feature_columns.index(column)
            ]
        )

        median = reference["median"]

        deviations.append(
            (
                abs(value - median),
                column,
                value - median
            )
        )

    if not deviations:
        return {
            "text": "Good alignment.",
            "speech": "Good alignment."
        }

    deviations.sort(reverse=True)

    _, column, difference = deviations[0]

    # -------- GARUDASANA --------

    if pose_name.lower().strip() == "garudasana":

        if "shoulder" in column:

            cue = (
                "Lift your elbows to shoulder height and keep your forearms crossed."
            )

        elif "elbow" in column:

            cue = (
                "Wrap your forearms more tightly and keep the elbows lifted."
            )

        elif "knee" in column:

            cue = (
                "Cross the thighs firmly and sink a little deeper into the standing leg."
            )

        elif "hip" in column:

            cue = (
                "Keep your hips centered and lengthen through the spine."
            )

    else:

        cue = (
            "Maintain balance and keep the Eagle pose compact and steady."
        )

    return {
        "text": cue,
        "speech": cue
    }

    # -------- GENERAL POSES --------

    if "elbow" in column:

        cue = (
            "Straighten the elbows slightly."
            if difference < 0
            else
            "Bend the elbows slightly."
        )

    elif "shoulder" in column:

        cue = (
            "Raise the arms a little higher."
            if difference < 0
            else
            "Lower the shoulders slightly."
        )

    elif "knee" in column:

        cue = (
            "Bend the knees slightly."
            if difference > 0
            else
            "Straighten the knees gently."
        )

    elif "hip" in column:

        cue = (
            "Open the hips a little more."
            if difference > 0
            else
            "Soften the hip fold slightly."
        )

    else:

        cue = "Adjust the pose gently."

    return {
        "text": cue,
        "speech": cue
    }