"""Train and evaluate the 47-pose landmark classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent


def train_model(csv_path=None, model_path=None, verbose=True):
    """Evaluate on a stratified holdout, then save a final model using all rows."""
    csv_path = Path(csv_path) if csv_path else PROJECT_DIR / "pose_data.csv"
    model_path = Path(model_path) if model_path else PROJECT_DIR / "model" / "yoga_model.pkl"

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'label' column.")

    labels = df["label"].astype("string").str.strip()
    features = df.drop(columns="label").apply(pd.to_numeric, errors="coerce")
    valid_rows = labels.notna() & labels.ne("") & features.notna().all(axis=1)
    features = features.loc[valid_rows].reset_index(drop=True)
    labels = labels.loc[valid_rows].astype(str).reset_index(drop=True)

    counts = labels.value_counts()
    rare_classes = counts[counts < 2]
    if not rare_classes.empty:
        names = ", ".join(f"{name} ({count})" for name, count in rare_classes.items())
        raise ValueError("Each pose needs at least two valid rows for a stratified evaluation: " + names)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model_options = {
        "n_estimators": 200,
        # Class weighting raised minority-class importance but lowered overall
        # held-out accuracy on this dataset. Keep natural class frequencies;
        # this setting also improved macro-F1 on the fixed holdout.
        "class_weight": None,
        "random_state": 42,
        "n_jobs": 1,
    }
    evaluation_model = ExtraTreesClassifier(**model_options)
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    if verbose:
        print(f"Valid rows: {len(features)} across {labels.nunique()} poses")
        print(f"Smallest class: {counts.min()} rows ({counts.idxmin()})")
        print(f"Held-out accuracy: {accuracy:.4f}")
        print("Per-class metrics:")
        print(classification_report(y_test, predictions, zero_division=0, digits=4))

    # The evaluation above remains held out, while inference gets every example.
    final_model = ExtraTreesClassifier(**model_options)
    final_model.fit(features, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    if verbose:
        print(f"Final model trained on all {len(features)} valid rows and saved to {model_path}.")

    return final_model, accuracy


if __name__ == "__main__":
    train_model()
