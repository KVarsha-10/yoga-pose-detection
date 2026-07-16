import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model(csv_path="pose_data.csv", model_path="model/yoga_model.pkl", verbose=True):
    """
    Trains a RandomForestClassifier on the joint-angle CSV and saves it.
    Can be run directly (python model/classifier.py) or imported and
    called from app.py to retrain after new correction data is added.
    """

    df = pd.read_csv(csv_path)

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        print("Accuracy:", accuracy)

    joblib.dump(model, model_path)

    if verbose:
        print("Model saved successfully!")

    return model, accuracy


if __name__ == "__main__":
    train_model()
