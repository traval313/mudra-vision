from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.features.hand_landmarks import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = ROOT / "outputs" / "features" / "train_landmarks.csv"
DEFAULT_VAL_CSV = ROOT / "outputs" / "features" / "val_landmarks.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "landmark_classifier.joblib"


def load_features(csv_path):
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"No landmark rows found in {csv_path}")

    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} is missing feature columns: {missing_columns[:5]}")

    return df[FEATURE_COLUMNS], df["label"]


def main():
    parser = argparse.ArgumentParser(
        description="Train a mudra classifier from MediaPipe hand landmarks."
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")

    x_train, y_train = load_features(args.train_csv)

    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": classifier,
            "feature_columns": FEATURE_COLUMNS,
            "classes": classifier.classes_.tolist(),
        },
        args.model_path,
    )

    print(f"Saved landmark classifier to: {args.model_path}")

    if args.val_csv.exists():
        try:
            x_val, y_val = load_features(args.val_csv)
        except ValueError as error:
            print(f"Skipped validation: {error}")
            return

        predictions = classifier.predict(x_val)
        accuracy = accuracy_score(y_val, predictions)

        print(f"\nValidation accuracy: {accuracy:.2%}")
        print("\nClassification report:")
        print(classification_report(y_val, predictions, zero_division=0))
    else:
        print(f"Validation CSV not found, skipped evaluation: {args.val_csv}")


if __name__ == "__main__":
    main()
