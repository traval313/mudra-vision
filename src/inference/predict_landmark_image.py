from pathlib import Path
import argparse

import joblib
import pandas as pd

from src.features.hand_landmarks import (
    create_hands_detector,
    extract_landmark_features_from_image,
)
from src.preprocessing.hand_crop import crop_hand_from_image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "landmark_classifier.joblib"


def main():
    parser = argparse.ArgumentParser(
        description="Predict a mudra class from MediaPipe hand landmarks."
    )
    parser.add_argument("image_path", type=Path, help="Path to the image to classify.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    cropped_hand = crop_hand_from_image(args.image_path)

    if cropped_hand is None:
        print("No hand detected. Try a clearer image with the full hand visible.")
        return

    with create_hands_detector() as hands:
        features = extract_landmark_features_from_image(cropped_hand, hands)

    if features is None:
        print("No hand landmarks detected in the cropped hand image.")
        return

    feature_frame = pd.DataFrame([features], columns=feature_columns)
    prediction = model.predict(feature_frame)[0]

    print(f"Predicted mudra: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)[0]
        class_scores = sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )

        print("\nTop predictions:")
        for label, confidence in class_scores[: args.top_k]:
            print(f"- {label}: {confidence:.2%}")


if __name__ == "__main__":
    main()
