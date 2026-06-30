from pathlib import Path
import argparse

from ultralytics import YOLO

from src.preprocessing.hand_crop import crop_hand_from_image


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "runs" / "mudra_yolo_cls_baseline" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "outputs" / "crops"


def predict(model, image):
    results = model.predict(source=image, verbose=False)
    result = results[0]

    probs = result.probs

    top1_id = int(probs.top1)
    top1_conf = float(probs.top1conf)
    top1_label = result.names[top1_id]

    print(f"Predicted mudra: {top1_label}")
    print(f"Confidence: {top1_conf:.2%}")

    print("\nTop 5 predictions:")
    for class_id, confidence in zip(probs.top5, probs.top5conf):
        label = result.names[int(class_id)]
        print(f"- {label}: {float(confidence):.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to uploaded mudra image")
    args = parser.parse_args()

    image_path = Path(args.image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cropped_hand = crop_hand_from_image(image_path)

    if cropped_hand is None:
        print("No hand detected. Try a clearer image with the full hand visible.")
        return

    crop_path = OUTPUT_DIR / f"{image_path.stem}_hand_crop.jpg"
    cropped_hand.save(crop_path)

    print(f"Saved hand crop to: {crop_path}")

    model = YOLO(str(MODEL_PATH))
    predict(model, cropped_hand)


if __name__ == "__main__":
    main()