from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "runs" / "mudra_yolo_cls_baseline" / "weights" / "best.pt"

# Change this folder to any class that exists in images/val
IMAGE_FOLDER = ROOT / "images" / "val" / "Pathaka"


def get_first_image(folder: Path) -> Path:
    for path in folder.iterdir():
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return path
    raise FileNotFoundError(f"No image found in {folder}")


def main():
    image_path = get_first_image(IMAGE_FOLDER)

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=str(image_path))

    result = results[0]
    probs = result.probs

    top_class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    class_name = result.names[top_class_id]

    print("Image:", image_path)
    print("Predicted mudra:", class_name)
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()