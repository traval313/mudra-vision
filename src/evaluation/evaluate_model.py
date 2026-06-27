from pathlib import Path
from ultralytics import YOLO
import csv 
import random 

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "runs" / "mudra_yolo_cls_baseline" / "weights" / "best.pt"
VAL_DIR = ROOT / "images" / "val"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_CSV = OUTPUT_DIR / "sample_predictions.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_class_folders():
    return sorted([p for p in VAL_DIR.iterdir() if p.is_dir()])


def get_sample_images(class_folder: Path, samples_per_class: int = 3):
    images = [
        p for p in class_folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) <= samples_per_class:
        return images

    return random.sample(images, samples_per_class)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    rows = []
    correct = 0
    total = 0

    class_folders = get_class_folders()

    for class_folder in class_folders:
        true_label = class_folder.name
        sample_images = get_sample_images(class_folder)

        for image_path in sample_images:
            results = model.predict(source=str(image_path), verbose=False)
            result = results[0]

            top_class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            predicted_label = result.names[top_class_id]

            is_correct = predicted_label == true_label

            rows.append({
                "image_path": str(image_path.relative_to(ROOT)),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 4),
                "correct": is_correct,
            })

            correct += int(is_correct)
            total += 1

    accuracy = correct / total if total else 0

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = [
            "image_path",
            "true_label",
            "predicted_label",
            "confidence",
            "correct",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluated {total} images.")
    print(f"Correct: {correct}")
    print(f"Sample accuracy: {accuracy:.2%}")
    print(f"Saved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()