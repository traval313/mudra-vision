from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "images"

MODEL_NAME = "yolo11n-cls.pt"

def main():
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATA_DIR),
        epochs=10,
        imgsz=224,
        batch=16,
        project=str(ROOT / "runs"),
        name="mudra_yolo_cls_baseline",
        device="cpu",
    )

    print("Training complete.")
    print(results)


if __name__ == "__main__":
    main()