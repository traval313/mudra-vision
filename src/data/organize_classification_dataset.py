from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "dataset.yaml"

TRAIN_DIR = ROOT / "images" / "train"
VAL_DIR = ROOT / "images" / "val"

def load_class_names():
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


def class_from_filename(filename: str, class_names: list[str]) -> str | None:
    """
    Finds which mudra class name appears at the start of a filename.

    Example:
    Pathaka_001.jpg -> Pathaka
    Alapadmam_something.jpg -> Alapadmam
    """
    clean_name = filename.lower()

    for class_name in class_names:
        if clean_name.startswith(class_name.lower()):
            return class_name

    return None

def organize_split(split_dir: Path, class_names: list[str]):
    image_paths = [
        p for p in split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    print(f"Found {len(image_paths)} images in {split_dir}")

    moved = 0
    skipped = 0

    for image_path in image_paths:
        class_name = class_from_filename(image_path.name, class_names)

        if class_name is None:
            print(f"Could not infer class for: {image_path.name}")
            skipped += 1
            continue

        target_dir = split_dir / class_name
        target_dir.mkdir(exist_ok=True)

        target_path = target_dir / image_path.name
        shutil.move(str(image_path), str(target_path))
        moved += 1

    print(f"Moved {moved} images into class folders.")
    print(f"Skipped {skipped} images.")


def main():
    class_names = load_class_names()

    organize_split(TRAIN_DIR, class_names)
    organize_split(VAL_DIR, class_names)

    print("Dataset organization complete.")


if __name__ == "__main__":
    main()