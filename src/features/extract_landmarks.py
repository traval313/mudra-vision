from pathlib import Path
import argparse

import pandas as pd

from src.features.hand_landmarks import (
    FEATURE_COLUMNS,
    create_hands_detector,
    extract_landmark_features,
    iter_image_paths,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_ROOT = ROOT / "images"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "features"


def display_path(image_path):
    try:
        return str(image_path.relative_to(ROOT))
    except ValueError:
        return str(image_path)


def extract_split(split_dir, output_csv):
    rows = []
    skipped = 0

    with create_hands_detector() as hands:
        for image_path in iter_image_paths(split_dir):
            label = image_path.parent.name
            features = extract_landmark_features(image_path, hands)

            if features is None:
                skipped += 1
                continue

            row = {
                "image_path": display_path(image_path),
                "label": label,
            }
            row.update(dict(zip(FEATURE_COLUMNS, features)))
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_path", "label", *FEATURE_COLUMNS]
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    if skipped:
        print(f"Skipped {skipped} images without detectable hand landmarks.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks for the mudra image dataset."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Dataset root containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where landmark CSV files will be written.",
    )
    args = parser.parse_args()

    for split in ("train", "val"):
        split_dir = args.image_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing dataset split: {split_dir}")

        extract_split(split_dir, args.output_dir / f"{split}_landmarks.csv")


if __name__ == "__main__":
    main()
