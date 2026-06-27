from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs" / "evaluation" / "sample_predictions.csv"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "Run src/evaluation/evaluate_model.py before running this script."
        )

    mistakes = []

    with open(CSV_PATH, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row["correct"] == "False":
                mistakes.append(row)

    print(f"Found {len(mistakes)} mistakes.\n")

    for mistake in mistakes[:20]:
        print("Image:", mistake["image_path"])
        print("True:", mistake["true_label"])
        print("Predicted:", mistake["predicted_label"])
        print("Confidence:", mistake["confidence"])
        print("-" * 40)


if __name__ == "__main__":
    main()