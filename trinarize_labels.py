import sys
import pandas as pd
from pathlib import Path

# Define the trinary mapping
TRINARY_MAP = {
    "false": "false",
    "pants-fire": "false",
    "barely-true": "neutral",
    "half-true": "neutral",
    "mostly-true": "true",
    "true": "true"
}

def trinarize_label(label):
    if pd.isna(label):
        return None
    label = label.strip().lower()
    return TRINARY_MAP.get(label, None)

def main(csv_path):
    input_path = Path(csv_path)
    output_path = input_path.with_name(input_path.stem + "_trinary.csv")

    df = pd.read_csv(input_path)

    if "extracted_label" not in df.columns or "true_label" not in df.columns:
        print("Error: 'true_label' or 'extracted_label' column not found in the CSV.")
        return

    df["true_label"] = df["true_label"].apply(trinarize_label)
    df["extracted_label"] = df["extracted_label"].apply(trinarize_label)

    df.to_csv(output_path, index=False)
    print(f"Trinary-labeled results saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trinarize_labels.py <extracted_csv_file>")
    else:
        main(sys.argv[1])
