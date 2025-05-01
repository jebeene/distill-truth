import sys
import pandas as pd
from pathlib import Path

# Define the binary mapping
BINARY_MAP = {
    "false": "false",
    "pants-fire": "false",
    "barely-true": "false",
    "half-true": "true",
    "mostly-true": "true",
    "true": "true"
}

def binarize_label(label):
    if pd.isna(label):
        return None
    label = label.strip().lower()
    return BINARY_MAP.get(label, None)

def main(csv_path):
    input_path = Path(csv_path)
    output_path = input_path.with_name(input_path.stem + "_binary.csv")

    df = pd.read_csv(input_path)

    if "extracted_label" not in df.columns or "true_label" not in df.columns:
        print("Error: 'true_label' or 'extracted_label' column not found in the CSV.")
        return

    df["true_label"] = df["true_label"].apply(binarize_label)
    df["extracted_label"] = df["extracted_label"].apply(binarize_label)

    df.to_csv(output_path, index=False)
    print(f"Binary-labeled results saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python binarize_labels.py <extracted_csv_file>")
    else:
        main(sys.argv[1])
