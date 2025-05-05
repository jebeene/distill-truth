import pandas as pd
import sys
import re
from pathlib import Path

LABELS = sorted([
    "pants-fire", "barely-true", "half-true", "mostly-true", "false", "true"
], key=lambda x: -len(x))

def extract_all_labels(text):
    if pd.isna(text):
        return None

    text = text.lower()
    matches = []
    for label in LABELS:
        pattern = rf"(?<![-\w]){re.escape(label)}(?![-\w])"
        if re.search(pattern, text):
            matches.append(label)
    return ", ".join(matches) if matches else None

def main(csv_path):
    input_path = Path(csv_path)
    output_path = input_path.with_name(input_path.stem + "_labeled.csv")

    df = pd.read_csv(input_path)

    if "cleaned_output" not in df.columns:
        print("Error: CSV must contain 'cleaned_output' column. Run remove_prompt.py first.")
        return

    df["pred_label"] = df["cleaned_output"].apply(extract_all_labels)

    df.to_csv(output_path, index=False)
    print(f"Saved labeled output to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_labels.py <cleaned_csv_file>")
    else:
        main(sys.argv[1])