import sys
import re
import pandas as pd
from pathlib import Path

def extract_label(text):
    if pd.isna(text):
        return None

    # Look for **Judgment:** or **Answer:** followed by a label
    match = re.search(r"\*\*(Judgment|Answer):\*\*\s*(\d+\.?\s*)?([-a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        return match.group(3).strip().lower()

    # Fallback: "The statement is [label]"
    match = re.search(r"The statement is\s+(\d+\.?\s*)?([-a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip().lower()

    return None

def main(csv_path):
    input_path = Path(csv_path)
    output_path = input_path.with_name(input_path.stem + "_extracted.csv")

    df = pd.read_csv(input_path)

    if "model_output" not in df.columns:
        print("Error: 'model_output' column not found in the CSV.")
        return

    df["extracted_label"] = df["model_output"].apply(extract_label)
    df.to_csv(output_path, index=False)
    print(f"Extracted labels saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_labels.py <csv_file>")
    else:
        main(sys.argv[1])
