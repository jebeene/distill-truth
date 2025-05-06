import pandas as pd
import sys, re
from pathlib import Path

CLASSIFICATION_OPTIONS = """0. false
1. half-true
2. mostly-true
3. true
4. barely-true
5. pants-fire"""

label_to_number = {
    'false': 0,
    'half-true': 1,
    'mostly-true': 2,
    'true': 3,
    'barely-true': 4,
    'pants-fire': 5,
    'label_not_found': -1,
}

def extract_masked_explanation(output, pred_label):
    if not isinstance(pred_label, str) or pd.isna(pred_label):
        return output

    output_lower = output.lower()
    label_lower = pred_label.lower()

    label_variants = [
        label_lower,
        f"{label_to_number[label_lower]}. " + label_lower,
        f"{label_to_number[label_lower]}." + label_lower,
        label_lower.replace("-", " "),
        label_lower.replace(" ", "-")
    ]

    masked_output = output
    for variant in label_variants:
        masked_output = re.sub(
            rf'\b{re.escape(variant)}\b',
            '___',
            masked_output,
            flags=re.IGNORECASE
        )

    return masked_output

def main(csv_path):
    input_path = Path(csv_path)
    output_path = input_path.with_name(input_path.stem + "_explained.csv")

    df = pd.read_csv(input_path)

    if "cleaned_output" not in df.columns or "extracted_label" not in df.columns:
        print("Error: CSV must contain 'cleaned_output' and 'extracted_label' columns.")
        return

    df["masked_explanation"] = df.apply(
        lambda row: extract_masked_explanation(row["cleaned_output"], row['extracted_label']),
        axis=1
    )

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned output to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_prompt.py <csv_file>")
    else:
        main(sys.argv[1])