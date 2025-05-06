import pandas as pd
import sys
import re
from pathlib import Path

LABELS = sorted([
    "pants-fire", "barely-true", "half-true", "mostly-true", "false", "true"
], key=lambda x: -len(x))  # longest first

def preprocess(text):
    text = text.lower()
    # Convert "not false" to "true", and "not true" to "false"
    text = re.sub(r'\bnot\s+false\b', 'true', text)
    text = re.sub(r'\bnot\s+true\b', 'false', text)
    return text

def extract_label(text):
    if pd.isna(text):
        return None

    text = preprocess(text)

    # 1. Word before "claim"
    match1 = re.search(r'\b([\w-]+)\s+claim\b', text)
    if match1 and match1.group(1) in LABELS:
        return match1.group(1)

    # 2. Word after "statement is"
    match2 = re.search(r'\bstatement is\s+([\w-]+)\b', text)
    if match2 and match2.group(1) in LABELS:
        return match2.group(1)

    # 3. Word after "claim is"
    match3 = re.search(r'\bclaim is\s+([\w-]+)\b', text)
    if match3 and match3.group(1) in LABELS:
        return match3.group(1)

    # 4. Match: statement "..." is {label}
    match4 = re.search(r'statement\s+"[^"]*"\s+is\s+([\w-]+)\b', text)
    if match4 and match4.group(1) in LABELS:
        return match4.group(1)

    # 5. "the claim is {label}"
    match5 = re.search(r'\bthe claim is\s+(false|half-true|mostly-true|true|barely-true|pants-fire)\b', text)
    if match5:
        return match5.group(1)

    # 6. "therefore, the claim is {label}"
    match6 = re.search(r'\btherefore,? the claim is\s+(false|half-true|mostly-true|true|barely-true|pants-fire)\b',
                       text)
    if match6:
        return match6.group(1)

    return None

def main(csv_path):
    input_path = Path(csv_path)
    output_path = input_path.with_name(input_path.stem + "_labeled.csv")

    df = pd.read_csv(input_path)

    if "cleaned_output" not in df.columns:
        print("Error: CSV must contain 'cleaned_output' column. Run remove_prompt.py first.")
        return

    df["pred_label"] = df["cleaned_output"].apply(extract_label)

    df.to_csv(output_path, index=False)
    print(f"Saved labeled output to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_labels.py <cleaned_csv_file>")
    else:
        main(sys.argv[1])