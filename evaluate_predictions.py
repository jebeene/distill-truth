import sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

# Set of valid binary labels
BINARY_LABELS = {"true", "false"}

# Multiclass labels for original LIAR dataset
MULTICLASS_LABELS = {
    "false", "pants-fire", "barely-true",
    "half-true", "mostly-true", "true"
}

def evaluate(true_labels, pred_labels, is_binary):
    average_type = "binary" if is_binary else "macro"
    pos_label = "true" if is_binary else None
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average=average_type, pos_label=pos_label)
    return acc, f1

def main(csv_path):
    input_path = Path(csv_path)
    output_path = input_path.with_name(input_path.stem + "_metrics.txt")

    df = pd.read_csv(input_path)

    if "true_label" not in df.columns or "extracted_label" not in df.columns:
        print("Error: Required columns 'true_label' and 'extracted_label' not found.")
        return

    y_true = df["true_label"].astype(str).str.strip().str.lower()
    y_pred = df["extracted_label"].astype(str).str.strip().str.lower()

    # Determine if this is a binary or multiclass task
    unique_labels = set(y_true.unique()) | set(y_pred.unique())
    is_binary = unique_labels <= BINARY_LABELS

    # Filter valid labels for the selected mode
    label_set = BINARY_LABELS if is_binary else MULTICLASS_LABELS
    valid_rows = y_true.isin(label_set) & y_pred.isin(label_set)
    y_true = y_true[valid_rows]
    y_pred = y_pred[valid_rows]

    if len(y_true) == 0:
        print("No valid labels found for evaluation.")
        with open(output_path, "w") as f:
            f.write("Accuracy: nan\n")
            f.write("F1 Score: 0.0000\n")
    else:
        acc, f1 = evaluate(y_true, y_pred, is_binary)
        with open(output_path, "w") as f:
            f.write(f"Label type: {'binary' if is_binary else 'multiclass'}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        print(f"Metrics saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_predictions.py <csv_file>")
    else:
        main(sys.argv[1])
