import pandas as pd
import sys
from pathlib import Path

def remove_prompt_from_output(prompt, model_output):
    if pd.isna(prompt) or pd.isna(model_output):
        return None

    # Find the first occurrence of the prompt in the model output and remove it
    index = model_output.find(prompt)
    if index != -1:
        return model_output[index + len(prompt):].strip()
    return model_output.strip()  # fallback: return full output if prompt not found

def main(csv_path):
    input_path = Path(csv_path)
    output_path = input_path.with_name(input_path.stem + "_cleaned.csv")

    df = pd.read_csv(input_path)

    if "prompt" not in df.columns or "model_output" not in df.columns:
        print("Error: CSV must contain 'prompt' and 'model_output' columns.")
        return

    df["cleaned_output"] = df.apply(
        lambda row: remove_prompt_from_output(row["prompt"], row["model_output"]),
        axis=1
    )

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned output to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_prompt.py <csv_file>")
    else:
        main(sys.argv[1])