import sys
import os
import csv
import torch
import pandas as pd
from math import ceil
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from PROMPTS import PE_PROMPT_TEMPLATE, PE_SYSTEM_PROMPT

# -------------------- Config --------------------
if len(sys.argv) != 3:
    print("Usage: python evaluate_proxy.py <model_dir_or_name> <justification_csv>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
CSV_PATH = sys.argv[2]

BATCH_SIZE = 16
USE_CUDA = torch.cuda.is_available()
USE_MPS = torch.backends.mps.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "mps" if USE_MPS else "cpu")

# -------------------- Load Model --------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if USE_CUDA else torch.float32)
model.to(DEVICE)
model.eval()

# -------------------- Load Data --------------------

df = pd.read_csv(CSV_PATH)
if "masked_explanation" not in df.columns:
    print("Error: CSV must contain an 'masked_explanation' column.")
    sys.exit(1)

# -------------------- Build Prompts --------------------

def build_proxy_prompt(expl):
    return f"{PE_SYSTEM_PROMPT}\n\n" + PE_PROMPT_TEMPLATE.format(EXPLANATION=expl.strip())

df["proxy_prompt"] = df["masked_explanation"].astype(str).apply(build_proxy_prompt)

# -------------------- Evaluate --------------------

basename = os.path.basename(MODEL_NAME.rstrip("/"))
outfile = f"proxy_eval_{basename.replace('/', '_')}.csv"

with open(outfile, "w", encoding="utf-8", newline='') as outpath:
    writer = csv.writer(outpath, delimiter=",", quoting=csv.QUOTE_ALL)
    writer.writerow(["masked_explanation", "true_label", "proxy_prompt", "proxy_output"])

    num_batches = ceil(len(df) / BATCH_SIZE)
    for i in tqdm(range(num_batches), desc=f"Proxy evaluating {basename}"):
        batch_df = df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        prompts = batch_df["proxy_prompt"].tolist()

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for row, output, prompt in zip(batch_df.itertuples(), decoded_outputs, prompts):
            writer.writerow([
                row.maasked_explanation.replace("\n", " ").strip(),
                getattr(row, "true_label", ""),  # optional
                prompt.replace("\n", " ").strip(),
                output.replace("\n", " ").strip()
            ])

print(f"Saved proxy evaluation results to: {outfile}")