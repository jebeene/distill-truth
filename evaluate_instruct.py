import sys
import os
import csv
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import PROMPTS
from math import ceil

# -------------------- Configuration --------------------

if len(sys.argv) != 4:
    print("Usage: python evaluate_instruct.py <model_dir_or_name> <test_file.tsv> <mode>")
    print("mode: zero-shot | one-shot | few-shot-k (e.g., few-shot-3)")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
TSV_PATH = sys.argv[2]
MODE = sys.argv[3]

BATCH_SIZE = 16
USE_CUDA = torch.cuda.is_available()
USE_MPS = torch.backends.mps.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "mps" if USE_MPS else "cpu")

# -------------------- Data Prep --------------------

df = pd.read_csv(TSV_PATH, sep="\t", header=None)
df.columns = [
    "id", "labels", "statement", "context", "speaker",
    "job", "state", "party_affiliation", "barely_true_counts",
    "false_counts", "half_true_counts", "mostly_true_counts",
    "pants_on_fire_counts", "source"
]

FEW_SHOT_TEMPLATE = PROMPTS.FEW_SHOT_PROMPT_TEMPLATE
CLASSIFICATION_OPTIONS = PROMPTS.CLASSIFICATION_OPTIONS
SYSTEM_PROMPT = PROMPTS.CLASSIFICATION_SYSTEM_PROMPT.format(CLASSIFICATION_OPTIONS=CLASSIFICATION_OPTIONS)

# Few-shot helper

def build_few_shot_examples(train_df, k):
    few_shots = []
    for _, row in train_df.sample(n=k).iterrows():
        few_shots.append(FEW_SHOT_TEMPLATE.format(STATEMENT=row['statement'], LABEL=row['labels']))
    return "\n\n".join(few_shots)

# Prompt builder

def build_prompt(row, few_shot_text=None):
    classification_prompt = PROMPTS.STATEMENT_CLASSIFICATION_PROMPT.format(
        STATEMENT=row["statement"],
        IS_INCLUDE_SPEAKER=PROMPTS.IS_INCLUDE_SPEAKER.format(SPEAKER=row.get("speaker", "Unknown")),
        IS_INCLUDE_PARTY=PROMPTS.IS_INCLUDE_PARTY.format(PARTY_AFFILIATION=row.get("party_affiliation", "Unknown")),
        IS_INCLUDE_EXPLANATION=PROMPTS.IS_INCLUDE_EXPLANATION,
        CLASSIFICATION_OPTIONS=CLASSIFICATION_OPTIONS
    )
    prompt = SYSTEM_PROMPT + "\n\n"
    if few_shot_text:
        prompt += few_shot_text + "\n\n"
    prompt += classification_prompt
    return prompt

# -------------------- Model Loading --------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if USE_CUDA else torch.float32)
model.to(DEVICE)
model.eval()

# -------------------- Few-Shot Construction --------------------

few_shot_text = None
if MODE.startswith("few-shot"):
    try:
        k = int(MODE.split("-")[-1])
        few_shot_text = build_few_shot_examples(df, k)
    except:
        raise ValueError("Invalid few-shot-k format. Use few-shot-3 for example.")
elif MODE == "one-shot":
    few_shot_text = build_few_shot_examples(df, 1)

# -------------------- Evaluation --------------------

basename = os.path.basename(MODEL_NAME.rstrip("/"))
outfile = f"results_{basename.replace('/', '_')}_{MODE}.csv"
with open(outfile, "w", encoding="utf-8", newline='') as outpath:
    writer = csv.writer(outpath, delimiter=",", quoting=csv.QUOTE_ALL)
    writer.writerow(["statement", "true_label", "prompt", "model_output"])

    num_batches = ceil(len(df) / BATCH_SIZE)
    for i in tqdm(range(num_batches), desc=f"Evaluating {basename} ({MODE})"):
        batch_df = df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        prompts = [build_prompt(row, few_shot_text) for _, row in batch_df.iterrows()]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for row, response, prompt in zip(batch_df.itertuples(), decoded_outputs, prompts):
            writer.writerow([
                row.statement.replace("\n", " ").strip(),
                row.labels,
                prompt.replace("\n", " ").strip(),
                response.replace("\n", " ").strip()
            ])

print(f"Saved results to {outfile}")
