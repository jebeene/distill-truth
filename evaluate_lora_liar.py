import torch
import re
import csv
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------- Config --------------------
model_path = "./tinyllama-liar-lora"
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tsv_path = "datasets/test.tsv"

# -------------------- Load Model + Tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(model_path)
base = AutoModelForCausalLM.from_pretrained(base_model_name)
base.resize_token_embeddings(len(tokenizer))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = PeftModel.from_pretrained(base, model_path)
model.to(device).eval()

# -------------------- Label Mappings --------------------
label_names = [
    "False", "Half True", "Mostly True",
    "True", "Barely True", "Pants on Fire"
]
label2id = {label.lower(): i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

# -------------------- Prompt + Extraction --------------------
def make_prompt(statement: str) -> str:
    return (
        "Below is a claim.  Determine its veracity by choosing **exactly one** of the six labels.\n\n"
        f"Claim: {statement}\n\n"
        "Options:\n"
        "  - False\n"
        "  - Barely True\n"
        "  - Half True\n"
        "  - Mostly True\n"
        "  - True\n"
        "  - Pants on Fire\n\n"
        "Answer (just the label name):"
    )

def extract_label(text):
    text = text.lower().strip()
    # Remove leading quotes, dashes, bullets, etc.
    text = re.sub(r'^[\'"“”‘’\-–•]+\s*', '', text)
    for lab in label_names:
        if text.startswith(lab.lower()):
            return lab
    return "Unknown"

# -------------------- Load TSV and Evaluate --------------------
true_ids = []
pred_ids = []
label_counter = Counter()
unknown_counter = 0
skipped = 0  # track skipped rows

with open(tsv_path, "r") as file:
    reader = list(csv.reader(file, delimiter="\t"))
    for idx, row in enumerate(tqdm(reader, desc="Evaluating TSV", dynamic_ncols=True)):
        if len(row) < 3:
            continue
        raw_label = row[1].strip().lower()
        normalized_label = raw_label.replace("-", " ").strip()
        statement = row[2].strip()
        justification = row[3].strip() if len(row) > 3 else ""

        if normalized_label not in label2id:
            skipped += 1
            continue

        prompt = make_prompt(statement)

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        tail = decoded[len(prompt):].strip()
        tail = tail.split("\n", 1)[0].strip()
        pred_label = extract_label(tail)
        pred_justif = tail.split("[JUSTIFICATION]", 1)[-1].strip() if "[JUSTIFICATION]" in tail else ""

        if idx < 10:  # Debug output
            print("\n--- SAMPLE ---")
            print(f"Prompt: {prompt}")
            print(f"Generated: {tail}")
            print(f"Predicted label: {pred_label}")
            print(f"True label: {normalized_label}")
            print(f"\n--- FULL DECODED ---\n{decoded}\n--- END ---")

        if pred_label != "Unknown":
            pred_ids.append(label2id[pred_label.lower()])
            true_ids.append(label2id[normalized_label])
            label_counter[pred_label] += 1
        else:
            unknown_counter += 1

# -------------------- Report --------------------
print("\nPrediction Label Frequencies:")
for label in label_names:
    print(f"{label}: {label_counter.get(label, 0)}")
print(f"Unknown predictions: {unknown_counter}")
print(f"Skipped rows due to label mismatch: {skipped}")

print("\nClassification Report:")
labels = list(range(len(label_names)))
print(classification_report(true_ids, pred_ids, labels=labels, target_names=label_names, zero_division=0))
print(f"Accuracy: {accuracy_score(true_ids, pred_ids):.3f}")
print(f"Macro F1: {f1_score(true_ids, pred_ids, average='macro'):.3f}")