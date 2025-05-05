import sys
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import PROMPTS

# Usage
if len(sys.argv) != 3:
    print("Usage: python distill-truth.py <model_name> <test_file.tsv>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
TSV_PATH = sys.argv[2]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# LIAR-style columns
STATEMENT_COLUMN = "statement"
CONTEXT_COLUMN = "context"
LABEL_COLUMN = "labels"

# Load TSV
df = pd.read_csv(TSV_PATH, sep="\t")

# Combine prompt with statement + context
def build_prompt(row):
    base = f"{PROMPTS.CLASSIFICATION_SYSTEM_PROMPT} {row[STATEMENT_COLUMN]}"

    return base

df["prompt"] = df.apply(build_prompt, axis=1)

# Convert to HF Dataset
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)

def preprocess(examples):
    return tokenizer(examples["prompt"], truncation=True, padding="max_length")

dataset = dataset.map(preprocess, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", LABEL_COLUMN])

accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    args=TrainingArguments(
        output_dir="../results",
        per_device_eval_batch_size=1,
        disable_tqdm=False,
        logging_dir="./logs",
        logging_steps=10,
    ),
)

results = trainer.evaluate(eval_dataset=dataset)
# Get raw predictions
predictions = trainer.predict(dataset)
logits = predictions.predictions
predicted_labels = torch.argmax(torch.tensor(logits), dim=-1).numpy()

# Save to CSV
output_df = pd.DataFrame({
    "id": df.index,
    "statement": df[STATEMENT_COLUMN],
    "context": df[CONTEXT_COLUMN],
    "true_label": df[LABEL_COLUMN],
    "predicted_label": predicted_labels,
})

output_df.to_csv("model_outputs.csv", index=False)
print("Saved predictions to model_outputs.csv")
print(results)