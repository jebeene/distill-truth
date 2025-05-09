import csv
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# -------------------- Device Setup --------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Model + Tokenizer --------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[STATEMENT]", "[LABEL]", "[JUSTIFICATION]"]
})
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Resize for special tokens
model.to(device)

# -------------------- LoRA Setup --------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# -------------------- Load TSV Data --------------------
tsv_path = "datasets/train.tsv"
label_map = {
    "pants on fire": "Pants on Fire",
    "false": "False",
    "barely true": "Barely True",
    "half true": "Half True",
    "mostly true": "Mostly True",
    "true": "True"
}
examples = []

with open(tsv_path, "r") as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        if len(row) < 3:
            continue
        label_raw = row[1].strip().lower()
        statement = row[2].strip()
        context = row[3].strip() if len(row) > 3 else ""
        if label_raw not in label_map:
            continue
        label = label_map[label_raw]
        text = f"[STATEMENT] {statement} [LABEL] {label} [JUSTIFICATION] {context}"
        examples.append({"text": text})

dataset = Dataset.from_list(examples)

# -------------------- Tokenization --------------------
def tokenize(example):
    full_text = example["text"]
    label_start = full_text.index("[LABEL]") + len("[LABEL]")
    tokenized_input = tokenizer(full_text, truncation=True, padding="max_length", max_length=128)
    labels = [-100] * len(tokenized_input["input_ids"])
    label_segment = full_text[label_start:].strip()
    label_ids = tokenizer(label_segment, truncation=True, padding="max_length", max_length=128)["input_ids"]

    for i in range(len(label_ids)):
        if label_start + i < len(labels):
            labels[label_start + i] = label_ids[i]

    tokenized_input["labels"] = labels
    return tokenized_input

tokenized = dataset.map(tokenize, remove_columns=["text"])
tokenized.set_format(type="torch")

# Split into train and validation sets
split = tokenized.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

# -------------------- Data Collator --------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------- Training Arguments --------------------

training_args = TrainingArguments(
    output_dir="./tinyllama-liar-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=10,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    report_to="none"
)

# -------------------- Trainer --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Ensure you have a validation dataset
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

# -------------------- Save --------------------
model.save_pretrained("./tinyllama-liar-lora")
tokenizer.save_pretrained("./tinyllama-liar-lora")