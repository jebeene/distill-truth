import time
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline
from bert_score import score

def evaluate(model, tokenizer, dataset, device):
    inputs = tokenizer(dataset["statement"], padding=True, truncation=True, return_tensors="pt").to(device)
    labels = torch.tensor(dataset["label"]).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()

    preds = outputs.logits.argmax(dim=-1)

    acc = accuracy_score(labels.cpu(), preds.cpu())
    f1 = f1_score(labels.cpu(), preds.cpu(), average="macro")
    latency = (end - start) / len(labels)
    memory = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0

    return {"accuracy": acc, "f1": f1, "latency": latency, "gpu_mem_MB": memory}

def generate_explanations(model_name, dataset):
    classifier = pipeline("text2text-generation", model=model_name)
    prompts = [f"Classify the statement and explain: '{s}'" for s in dataset["statement"]]
    return [classifier(p)[0]["generated_text"] for p in prompts]

def compare_explanations(generated, ground_truth):
    P, R, F1 = score(generated, ground_truth, lang="en")
    return {"bertscore_P": P.mean().item(), "bertscore_R": R.mean().item(), "bertscore_F1": F1.mean().item()}

