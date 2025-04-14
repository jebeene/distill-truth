import os
import re
import json
import logging
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import CLASS_LABELS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def build_prompt(statement: str) -> str:
    return (
        "Classify the following political statement into one of these labels:\n"
        "0 = false\n"
        "1 = half-true\n"
        "2 = mostly-true\n"
        "3 = true\n"
        "4 = barely-true\n"
        "5 = pants-fire\n"
        "Respond with only the number (0–5) corresponding to the label. Do NOT respond with anything other than the number.\n"
        "You are limited to producing one token (the label number). If you produce anything else, you will fail.\n"
        f"Statement: {statement}\n"
        "Label number:"
    )

def extract_label(text: str) -> str | None:
    if not text:
        return None

    cleaned = text.strip()
    digits = ''.join(c for c in cleaned if c.isdigit())

    if not digits:
        return None

    try:
        label = int(digits)
        return label if 0 <= label <= 5 else None
    except ValueError:
        return None

def load_cached_predictions(cache_path: str) -> Dict[int, Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return {entry["index"]: entry for entry in map(json.loads, f)}


def append_prediction(cache_path: str, entry: Dict[str, Any]) -> None:
    with open(cache_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def compute_metrics(labels, preds):
    return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average='macro'),
            "classification_report": classification_report(labels, preds, zero_division=0)
            }


def evaluate_generation(
        model,
        tokenizer,
        dataset,
        device: str,
        max_samples: int = 100,
        cache_path: str = "generation_results.jsonl",
        ) -> Dict[str, Any]:
    logger.info("Starting generation evaluation...")
    cached = load_cached_predictions(cache_path)
    predictions = []

    for idx in tqdm(range(min(len(dataset), max_samples)), desc="Evaluating"):
        if idx in cached:
            predictions.append(cached[idx])
            continue

        statement = dataset["statement"][idx]
        true_label = dataset["label"][idx]
        prompt = build_prompt(statement)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id
                )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        predicted_label = extract_label(output_text)
        entry = {
                "index": idx,
                "statement": statement,
                "true_label": true_label,
                "predicted": predicted_label,
                "generated_text": output_text,
                }
        append_prediction(cache_path, entry)
        predictions.append(entry)


    labels = []
    preds = []
    for entry in predictions:
        if entry["predicted"] is not None:
            labels.append(entry["true_label"])
            preds.append(entry["predicted"])

    if not labels:
        logger.warning("No valid predictions to evaluate.")
        return {
            "accuracy": None,
            "f1_macro": None,
            "classification_report": "No valid predictions."
        }

    metrics = compute_metrics(labels, preds)

    logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_macro']:.3f}, Classification Report: {metrics['classification_report']}")
    return metrics
