# main.py
import argparse
from data import load_liar_dataset
from loader import load_model
from evaluate import evaluate
from evaluate import generate_explanations, compare_explanations
from config import DEFAULT_MODEL_NAME
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name or path")
    args = parser.parse_args()

    dataset = load_liar_dataset()
    tokenizer, model, device = load_model(args.model)

    print("Evaluating classification...")
    results = evaluate(model, tokenizer, dataset["test"], device)
    print("Classification Results:", results)

    print("Evaluating explanations...")
    generated = generate_explanations(args.model, dataset["test"][:100])
    bertscore = compare_explanations(generated, dataset["test"]["justification"][:100])
    print("Explainability Results:", bertscore)
