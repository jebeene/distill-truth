import argparse
from data import load_liar_dataset
from loader import load_generation_model
from evaluate import evaluate_generation
from config import DEFAULT_MODEL_NAME

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--cache-path", type=str, default="generation_results.jsonl")
    args = parser.parse_args()

    # Load dataset and model
    dataset = load_liar_dataset()
    tokenizer, model, device = load_generation_model(args.model)

    # Run evaluation
    results = evaluate_generation(
        model,
        tokenizer,
        dataset["test"],
        device,
        max_samples=args.max_samples,
        cache_path=args.cache_path
    )

    print("Final Evaluation:", results)
