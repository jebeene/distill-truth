#!/usr/bin/env python3
import argparse
from archive.openrouter.evaluation.evaluate import evaluate_liar_dataset
from archive.openrouter.utils.config_loader import load_config, validate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key or full model name. See open router_config.yaml for available models.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    config = load_config("openrouter/configs/openrouter_config.yaml")
    config["model_selection"] = validate_model(config, args.model)

    evaluate_liar_dataset(
        config=config,
        dataset_name="liar",
        split=args.split,
        limit=args.limit
    )

if __name__ == "__main__":
    main()