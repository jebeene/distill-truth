#!/usr/bin/env python3
import argparse
from archive.openrouter.pipelines.chat_pipeline import chat
from archive.openrouter.utils.config_loader import load_config, validate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Prompt to send to the model")
    parser.add_argument("--model", required=True, help="Model key or full model name. See open router_config.yaml for available models.")
    args = parser.parse_args()

    model_config = load_config("openrouter/configs/openrouter_config.yaml")
    model_config["model_selection"] = validate_model(model_config, args.model)

    response = chat(args.prompt, config=model_config)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()