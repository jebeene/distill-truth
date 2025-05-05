import sys

from dotenv import load_dotenv
import yaml
import os

from archive.openrouter.utils.confirm import ask_approval

load_dotenv()

def load_config(path):
    """
    Load a YAML configuration file and replace environment variable references.

    :param path: Path to the YAML config file.
    :return: A dictionary with configuration parameters.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    # Replace any values starting with 'env:' with the corresponding environment variable.
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("env:"):
            env_var = value[4:]
            config[key] = os.getenv(env_var)

    return config

def validate_model(config, model_alias):
    """
    Set the model configuration based on the provided model name.

    :param config: The configuration dictionary.
    :param model_name: The name of the model to set.
    :return: The updated configuration dictionary.
    """
    available_models = config.get("available_models", {})

    if model_alias not in available_models:
        raise ValueError(f"Model '{model_alias}' not found in available models (openrouter_config.yaml).")

    model_name = available_models[model_alias]

    if "free" not in model_name:
        if not ask_approval(f"Do you want to proceed with a paid model '{model_name}'? (y/n)"):
            print("Exiting...")
            sys.exit(0)

    return model_name
