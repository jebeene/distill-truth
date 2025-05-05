from archive.openrouter.models.openrouter import query_openrouter

def run_model(prompt, config):
    """
    Run the model with the given prompt using the OpenRouter API.

    :param prompt: The text prompt.
    :param config: Configuration dictionary from YAML.
    :return: The model's response.
    """
    return query_openrouter(prompt, config=config)