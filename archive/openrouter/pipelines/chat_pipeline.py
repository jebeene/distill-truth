from archive.openrouter.prompts.prompt_builder import build_user_prompt
from archive.openrouter.models.model_runner import run_model


def chat(prompt_input, config):
    """
    Chat pipeline that processes user input and returns the model's response.

    :param prompt_input: User input string.
    :param config: configuration
    :return: The model's response.
    """
    final_prompt = build_user_prompt(prompt_input)
    response = run_model(final_prompt, config)
    return response