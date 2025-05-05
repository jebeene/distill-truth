import sys

import requests

from archive.openrouter.utils.pretty import log_error


def query_openrouter(prompt, config=None):
    if config is None:
        raise ValueError("Config for OpenRouter API not provided")

    if not config['model_selection']:
        raise ValueError("Model not specified.")

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config['model_selection'],
        "messages": [
            { "role": "user", "content": prompt },
            { "role": "assistant", "content": "" } # can tweak this later if we want to change user prompt
        ]
    }

    url = f"{config['base_url']}/chat/completions"

    response = None
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        log_error("Request to OpenRouter failed.")
        log_error("Double-check that your API key is correct.")
        log_error("If the key is correct, the model may be temporarily down.")
        log_error(f"HTTP Status: {response.status_code} - {response.reason}")
        sys.exit(1)

    result = response.json()

    # Print for debugging
    # log_info(result)

    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return text.strip()