from openrouter.prompts.prompts import BASE_PROMPT, ASSISTANT_PROMPT

def build_user_prompt(statement: str):
    return BASE_PROMPT.format(statement=statement)

def build_assistant_prompt(statement: str):
    return ASSISTANT_PROMPT.format(statement=statement)