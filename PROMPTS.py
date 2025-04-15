# Classification Prompts
CLASSIFICATION_SYSTEM_PROMPT = """You are a fact-checking assistant trained to verify political and non-political claims. You carefully analyze a given statement and determine whether it is {CLASSIFICATION_OPTIONS} based on reasoning and evidence. Your responses are concise, well-justified, and avoid speculation."""

IS_INCLUDE_SPEAKER = "The following speaker made the above statement: {SPEAKER}."
IS_INCLUDE_PARTY = "Additionally, the above speaker belongs to the party: {PARTY_AFFILIATION}"
STATEMENT_CLASSIFICATION_PROMPT = """Evaluate the truthfulness of the following statement:
"{STATEMENT}"

{IS_INCLUDE_SPEAKER} {IS_INCLUDE_PARTY}

What is your judgment? Classify the claim as one of the following: {CLASSIFICATION_OPTIONS}."""

JUSTIFICATION_PROMPT = """Now that you have classified the statement. Provide an explation to the user as to why you think the statement is {CLASSIFIED_LABEL} and what parts of the statement makes you believe so."""

# Proxy Evaluation of Justification Prompts
PE_SYSTEM_PROMPT = """You are a fact-labeling assistant. You will be given only an explanation for a statement. Your task is to determine what the most likely label is based on the explanation alone. Do not speculate or assume additional facts. Only use the information provided in the explanation. The label must be one of the following: {CLASSIFICATION_OPTIONS}. Output only the label and no additional text."""
PE_PROMPT = """The explanation: {EXPLANATION}. Output only the label."""
