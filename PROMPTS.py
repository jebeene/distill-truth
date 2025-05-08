CLASSIFICATION_OPTIONS = """0. false
1. half-true
2. mostly-true
3. true
4. barely-true
5. pants-fire"""

# Classification Prompts
CLASSIFICATION_SYSTEM_PROMPT = """---You are a fact-checking assistant trained to verify political and non-political claims. You carefully analyze a given statement and determine whether it is any of the following options based on reasoning and evidence:

{CLASSIFICATION_OPTIONS}

Your responses are concise, well-justified, and avoid speculation.---"""

#Few Shot Prompts
FEW_SHOT_PROMPT_TEMPLATE = """Example:
Statement: "{STATEMENT}"  
Label: {LABEL}  
Explanation: "{EXPLANATION}"  
---"""
IS_INCLUDE_SPEAKER = "Speaker: {SPEAKER}"
IS_INCLUDE_PARTY = "Party: {PARTY_AFFILIATION}"

STATEMENT_CLASSIFICATION_PROMPT = """Now analyze the following statement:

Statement: "{STATEMENT}"  
{IS_INCLUDE_SPEAKER}  
{IS_INCLUDE_PARTY}  

Respond only in the following format:  
Label: <label>  
Explanation: <short justification>
"""

JUSTIFICATION_PROMPT = """Now that you have classified the statement. Provide an explanation to the user as to why you think the statement is {CLASSIFIED_LABEL} and what parts of the statement makes you believe so."""

# Proxy Evaluation of Justification Prompts
PE_SYSTEM_PROMPT = """You are a fact-labeling assistant. You will be given only an explanation for a statement. Your task is to determine the most likely label is to replace \"___\" based on the explanation alone. Do not speculate or infer beyond the provided explanation. Only use the information provided in the explanation. The label must be one of the following: {CLASSIFICATION_OPTIONS}. Output only the label—no additional text."""
PE_PROMPT_TEMPLATE = """The explanation: \"{EXPLANATION}\". Output only the label."""
