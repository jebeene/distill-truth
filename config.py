# config.py
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
