import sys
import os
import pytest
from sklearn.metrics import accuracy_score, f1_score
from evaluate import extract_label, compute_metrics, build_prompt

# --------------- LABEL EXTRACTION ----------------
@pytest.mark.parametrize("generated_text,expected_label", [
    ("Number 0", 0),
    ("Barely true 4", 4),
    ("Def 1: half-true", 1),
    ("I think the answer is 5", 5),
    ("unknown", None),
    ("500", None),
    ("I think 4 or 5", None),
    ("", None),
])
def test_extract_label(generated_text, expected_label):
    assert extract_label(generated_text) == expected_label

# --------------- METRIC COMPUTATION ----------------

def test_compute_metrics():
    preds = ["true", "false", "half-true", "true"]
    labels = ["true", "false", "false", "true"]

    metrics = compute_metrics(preds, labels)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert abs(metrics["accuracy"] - accuracy_score(labels, preds)) < 1e-6
    assert abs(metrics["f1_macro"] - f1_score(labels, preds, average='macro')) < 1e-6


# --------------- PROMPT FORMAT ----------------

def test_prompt_format():
    statement = "The Earth revolves around the Sun."
    prompt = build_prompt(statement)

    assert isinstance(prompt, str)
    assert statement in prompt
    assert "classify" in prompt.lower() or "truth" in prompt.lower()
