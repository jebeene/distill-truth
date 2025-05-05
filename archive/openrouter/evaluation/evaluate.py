from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import re
from tqdm import tqdm
from openrouter.models.model_runner import run_model
from openrouter.prompts.prompt_builder import build_user_prompt
from openrouter.utils.pretty import print_metrics_summary, print_confusion_matrix, log_info, log_error
import json
from datetime import datetime
from pathlib import Path
from openrouter.utils.pretty import log_info

def evaluate_liar_dataset(config, dataset_name="liar", split="test", limit=100):
    """
    Evaluate the OpenRouter model on the LIAR dataset using numeric class labels (0–5).
    """
    log_info(f"Loading '{dataset_name}' dataset split: '{split}'...")
    dataset = load_dataset(dataset_name, split=split)

    if limit and len(dataset) > limit:
        dataset = dataset.select(range(limit))

    log_info(f"Evaluating {len(dataset)} examples from '{split}' split...")

    y_true = []
    y_pred = []

    for example in tqdm(dataset, desc="Evaluating LIAR dataset"):
        statement = example["statement"]
        true_label = int(example["label"])

        prompt = build_user_prompt(statement)
        model_output = run_model(prompt, config).strip()

        match = re.search(r"\b[0-5]\b", model_output)
        log_info(f"Prompt: {prompt}")
        log_info(f"True Label: {true_label}")
        log_info(f"Model output: {match}")
        if match:
            pred_label = int(match.group(0))
        else:
            log_error(f"Could not extract label from model output: '{model_output}'")
            pred_label = -1

        y_true.append(true_label)
        y_pred.append(pred_label)

    invalid_preds = sum(1 for p in y_pred if p == -1)
    if invalid_preds > 0:
        log_error(f"{invalid_preds}/{len(y_pred)} predictions could not be parsed.")

    log_info("Finished querying. Calculating metrics with valid labels...")

    y_true_clean = [yt for yt, yp in zip(y_true, y_pred) if yp != -1]
    y_pred_clean = [yp for yp in y_pred if yp != -1]

    accuracy = accuracy_score(y_true_clean, y_pred_clean)
    f1_macro = f1_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
    report = classification_report(y_true_clean, y_pred_clean, zero_division=0)
    cm = confusion_matrix(y_true_clean, y_pred_clean)

    print_metrics_summary(accuracy, f1_macro)
    print_confusion_matrix(cm)

    results = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm,
        "invalid_predictions": invalid_preds
    }

    save_evaluation_results(results, config, dataset_name, split, limit)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "report": report,
        "confusion_matrix": cm
    }

def save_evaluation_results(results, config, dataset_name="liar", split="test", limit=100):
    """
    Save evaluation results to a timestamped JSON file.

    :param results: Dict containing metrics and evaluation output.
    :param config: Config dictionary (should contain 'model' at minimum).
    :param dataset_name: Name of the dataset used.
    :param split: Dataset split.
    :param limit: Number of samples used.
    """
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get("model_selection", "unknown")
    filename = f"{dataset_name}_{split}_{model_name}_{timestamp}.json"

    results.update({
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "limit": limit
    })

    if isinstance(results.get("confusion_matrix"), (list, tuple)):
        pass
    elif hasattr(results.get("confusion_matrix"), "tolist"):
        results["confusion_matrix"] = results["confusion_matrix"].tolist()

    output_file = output_dir / filename
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    log_info(f"Evaluation results saved to: {output_file}")