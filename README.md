# DistillTruth: Fake News Detection with Distilled LLMs

This project evaluates the effectiveness of open-source **distilled language models** for fake news detection using the [LIAR dataset](https://huggingface.co/datasets/liar).

## Project Overview

- **Task**: Multi-class classification of political statements into 6 truthfulness categories.
- **Dataset**: LIAR (13K+ labeled statements with metadata)
- **Models**: Distilled LLMs (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
- **Evaluation**: Prompt-based text generation with label extraction and metric computation
---

## Directory Structure

- **configs/**: Configuration files for API keys, base URLs, and model mappings.
- **data/**: Modules for loading data from local files and Hugging Face datasets, and preprocessing.
- **models/**: Model wrappers including OpenRouter API integration and constants.
- **prompts/**: Prompt templates and builder functions.
- **pipelines/**: Orchestration pipelines for chat and evaluation.
- **evaluation/**: Evaluation scripts and metrics.
- **scripts/**: CLI entry points and SLURM batch script for execution.
- **utils/**: Shared helper functions (config loader, logger).
- **tests/**: Unit tests for the project.
- 
## LIAR Labels

| ID | Label         |
|----|---------------|
| 0  | false         |
| 1  | half-true     |
| 2  | mostly-true   |
| 3  | true          |
| 4  | barely-true   |
| 5  | pants-fire    |

---

## Project Setup
1. Install `pip` and run `pip install -r requirements.txt`
2. Create an [OpenRouter API key](https://openrouter.ai/settings/keys) and run `export OPENROUTER_API_KEY=your_actual_api_key`
3. Query the default model (defined in `/configs/openrouter_config.yaml)` using `python main.py "Your prompt here"`
4. 