# DistillTruth: Fake News Detection with Distilled LLMs

This project evaluates the effectiveness of open-source **distilled language models** for fake news detection using the [LIAR dataset](https://huggingface.co/datasets/liar).

## Project Overview

- **Task**: Multi-class classification of political statements into 6 truthfulness categories.
- **Dataset**: LIAR (13K+ labeled statements with metadata)
- **Models**: Distilled LLMs (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
- **Evaluation**: Prompt-based text generation with label extraction and metric computation
---

## Project Setup

1. Create an [OpenRouter API key](https://openrouter.ai/settings/keys)
2. Create `.env` in project root with as `OPENROUTER_API_KEY=your_actual_api_key`

**You can either setup the environment using pip / conda OR Docker.**

### pip / conda
1. Install `pip`
2. Run `pip install -r requirements.txt`
3. Run `make` to see available commands

### Docker
1. Download Docker
2. Run `make build` to build the Docker Image
3. Run `make run` to launch a `bash` shell in the Docker container
4. Run `make` to see available commands

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
