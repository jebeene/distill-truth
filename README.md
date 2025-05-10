# DistillTruth: Fake News Detection with Small LLMs

This project evaluates the effectiveness of open-source **distilled language models** for fake news detection using the [LIAR dataset](https://huggingface.co/datasets/liar).

## Paper Introduction
The proliferation of false and misleading information (fake news) has emerged as one of the most pressing challenges of the past decade. By eroding public trust, distorting political discourse, and shaping individual and collective decision-making, misinformation poses a clear threat to the integrity of our digital society. Although advances in sequential modeling—particularly Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs)—have demonstrated encouraging classification performance, their opaque internal representations hinder our ability to understand and justify their predictions.

In recent years, Large Language Models (LLMs) have made significant strides in both accuracy and transparency when applied to fake-news detection, offering the dual benefits of high-fidelity classification and interpretable rationales. However, the substantial computational resources required to train and deploy these models limit their practical adoption, especially in resource-constrained environments. To date, little work has examined whether compression techniques—such as pruning, quantization, or knowledge distillation—can be applied to LLMs in a way that preserves their predictive power and explanatory capabilities.

In this paper, we explore the trade-off between efficiency, accuracy, and interpretability in compressed LLMs for fake-news classification. Specifically, we (1) apply state-of-the-art compression methods to a pretrained LLM; (2) evaluate its classification performance against uncompressed baselines; and (3) assess its ability to generate coherent, human-readable explanations for each decision. By demonstrating that a compressed LLM can maintain strong performance while substantially reducing computational overhead—and by providing evidence of preserved interpretability—we aim to bridge the gap between model efficacy and practical deployability in the ongoing fight against misinformation.

## Project Overview

- **Task**: Multi-class classification of political statements into 6 truthfulness categories.
- **Dataset**: LIAR (13K+ labeled statements with metadata)
- **Models**: Small LLMs (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
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
