# DistillTruth: Fake News Detection with Distilled LLMs

This project evaluates the effectiveness of open-source **distilled language models** for fake news detection using the [LIAR dataset](https://huggingface.co/datasets/liar).

## Project Overview

- **Task**: Multi-class classification of political statements into 6 truthfulness categories.
- **Dataset**: LIAR (13K+ labeled statements with metadata)
- **Models**: Distilled LLMs (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
- **Evaluation**: Prompt-based text generation with label extraction and metric computation
---

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
