# Alignment via QLoRA & DPO for LLM Safety

This repository contains the practical project developed for the Deep Learning Graduate Seminar. The objective is to evaluate intrinsic alignment methodologies for small language models (sub-3B parameters), focusing on mitigating safety risks and harmful outputs. Using Direct Preference Optimization (DPO) combined with QLoRA (Quantized Low-Rank Adaptation) via Hugging Face's TRL and PEFT libraries, we optimize a quantized instruction-tuned model against constitutional safety preference pairs without the overhead of training an explicit reward model.

The project environment is managed with [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency resolution. All runtime dependencies, tooling definitions, and metadata are specified in `pyproject.toml`. To set up the virtual environment and install the required dependencies (PyTorch with CUDA support, Transformers, Datasets, TRL, PEFT, and BitsAndBytes), run:

```bash
uv sync

```

The repository is structured around two primary entry points: a training pipeline (`train.py`) that loads 4-bit base checkpoints alongside safety preference datasets (such as Anthropic's `hh-rlhf`), and a side-by-side evaluation script (`eval.py`). The evaluation workflow dynamically toggles the trained LoRA adapter on and off (`model.disable_adapter()`), allowing direct, zero-overhead qualitative comparisons between the base model and the aligned model across red-teaming and safety benchmark prompts.
