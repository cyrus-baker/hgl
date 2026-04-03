# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commonly Used Commands

- **Install dependencies:** `uv sync`
- **Run the custom causal LM script:** `python lt.py`
- **Run the translation fine-tuning script:** `python new_main.py`
- **Run translation fine-tuning with DDP:** `torchrun --nproc_per_node=<num_gpus> new_main.py`

## High-Level Architecture

The repository serves as an experimental workspace for learning to train Transformers (specifically a causal LM) and fine-tuning seq2seq models.

- **Custom Causal Language Model (`lt.py`)**
  - Implements a GPT-style causal transformer (`MiniGPT`) from scratch using PyTorch.
  - Contains core components: `TransformerBlock` (with `nn.MultiheadAttention` and causal masking) and `TokenAndPositionEmbedding`.
  - Built to understand causal LLM mechanics. Uses the `Qwen/Qwen3.5-27B` tokenizer for tokenization experiments.

- **Seq2Seq Translation Fine-tuning (`main.py` & `new_main.py`)**
  - Scripts to fine-tune `facebook/nllb-200-distilled-600M` for English-to-Chinese translation.
  - Operates on the `Helsinki-NLP/opus-100` dataset.
  - `new_main.py` incorporates logic for Distributed Data Parallel (DDP) execution (e.g., `is_main_process()`, rank-aware data caching).
  - Uses Hugging Face's `Seq2SeqTrainer` and the `sacrebleu` metric for evaluation.

- **Dependency Management (`pyproject.toml`)**
  - Managed by `uv`.
  - Defines explicit package indexes for PyTorch CUDA 12.8 wheels to ensure correct GPU support.
