# Resource manifest

Everything the project needs, mapped to the rubric. `scripts/fetch_resources.py` downloads the
"pulled" items and records exact versions and hashes in `resources.lock.json`.
`scripts/verify_resources.py` checks them.

## Pulled locally

| Item | Saved to | Size | Needed for |
|---|---|---|---|
| TrialGPT Criterion Annotations (the only allowed dataset) | `data/raw/` | 176 KB | B1, B2, C1-C4 |
| Qwen/Qwen2.5-1.5B-Instruct (target model) | `models/qwen2.5-1.5b/` | 2.9 GB | A2, B3, B4, D, F, G |
| Qwen/Qwen2.5-0.5B-Instruct (draft model) | `models/qwen2.5-0.5b/` | 0.9 GB | F1-F3 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct (second candidate) | `models/smollm2-1.7b/` | 3.2 GB | A2, A3 |
| NLTK punkt_tab sentence splitter (numbers sentences in unseen notes) | `data/nltk_data/` | small | G2 |
| 11 papers (listed below) | `papers/` | about 10 MB | A1, H1 |

## Papers

| arXiv ID | Title | Used for |
|---|---|---|
| 2307.15051 | Matching Patients to Clinical Trials with Large Language Models | Problem, dataset |
| 1706.03762 | Attention Is All You Need | Architecture, Task 4 reference |
| 2106.09685 | LoRA: Low-Rank Adaptation of Large Language Models | Fine-tuning |
| 2305.14314 | QLoRA: Efficient Finetuning of Quantized LLMs | Fine-tuning alternative |
| 2211.17192 | Fast Inference from Transformers via Speculative Decoding | Task 5 |
| 2302.01318 | Accelerating Large Language Model Decoding with Speculative Sampling | Task 5 |
| 2205.14135 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Task 4 |
| 1805.02867 | Online normalizer calculation for softmax | Task 4 |
| 2412.15115 | Qwen2.5 Technical Report | A2 |
| 2505.09388 | Qwen3 Technical Report | A2, rejected alternative |
| 2502.02737 | SmolLM2: When Smol Goes Big | A2 |

## Software

`.venv` with GPU PyTorch (CUDA 12.8 build) and `requirements.txt`, created by `scripts/setup_env.ps1`.

## Deliberately not pulled

| Item | Why |
|---|---|
| TrialGPT source code | Reference only under the assignment rules. Read it at https://github.com/ncbi-nlp/TrialGPT |
| Qwen3-4B, Qwen3-1.7B, Qwen3-0.6B weights | Discussed as a rejected alternative. Thinking mode complicates structured output and 4B will not fit a 4 GB GPU. Facts come from the model cards and the Qwen3 report. |
| ClinicalTrials.gov records | Optional and fetched live: https://clinicaltrials.gov/api/v2/ |
| IEEEtran template | Built into Overleaf |

## Accounts and manual steps (only the owner can do these)

| Step | Why | Blocks |
|---|---|---|
| Kaggle account with phone verification | Free GPU for training and benchmarks | B3, D3, F3 |
| Hugging Face account and write token | Host the trained adapter | H2 |
| GitHub repository | Required submission | H2 |
| Overleaf account | Compile the IEEE LaTeX paper | H1 |
| Screen recorder and video link | 5-7 minute demo | H2 |

## Hardware

| Machine | Spec | Role |
|---|---|---|
| Laptop | i7-11800H, 15.7 GB RAM, RTX 3050 Ti 4 GB | Website, tests, small benchmarks |
| Kaggle (free) | GPU with about 16 GB | Fine-tuning, full benchmarks |
