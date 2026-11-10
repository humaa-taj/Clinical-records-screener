# CRS — Clinical Records Screener

A Transformer-based system that matches a patient description against individual clinical-trial
inclusion and exclusion criteria. For every criterion it returns an eligibility label, the IDs of the
patient sentences that support it, the exact evidence text and an explanation built only from that
evidence. It also implements and benchmarks KV-cached decoding, tiled online-softmax attention and
greedy speculative decoding from scratch.

> **Educational screening prototype — not a clinical enrollment decision.**

GenAI Assignment #2. Model: **Qwen2.5-1.5B-Instruct + LoRA** (target) and
**Qwen2.5-0.5B-Instruct + LoRA** (speculative draft). Dataset:
[ncbi/TrialGPT-Criterion-Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations).

---

## Contents

1. [Quick start](#quick-start)
2. [Using the application](#using-the-application)
3. [Results at a glance](#results-at-a-glance)
4. [Repository layout](#repository-layout)
5. [Reproducing the experiments](#reproducing-the-experiments)
6. [Dataset and split](#dataset-and-split)
7. [Model and fine-tuning](#model-and-fine-tuning)
8. [Inference mechanisms](#inference-mechanisms)
9. [Hardware and limitations](#hardware-and-limitations)
10. [Downloads](#downloads)
11. [AI-use acknowledgement](#ai-use-acknowledgement)

---

## Quick start

Requirements: Python 3.11, about 6 GB of free disk space and 12 GB of RAM. A GPU is **optional**:
the app runs on CPU (slower, but fully functional).

### Linux / macOS

```bash
git clone https://github.com/humaa-taj/Clinical-records-screener.git
cd Clinical-records-screener
bash scripts/setup_env.sh          # CPU PyTorch; use CRS_CUDA=1 bash scripts/setup_env.sh for an NVIDIA GPU
./start_app.sh                     # opens http://127.0.0.1:8600
```

### Windows

```powershell
git clone https://github.com/humaa-taj/Clinical-records-screener.git
cd Clinical-records-screener
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.venv\Scripts\python scripts\fetch_resources.py --app
.venv\Scripts\python scripts\fetch_adapters.py
start_app.bat
```

`setup_env.sh` creates `.venv`, installs the pinned dependencies (`requirements.txt`), downloads the
two Qwen models at the revisions locked in `resources.lock.json`, the NLTK sentence splitter, and
the fine-tuned LoRA adapters at the revisions locked in `adapters.lock.json`.

The frontend is shipped prebuilt in `app/web/dist`, so Node.js is **not** needed to run the app.
To rebuild it: `cd app/web && npm ci && npm run build`.

The server loads the model in a background thread. The page is usable immediately and shows a
"loading model" state until the model is ready (about 30–60 s on CPU).

---

## Using the application

1. **Patient note** — paste a free-text patient description, or load the built-in example.
2. **Sentence review** — the note is split into numbered sentences (NLTK `punkt_tab`). These IDs are
   what the model cites as evidence; confirm or edit them.
3. **Trial criteria** — enter inclusion and exclusion criteria (one per line), load the example
   trial, or fetch a trial by NCT ID from the public ClinicalTrials.gov API.
4. **Run** — every criterion is screened by the fine-tuned model. Choose the decoding mode
   (KV-cached greedy or speculative).
5. **Results** — per-criterion label, cited sentence IDs, highlighted evidence text and explanation.
   The overall verdict is a rule-based **aggregation** of criterion labels, not a ground-truth label.
6. **Export** — download the screening report as HTML or JSON. Each report records the exact model,
   adapter revision and decoding settings.
7. **Inspect / benchmark** — run live comparisons on the current machine:
   uncached vs KV-cached vs speculative decoding (token equality, latency, tokens/s, acceptance
   statistics, memory) and reference vs tiled online-softmax attention.

### API (for scripting)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Model load state, device, model/adapter versions |
| GET | `/api/example` | Example patient and trial |
| POST | `/api/sentences` | Split a note into numbered sentences |
| GET | `/api/trial/{nct_id}` | Fetch criteria from ClinicalTrials.gov |
| POST | `/api/screen` | Start a screening job |
| POST | `/api/compare/decoding` | Start a decoding comparison job |
| POST | `/api/compare/attention` | Start an attention comparison job |
| GET | `/api/jobs/{id}` | Poll a job; `POST /api/jobs/{id}/cancel` cancels it |
| GET | `/api/reports/{id}/download.html` / `.json` | Export a report |

---

## Results at a glance

All numbers are on the **held-out test split** (8 patients, 206 criteria) with greedy decoding,
`max_new_tokens=96`, unless stated otherwise. Full metrics with per-class P/R/F1, confusion
matrices and patient-level bootstrap intervals are in `results/eval/*/metrics.json`.

### Criterion classification

| System | Overall acc. | Overall macro-F1 | Inclusion macro-F1 | Exclusion macro-F1 |
|---|---|---|---|---|
| Majority-class baseline | 0.626 | 0.272 | 0.168 | 0.273 |
| Qwen2.5-1.5B, before fine-tuning | 0.204 | 0.234 | 0.203 | 0.170 |
| **Qwen2.5-1.5B + LoRA (R3), after** | **0.728** | **0.560** | **0.469** | **0.427** |

Invalid-output rate after fine-tuning: 0.0.

### Evidence sentence retrieval (test)

| System | Micro P | Micro R | Micro F1 | Macro F1 (rows with gold evidence, n=61) |
|---|---|---|---|---|
| Before fine-tuning | 0.078 | 0.394 | 0.130 | — |
| **After fine-tuning** | **0.577** | **0.479** | **0.523** | **0.528** |

Empty-set rules: both gold and prediction empty counts as correct in the macro-all-rows score
(0.817); gold empty with predictions gives precision 0; gold present with no prediction gives
recall 0. Absent evidence is **never** treated as proof of eligibility or ineligibility.

### Fine-tuning experiments (validation only)

| Run | Model | Setting | Val macro-F1 | Val evidence F1 |
|---|---|---|---|---|
| R0 | Qwen2.5-1.5B | no fine-tuning | 0.147 | 0.106 |
| R0b | SmolLM2-1.7B | no fine-tuning | 0.120 | 0.111 |
| R1 | Qwen2.5-1.5B | LoRA r=16 | 0.534 | 0.462 |
| R2 | Qwen2.5-1.5B | r=16, oversampled rare labels | 0.552 | 0.487 |
| **R3** | **Qwen2.5-1.5B** | **r=32, oversampled** | **0.558** | **0.509** |
| R4 | Qwen2.5-1.5B | r=32, evidence before label | 0.459 | 0.526 |
| R5 | SmolLM2-1.7B | best config | 0.514 | 0.446 |
| R6 | Qwen2.5-0.5B | best config (draft model) | 0.446 | 0.491 |

R3 was selected by highest validation macro-F1 (tie-break: evidence F1). The test split was opened
once, after the configuration was frozen (`results/test_access_log.jsonl`).

### KV-cached decoding (Tesla T4, fp16, greedy)

Greedy tokens were identical between cached and uncached paths in all 9 input/output length
combinations. Decode throughput (tokens/s, median of 5):

| Input tokens | Uncached | KV-cached |
|---|---|---|
| 64 | 23.1 | 25.6 |
| 256 | 12.4 | 25.9 |
| 1024 | 3.1 | 25.8 |

(Output length 64 shown; 16 and 256 are in `results/bench_cache_kaggle-t4-finetuned.json`.)

### Speculative decoding (Tesla T4, fine-tuned 0.5B draft)

Output tokens were identical to plain greedy decoding for every k. On this hardware speculative
decoding was **slower** (0.51×–0.85×): a 0.5B draft is a third of the target's size, and at batch
size 1 the T4 is latency-bound, so draft forwards cost almost as much as target forwards. The
measured breakdown (draft time, verify time, cache sync) is in
`results/bench_speculative_*.json` and discussed in the report.

| Scenario | Greedy tok/s | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| Screening JSON (20 prompts) | 23.9 | 18.2 | 19.3 | 20.2 |
| 1024 in / 128 out | 24.5 | 17.7 | 16.5 | 12.5 |

### Tiled online-softmax attention

Matches the reference attention to ~1e-6 (fp32) across shapes, tile sizes and masks; it is slower
than PyTorch's fused SDPA because it runs as Python loops over tiles with no kernel fusion
(`results/bench_attention_*.json`).

---

## Repository layout

```
screener/data/        split.py (37/8/8 split), prepare.py, prompts.py, sentences.py
screener/train/       train_lora.py, run_experiments.py, config/experiments.yaml
screener/eval/        metrics.py, evaluate.py (sealed test access), verifier.py, explain.py, predict.py
screener/inference/   uncached_decoder.py, cached_decoder.py, speculative.py, common.py
screener/attention/   reference.py, tiled_online.py, model_patch.py
benchmarks/           bench_cache.py, bench_attention.py, bench_speculative.py
app/                  server.py (FastAPI), engine.py, pipeline.py, report.py, number_check.py
app/web/              React + Vite frontend (prebuilt in dist/)
kaggle/               train.ipynb, final_eval_and_benchmarks.ipynb
tests/                pytest suite (split leakage, metrics, decoders, attention, speculative, app)
splits/manifest.json  published patient-grouped split
results/              metrics, predictions, training logs, benchmark logs, error analysis
docs/                 DECISIONS.md (alternatives and rejected options), IMPLEMENTATION_PLAN.md
scripts/              setup_env.sh / .ps1, fetch_resources.py, fetch_adapters.py, verify_resources.py
```

---

## Reproducing the experiments

```bash
.venv/bin/python -m pytest -q                      # full test suite; model-dependent tests skip if weights are absent
.venv/bin/python -m screener.data.split            # regenerate splits/manifest.json
.venv/bin/python -m screener.train.run_experiments # R0-R6 (GPU recommended; we used a Kaggle T4)
.venv/bin/python -m screener.eval.evaluate --split validation \
    --model models/qwen2.5-1.5b --adapter models/crs-adapter --output results/eval/after_validation
.venv/bin/python benchmarks/bench_cache.py
.venv/bin/python benchmarks/bench_attention.py
.venv/bin/python benchmarks/bench_speculative.py
```

Training and the final benchmarks were run on a free Kaggle Tesla T4 with
`kaggle/train.ipynb` and `kaggle/final_eval_and_benchmarks.ipynb`. The evaluator refuses the test
split unless called with `--final`, and logs every such access.

---

## Dataset and split

- Only the prescribed TrialGPT Criterion Annotations dataset (1,015 rows, 53 patients) is used.
- Unique patient IDs are sorted, shuffled with `numpy.random.default_rng(42)` and assigned
  **37 / 8 / 8** to train / validation / test. Every annotation of a patient is in exactly one split.
  `tests/test_split.py` checks disjointness and determinism.
- Train 603 rows (602 usable; one blank-criterion row removed and recorded), validation 206, test 206.
- Model inputs are built from an allow-list of columns: patient sentences, criterion type and
  criterion text. GPT-4 predictions and explanations can never reach a prompt (tested).
- No prompt is truncated: the longest note is 372 tokens (`results/truncation.json`).

## Model and fine-tuning

| | Qwen2.5-1.5B-Instruct (chosen) | SmolLM2-1.7B-Instruct (candidate) | Qwen2.5-0.5B-Instruct (draft) |
|---|---|---|---|
| Parameters | 1,543.7 M | 1,711.4 M | 494.0 M |
| Layers | 28 | 24 | 24 |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| Tokenizer | Qwen2 BPE | SmolLM2 BPE | identical to target |

Qwen2.5-1.5B was chosen for its higher validation macro-F1 after the same fine-tuning (R3 vs R5),
its 32K context window and a same-tokenizer 0.5B sibling for speculative decoding. The Qwen3
family was rejected (thinking mode complicates strict JSON output; 4B does not fit a 4 GB GPU).
See `docs/DECISIONS.md`.

**Fine-tuning:** LoRA (rank 32, alpha 64) on the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), loss on assistant
answer tokens only, rare labels oversampled (power 0.5), early stopping on validation macro-F1.
**Output schema:** `{"label": "<category>", "evidence": [sentence ids]}`. Labels preserve all expert
categories: *included / not included / not enough information / not applicable* for inclusion and
*excluded / not excluded / not enough information / not applicable* for exclusion. Explanations
are template-based and quote the cited sentences verbatim, so they cannot invent facts.

## Inference mechanisms

- **KV cache** (`screener/inference/cached_decoder.py`): explicit loop — prefill the prompt, then
  feed one token per step with an extended attention mask, explicit `position_ids` and
  `cache_position`, asserting the cache length every step. Verified against a full-recompute
  reference (`uncached_decoder.py`).
- **Tiled attention** (`screener/attention/tiled_online.py`): loops over query/key tiles keeping a
  running max `m`, normaliser `l` and accumulator `O`, rescaling by `exp(m_old − m_new)`; causal
  and padding masks per tile; no fused backend. Can be patched into the real model
  (`model_patch.py`) with identical generated tokens.
- **Speculative decoding** (`screener/inference/speculative.py`): the draft proposes k tokens, the
  target verifies them in one forward pass, the longest matching prefix is accepted, the target's
  own token is appended as correction/bonus, and both caches are cropped so each holds every token
  except the last.

## Hardware and limitations

- Training and the reported benchmarks: Kaggle Tesla T4 (16 GB). Development: RTX 3050 Ti 4 GB.
  The demo runs on a CPU-only laptop, where the model is loaded in float32 (~6 GB RAM) and screening
  takes several seconds per criterion.
- The test split has only 8 patients, so confidence intervals are wide; rare classes
  (*excluded*, *not applicable*) have very few examples.
- Speculative decoding did not yield a speedup on the hardware tested (see above).
- This is a research prototype and must not be used for real enrollment decisions.

## Downloads

| Artifact | Location |
|---|---|
| Target LoRA adapter (R3) | [faa-tehe/crs-qwen2.5-1.5b-lora](https://huggingface.co/faa-tehe/crs-qwen2.5-1.5b-lora) |
| Draft LoRA adapter (R6) | [faa-tehe/crs-qwen2.5-0.5b-draft-lora](https://huggingface.co/faa-tehe/crs-qwen2.5-0.5b-draft-lora) |
| Base models | Hugging Face `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct` (revisions in `resources.lock.json`) |
| Split manifest | `splits/manifest.json` |
| Benchmark logs | `results/bench_*.json` |

`scripts/fetch_adapters.py` downloads the adapters at the pinned revisions.

## AI-use acknowledgement

AI coding assistants (Claude) were used to help write and refactor code, set up the
repository and environment, and draft documentation. All reused or generated code is covered by
the test suite, and every design decision, experiment and result was reviewed and understood by
the author. The model's own outputs never use GPT-4 text from the dataset.
