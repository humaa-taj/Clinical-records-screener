# Design decisions and failures

A running record of options considered, what was rejected, and what went wrong.

## Data

**The dataset's `training` column is ignored.** The assignment prescribes a split by patient with seed 42.

**Only allowlisted columns are read.** Blocking GPT-4 columns by name can miss a column, so the loader requests just the eight columns it needs. `gpt4_*` and `explanation_correctness` are never loaded for prompts or targets.

**One row with a blank criterion is removed (annotation 883).** It has nothing to judge. This was decided before any results existed, and the row stays in the manifest's total counts.

**Absent evidence and expert labels are in tension.** In the training split, 74.7% of "not excluded" rows have no cited evidence, along with 82.1% of "not applicable" rows and 5.7% of "included" rows. Experts often reached "not excluded" because a condition was never mentioned. The assignment says absent evidence must not be treated as proof, so the model is scored against expert labels as required, and the app flags any verdict with no cited evidence for human review instead of presenting it as established.

**Rare labels.** Training has 8 "excluded" and 13 "not included" rows out of 602. The split is mandatory, so this cannot be fixed by resampling patients. Run R2 tests oversampling these labels in training only.

## Models

**Qwen3 rejected.** Its thinking mode is on by default and adds reasoning text that complicates structured output, and the 4B model needs about 8 GB at float16, which exceeds the laptop's 4 GB GPU.

**Qwen2.5-1.5B and 0.5B chosen as target and draft.** Verified identical vocabulary, special tokens, chat template and encodings, as Task 5 requires.

**Explanations are not trained.** The dataset's only written explanations come from GPT-4, and the assignment bars them from input and evaluation. Using them as training targets would teach the model GPT-4's outputs, so explanations are built only from the cited sentences.

## Training

**LoRA was chosen over full fine-tuning.** With only 602 training rows, training all 1.54B parameters would invite overfitting and would need far more memory. LoRA on the attention projections trains 4.36M parameters at rank 16, about 0.28% of the model.

**Decoding settings are written out explicitly.** Qwen's bundled generation settings enable sampling and a repetition penalty. Left in place, they would make before/after comparisons unclear and break token equality with the custom decoders, so greedy decoding with a repetition penalty of 1.0 is stated in the code.

**Gradient checkpointing added after a memory surprise.** A two-example laptop run reported a peak of 8.38 GB on a 4 GB GPU, against 2.98 GB for prediction. The model had stayed at float16; the extra memory was the autograd graph, which keeps activations for every layer and grows with token count, and Windows paged the overflow into system RAM. A batch of 8 long examples could exceed the 16 GB on Kaggle's T4, so checkpointing was enabled, which cut the same run's peak to 5.2 GB and its time from 264 to 74 seconds because it no longer paged. Batches were also set to 4 with 2 accumulation steps, keeping 8 examples per update with less peak memory.

**Model-written explanations were tested and rejected.** The base model, with the adapter disabled, was asked to explain verdicts from only the cited sentences. On real validation rows it invented facts: it called a female patient male, and said age 58 falls outside a 25-65 range. A check that rejected numbers absent from the sources did not catch either error, since both used numbers or words already present. The assignment requires explanations based only on available information, so explanations now state the verdict in fixed wording and quote the cited sentences exactly. This is less fluent, but it cannot introduce a fact.

## Fine-tuning results (validation only)

**R3 was selected: rank 32 with oversampled rare labels.** Validation macro-F1 was 0.558 and accuracy 74.3%, against 0.147 and 17.5% for the same model untrained, and 0.215 and 62.1% for always guessing the most common label. R1 (0.534) and R2 (0.552) were close. With 8 validation patients the patient-level 95% interval for R3 is 0.42 to 0.64, so these gaps may be noise. R3 is selected by the pre-stated rule, not because it is clearly better.

**Listing evidence first hurt the labels.** R4 reached only 0.459 macro-F1, although its evidence F1 (0.526) was slightly higher. The label-first output format was kept.

**Qwen beat SmolLM2.** Fine-tuned SmolLM2-1.7B (R5) reached 0.514 macro-F1 with low evidence recall (0.304), and has no matching smaller draft in this setup.

**Overfitting appeared early.** In most runs validation loss rose after epochs 2 to 4 while training loss kept falling. Early stopping uses generated macro-F1, since loss and macro-F1 did not move together.

**The adapter reproduces on other hardware.** Reloading R3 on the laptop GPU with a different batch size gave 205 of 206 identical outputs and identical labels compared with the Kaggle run. The untrained model was more sensitive: 17.0% accuracy locally against 17.5% on Kaggle, because padded batches change floating-point results. Final comparisons will use a single batch size on one machine.

## Attention

**The tiled version is much slower in Python, and that is expected.** On the laptop GPU at length 4096 with causal masks, the reference took 53 ms, PyTorch's fused kernel 19 ms, and tiled attention 101 ms with 256-token tiles or 21.7 s with 16-token tiles. Each tile costs interpreter time, separate GPU submissions and small allocations, so small tiles lose to overhead. A compiled kernel does the same loop in fused native code with no interpreter between tiles. Larger tiles reduce that overhead, which is why the 256-token tile is fastest.

**Memory is where tiling helps.** At length 4096 the reference needed 1,536 MB of GPU memory above the inputs, against 16-19 MB for every tile size, because the full score matrix never exists. Maximum error against the reference stayed near 1e-6 in float32 at every length.

**Causal masking skips about half the tiles.** Tiles above the diagonal are fully masked and never computed. At length 4096 with 256-token tiles, 120 of 256 tiles were skipped.

## Speculative decoding

**Draft on CPU made decoding about three times slower, even with high acceptance.** With the 1.5B target on the laptop GPU and the 0.5B draft on CPU, ten real screening prompts took 12.8 s with ordinary greedy decoding and about 40 s speculatively, at 85-95% acceptance. Tokens matched greedy decoding in every case. The target's own work fell sharply (verification took 3.4-6.0 s against 11.5 s of greedy decoding), but draft generation on CPU took 15-23 s and prefilling the CPU draft about 11 s more. Speculative decoding pays off only when drafting is much cheaper than target steps. The Kaggle run, with both models on the GPU, tests that condition.

**Moving the draft to the laptop GPU did not produce a real speedup.** Speculative decoding took 37-38 s on the ten screening prompts, against 20.1 s for greedy, at the same acceptance. A draft step cost about the same as a target step on this GPU: 0.5B drafting took 15.8-23.8 s for that workload, against 15.5 s of greedy decoding. Fixed per-step costs, such as interpreter time and mask building, dominate at these model sizes, so a smaller model is not proportionally faster. With both models loaded the 4 GB GPU was almost full, which also slowed greedy decoding here (20.1 s against 12.8 s with the draft on CPU). The single 1.06-1.09x gain, at 1024 input tokens, should be read with that memory pressure in mind, since it lengthened greedy prefill and decoding.

**Longer drafts were not better.** At k=8 on longer generations acceptance fell to 51-62%, and throughput fell with it, because more drafted tokens were thrown away.

## Application

**The overall result never treats missing evidence as proof.** A patient is "likely ineligible" only if a criterion failed with cited evidence. A failure with no cited sentence becomes "needs review", as does anything unknown, unverified or uncited. This makes "needs review" common, because experts often labeled "not excluded" without citing evidence, but the assignment forbids treating absence as proof either way.

**Vercel was rejected.** It has no GPU and its serverless functions cannot keep a 3 GB model loaded. The assignment accepts a locally hosted browser application, which also keeps the live benchmark on real hardware and works offline.

**Retraining and a larger model were rejected after the test split was opened; a deterministic number check was added instead.** Rehearsal showed the model is unreliable at numeric comparisons. Retraining or changing models would break the rule that the test set stays untouched and the models are not tuned on it, and added training data would break the rule to use only the prescribed dataset. A model larger than 1.5B also does not fit the 4 GB laptop GPU that runs the live application. The check corrects only bare comparisons with one clear value in the note and otherwise stays silent. It passed a validation bar set in advance, 5 of 5 decisions correct, but all 5 were age criteria the model already had right, so its benefit on BMI and laboratory values is unmeasured on the dataset. It sits outside every reported metric.

**MLflow was not used.** CSV and JSON logs with pinned configurations already meet the logging and reproducibility requirements, and it would add a dependency with no assessed benefit.

## Environment failures

**System PyTorch was CPU-only.** Found during the resource check. A project virtual environment now installs the CUDA 12.8 build.

**First dependency install failed.** `datasets` 5.1.0 needs `huggingface_hub` 1.31 or newer, and the initial pin was 1.14.0. Corrected to 1.33.0.

**First Kaggle training run failed on a preinstalled package.** Kaggle ships torchao 0.10.0, and peft 0.21.2 raises an error when it finds a torchao older than 0.16 while attaching LoRA layers. The baselines finished because they attach no adapter. torchao is now pinned to 0.18.0, and this was tested on the laptop with a real LoRA epoch.

**Dataset arrived under Git LFS.** The upstream `.gitattributes` routed the parquet file through LFS, so a plain clone without git-lfs would receive a pointer file. The rule was removed and the file is stored directly.

**The final Kaggle session timed out, so the sealed test evaluation ran more than once.** The first run completed the test evaluation at 12:24-12:25 UTC on 2026-11-09, then the session stopped for inactivity during the last benchmark and its files were erased. The notebook was rerun from the same `final-config` commit with the same adapters. No code, setting or model changed between the runs, and the test results were not used for any decision. Test accuracy and macro-F1 matched the first run to four decimal places for the untrained model, the fine-tuned model and the baseline. The first run's console output is kept in `results/kaggle_first_attempt.txt`, and `results/test_access_log.jsonl` lists the accesses from both runs.
