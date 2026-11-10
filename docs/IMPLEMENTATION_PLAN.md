# Implementation plan

Goal: full marks on all 27 rubric items.

**How to read this:** each task has an ID, the rubric items it earns, and a "done when" test.
`[YOU]` marks work only you can do. Everything else is built and committed as we go.

Standing rules:

- The test split is sealed. Only one script can read it, and only in Phase 7.
- Prompts, thresholds and hyperparameters are chosen on validation data only.
- Every failed experiment goes in `docs/DECISIONS.md`, because A3 gives marks for it.
- Every benchmark result is labeled with its hardware, model and decoding settings.
- Commit and push after each task, with no AI co-author lines.

## Status of decisions already made

| Decision | Choice | Evidence |
|---|---|---|
| Target model | Qwen2.5-1.5B-Instruct, 1,543.7M parameters, 28 layers, Apache 2.0 | `results/resource_check.txt` |
| Draft model | Qwen2.5-0.5B-Instruct, 494.0M parameters, 24 layers, identical tokenizer | `results/resource_check.txt` |
| Competing candidate | SmolLM2-1.7B-Instruct, 1,711.4M parameters | `results/resource_check.txt` |
| Rejected | Qwen3 family: thinking mode complicates JSON output, 4B does not fit a 4 GB GPU | `RESOURCES.md` |
| Compute | Kaggle T4 for training and benchmarks, laptop for the app | |
| Explanations | Not trained, because the only written explanations are GPT-4's | Task 2.3 |
| Blank criterion row | Removed from training and scoring in every split, listed in the manifest. Decided before the split so it cannot depend on results. | Task 1.3 |

## Target repository layout

```
screener/data/        split.py, prepare.py, prompts.py, sentences.py
screener/train/       train_lora.py, config/*.yaml
screener/eval/        metrics.py, evaluate.py, verifier.py, errors.py
screener/inference/   cached_decoder.py, uncached_decoder.py, speculative.py
screener/attention/   reference.py, tiled_online.py
benchmarks/           bench_cache.py, bench_attention.py, bench_speculative.py
tests/                pytest suite for all of the above
kaggle/               train.ipynb, final_eval_and_benchmarks.ipynb
app/                  server.py, pipeline.py, static/
splits/               manifest.json
results/              logs, metrics, figures data
report/               main.tex, figures/
docs/                 DECISIONS.md, DEFENSE.md, AI_USE.md
start_app.bat         double-click launcher
```

## Phase 0: Resources (complete)

- [x] Dataset, 11 papers, 3 models, sentence splitter downloaded and pinned.
- [x] GPU environment built and 15 checks passed.
- [x] GitHub connected.

## Phase 1: Data foundation (B1, B2)

- [x] **1.1 Fixed split.** `split.py`: sort unique patient IDs, shuffle with `np.random.default_rng(42)`, assign 37/8/8. Ignore the dataset's `training` column.
  Done when: `splits/manifest.json` lists patient IDs, row counts and label counts per split, and the test split's label counts are written without being printed or used.
- [x] **1.2 Leakage tests.** `tests/test_split.py`: three disjoint patient sets, 53 total, every row in exactly one split, rerun gives identical output.
  Done when: tests pass and the output is committed.
- [x] **1.3 Sentence parsing.** `sentences.py`: turn each numbered note into `{id: text}` and keep the original IDs. Add a splitter for unseen notes using NLTK. Remove the one blank-criterion row and record it.
  Done when: reassembled sentences match the note and every evidence ID resolves.
- [x] **1.4 Input and output format.** `prompts.py` builds the prompt from patient sentences, criterion type and criterion text. Output is JSON: `{"label": ..., "evidence": [ids]}`.
  Done when: a column allowlist is used, and a test proves no `gpt4_*` or `explanation_correctness` value can reach a prompt.
- [x] **1.5 Truncation audit.** Measure full prompt length on train and validation. No truncation in training (longest note is 372 tokens).
  Done when: `results/truncation.json` shows zero truncated examples.
- [x] **1.6 Repo fix.** Remove the LFS rule on the dataset so a plain `git clone` gets the real parquet file.
  Done when: a fresh clone reads 1,015 rows without git-lfs.

## Phase 2: Fine-tuning (A3, B3)

- [x] **2.1 Training script.** `train_lora.py` with PEFT LoRA. Loss covers only the assistant's answer tokens. Fixed seeds, YAML config, per-epoch CSV log of train loss, validation loss and validation macro-F1 from real greedy generation. Early stopping on validation macro-F1.
- [x] **2.2 Local smoke test.** Run a few examples on the laptop to catch bugs before spending Kaggle time.
- [x] **2.3 Explanation design.** Explanations state the verdict in fixed wording and quote the cited sentences exactly. A base-model paraphrase was tested and rejected because it invented facts (see `docs/DECISIONS.md`). No GPT-4 text is used anywhere.
- [x] **2.4 Kaggle notebook.** `kaggle/train.ipynb` clones the repo, installs pinned packages and runs the experiment list.
- [x] **2.5 `[YOU]` Kaggle run.** Phone verification is done. Select T4 GPU, turn on internet, run the notebook.
- [x] **2.6 Experiment list, validation only.**

| Run | What it tests | Why |
|---|---|---|
| R0 | Qwen2.5-1.5B with no fine-tuning | Before-training baseline |
| R0b | SmolLM2-1.7B with no fine-tuning | Evidence for model selection |
| R1 | LoRA rank 16, no rebalancing | Starting point |
| R2 | LoRA rank 16, oversampled rare labels | Tests the rare-label fix |
| R3 | LoRA rank 32, oversampled | Tests adapter size |
| R4 | Best so far, with evidence listed before the label | Tests output order |
| R5 | Best config on SmolLM2-1.7B | Competing model after training |
| R6 | Best config on Qwen2.5-0.5B | Fine-tuned draft for Task 5 |

  Done when: best checkpoint is chosen by validation macro-F1, with evidence F1 as tie-break, and curves and logs are committed.
- [x] **2.7 Publish adapter.** `[YOU]` create a Hugging Face write token. Upload adapters and add `scripts/fetch_adapter.py`.
  Done when: a fresh download loads and reproduces one known validation output.

## Phase 3: Evaluation and verification (C1, C2, C3, B4)

- [x] **3.1 Verifier.** `verifier.py` parses the JSON, checks the label is allowed for the criterion type, and checks IDs exist. Invalid output counts as wrong and is reported as its own rate. It never guesses a label from missing evidence.
- [x] **3.2 Classification metrics.** `metrics.py`: accuracy, macro-F1, per-class precision/recall/F1 and confusion matrices, for overall, inclusion and exclusion.
  Rule for labels with no examples in a split: marked "n/a" and left out of macro-F1, with the macro-F1 counting them as zero also shown.
- [x] **3.3 Evidence metrics.** Micro and macro precision/recall/F1. Rules, stated in the report: both empty counts as correct in the macro-all score; gold empty with predictions gives precision 0; gold present with no prediction gives recall 0. Report macro-all and macro on rows with gold evidence.
- [x] **3.4 Baselines and uncertainty.** Majority-class predictor and "cites nothing" predictor. Patient-level bootstrap confidence intervals, because the test set has only 8 patients.
- [x] **3.5 Sealed evaluator.** `evaluate.py` refuses the test split unless given `--final`, and logs each such run.
  Done when: metric code passes hand-computed unit tests.

## Phase 4: KV-cached decoding (D1, D2, D3)

- [x] **4.1 Reference path.** `uncached_decoder.py`: full forward pass on the whole sequence every step, `use_cache=False`.
- [x] **4.2 Cached decoder.** `cached_decoder.py`, written as an explicit loop. Prefill runs the full prompt. Each later step sends only the new token, with an extended attention mask, the next `position_ids` taken from the mask, and `cache_position`. The cache object is passed in and updated by hand.
- [x] **4.3 State checks.** Assert the cache length equals tokens processed at every step. Log cache bytes per step to show growth. Test a left-padded prompt to prove masking and positions are correct.
- [x] **4.4 Equivalence tests.** At least 3 input lengths by 3 output lengths under greedy decoding, with EOS ignored so lengths are exact. Compare tokens, record maximum logit difference per step, and run in float32 and float16. Use `model.generate` as a third check. Investigate and write up any mismatch.
- [x] **4.5 Benchmark.** `bench_cache.py`: warm-up run, `torch.cuda.synchronize()` around timers, 5 repeats, median and spread, for prefill, decode, tokens/s and peak GPU and CPU memory.
  Done when: results are in `results/bench_cache.json`, and the report explains positions, masking, cache growth and numerical differences.

## Phase 5: Tiled attention with online softmax (E1, E2, E3)

- [x] **5.1 Reference attention.** `reference.py`: scaled by the square root of d, causal and padding masks applied before softmax, stable softmax by subtracting the row maximum, safe handling of fully masked rows.
- [x] **5.2 Tiled attention.** `tiled_online.py`: loop over query and key tiles, applying the mask per tile. For each row keep a running maximum `m`, normalizer `l` and output `O`. When `m` rises, rescale old `l` and `O` by `exp(m_old - m_new)`. Divide `O` by `l` at the end. No fused backend.
- [x] **5.3 Numerical tests.** Multiple shapes (including length 1 and non-square), tile sizes, causal, padding and no mask, large score magnitudes, float32 and float64. Report maximum and mean absolute error against the reference, and compare with `F.scaled_dot_product_attention`.
- [x] **5.4 Performance.** `bench_attention.py`: runtime and peak memory on CPU and GPU across lengths and tile sizes.
  Done when: the report explains honestly why Python loops, repeated small allocations and no fused kernel can make the educational version slower.
- [x] **5.5 Stretch.** Swap the tiled attention into the real model and confirm identical generated tokens on one prompt.

## Phase 6: Speculative decoding (F1, F2, F3)

- [ ] **6.1 Pair justification.** Identical tokenizer (already verified), memory table, and comparison of base draft against fine-tuned draft (R6).
- [x] **6.2 Algorithm.** `speculative.py`, with cache rule "cache holds every token except the last one". Each round: the draft proposes k tokens; the target checks them in one forward pass; the longest matching prefix is accepted; the target's own next token is added as the correction or bonus token; both caches are cropped to match. EOS and length limits are handled mid-round.
- [x] **6.3 State tests.** Assert the caching rule on both models every round, including all-accepted, all-rejected and EOS-mid-draft cases.
- [x] **6.4 Equivalence tests.** Token equality against the Phase 4 greedy decoder on many prompts, and Hugging Face assisted generation as a reference check only.
- [x] **6.5 Benchmark.** k of 2, 4 and 8, on real short JSON outputs and on longer generations. Report acceptance rate, accepted tokens per round, latency, tokens/s, speedup or slowdown and memory, using the same timing method as Phase 4.
  Done when: any slowdown is explained with measured draft and verification costs.

## Phase 7: Final runs on Kaggle (B4, C1, C2, C3, D3, E3, F3)

- [x] **7.1 Freeze.** Commit the chosen configuration before opening the test set. Tag it `final-config`.
- [x] **7.2 `[YOU]` Run `kaggle/final_eval_and_benchmarks.ipynb`.** Before/after evaluation on identical validation and test examples, all baselines, then all three benchmarks.
- [x] **7.3 Commit outputs.** Metrics, predictions and logs go to `results/`. No model, prompt or threshold changes after this point.

## Phase 8: Error analysis (C4)

- [ ] **8.1** Analyze at least 8 distinct errors, each linked to annotation ID, gold, prediction, cited sentences and cause. Categories to cover: inclusion/exclusion polarity flip, missing information treated as yes or no, unsupported citation, right label with wrong evidence, contradictory patient facts, not-applicable confusion, temporal or negation mistakes, abbreviation misread, invalid output.
  Done when: `results/error_analysis.md` is written and carried into the report.

## Phase 9: Browser application (G1, G2, G3, G4)

Design: minimal, futuristic, simple dark theme. A landing page with motion components from 21st.dev leads into a guided flow, one screen per step.

| Screen | Purpose |
|---|---|
| Landing | Orb, name and start button only |
| 1. Patient note | Paste or upload, with validation |
| 2. Sentence review | Confirm the numbered sentences evidence will cite |
| 3. Trial criteria | Enter inclusion and exclusion criteria, load an example or a trial ID |
| 4. Run | Choose decoding mode, watch per-criterion progress |
| 5. Results | Overall preliminary result and a summary of every criterion |
| 6. Criterion inspection | Decision, highlighted evidence, explanation, missing information, raw output, timing |
| Inference Lab | Live decoding and attention comparison |
| Reports | Saved screenings, with HTML and JSON download |

Frontend: React, Vite, Tailwind and the `thinking-orbs` package, with bundled fonts so it works offline. Styled after the owner's earlier app: black background, soft center glow, glass pill buttons, thin cards. Home on every screen, Back and Continue through the steps, and browser navigation supported. The built files are committed and served by FastAPI, so running the app needs no Node. MLflow is not used; CSV and JSON logs meet the logging requirement.

- [x] **9.1 Backend.** FastAPI with the screening pipeline, status endpoint, single-job queue and per-criterion progress updates. Uses our own decoders.
- [x] **9.2 Memory strategy.** Target at float16 on GPU, matching the evaluated model. Draft on GPU if it fits, otherwise CPU. Catch out-of-memory errors and show a clear message.
- [x] **9.3 Launcher.** `start_app.bat`: double-click, check environment and weights, start the server, open the browser, show a loading screen.
- [x] **9.4 Screening flow.** Paste or upload a note, review numbered sentences, enter inclusion and exclusion criteria, or load an example or a ClinicalTrials.gov ID.
- [x] **9.5 Results.** A card for each criterion showing label, cited sentence IDs, evidence highlighted in the note, explanation, and a missing-information panel.
- [x] **9.6 Overall result.** Explicit rule, displayed in the app: any "excluded" or "not included" gives *likely ineligible*; otherwise any "not enough information" gives *needs review*; otherwise *potentially eligible*. Always marked preliminary and not ground truth.
- [x] **9.7 Benchmark view.** Run uncached, cached and speculative decoding live on the current input, showing timings, tokens/s, memory, token equality and acceptance rate. Run reference against tiled attention live and show error and time.
- [x] **9.8 Validation and recovery.** Empty note, oversized note with a visible warning and no silent truncation, missing criteria, bad file, bad trial ID, network failure, GPU out of memory.
- [x] **9.9 Export.** Downloadable HTML and JSON report with provenance: model and adapter revision, decoding mode, time and input hash. Saved in `reports/`.
- [x] **9.10 Notice.** "Educational screening prototype, not a clinical enrollment decision" appears in every exported report.
- [x] **9.11 Rehearsal.** Fresh start from the launcher with unseen notes on the laptop. Fix anything found.

## Phase 10: Paper (A1, A2, A3, H1)

- [ ] **10.1** IEEE LaTeX in `report/`, with all required sections from title through references.
- [ ] **10.2** Literature comparison of at least 5 sources, with limitations, tied to our choices (11 papers are ready).
- [ ] **10.3** Model comparison table: architecture, parameters, tokenizer, context window, license, memory.
- [ ] **10.4** Decisions and failures from `docs/DECISIONS.md`.
- [ ] **10.5** Architecture, attention, decoding and application diagrams.
- [ ] **10.6** Generated figures and tables: training curves, confusion matrices, before/after, evidence scores, cache, attention and speculative benchmarks, error table, app screenshots. Each one cited and interpreted in the text.
- [ ] **10.7** Limitations: small test set, rare labels, single annotator effects, hardware differences, screening-only use.
- [ ] **10.8** AI-use appendix, written honestly.
- [ ] **10.9 `[YOU]`** Compile on Overleaf and read it through.

## Phase 11: Submission and defense (H2, H3)

- [ ] **11.1** README: setup, Kaggle steps, app startup, adapter link, results map.
- [ ] **11.2** Clean-clone test following the README exactly.
- [ ] **11.3 `[YOU]`** Record the 5-7 minute demo. Script: problem, unseen note, evidence inspection, inference comparison, invalid input recovery, export.
- [ ] **11.4** `docs/DEFENSE.md`: plain explanations of the model, cache, attention, decoding and results, and code locations.
- [ ] **11.5 `[YOU]`** Practice live changes: tile size, draft length k, prompt wording, a new unseen case, and explaining each decoder line by line.
- [ ] **11.6 `[YOU]`** Submit on Google Classroom: PDF, GitHub URL, instructions, adapter link, manifest, benchmark logs, video link.

## Rubric coverage

| Item | Marks | Evidence | Phase |
|---|---|---|---|
| A1 Literature | 5 | Paper related work and comparison | 10 |
| A2 Model choice | 5 | Model table, check log | 10 |
| A3 Decisions | 5 | `docs/DECISIONS.md`, experiment table | 2, 10 |
| B1 Dataset and split | 5 | Manifest, leakage tests | 1 |
| B2 Data processing | 5 | `prompts.py`, allowlist test, truncation audit | 1 |
| B3 Fine-tuning | 5 | Script, config, curves, adapter | 2 |
| B4 Before/after | 5 | Same examples, same decoding, discussion | 2, 7 |
| C1 Label correctness | 5 | Verifier, held-out accuracy and macro-F1 | 3, 7 |
| C2 Detailed metrics | 5 | Per-class tables, matrices, inclusion/exclusion | 3, 7 |
| C3 Evidence | 5 | Original IDs, micro and macro, empty-set rules | 3, 7 |
| C4 Explanations and failures | 5 | Linked evidence, at least 8 analyzed errors | 2, 8 |
| D1 Cached decoder | 5 | `cached_decoder.py`, state checks | 4 |
| D2 Correctness | 5 | 3x3 equivalence tests, logit differences | 4 |
| D3 Benchmark | 5 | Warm-up, synchronized, repeated timing | 4, 7 |
| E1 Reference attention | 5 | `reference.py`, tests | 5 |
| E2 Tiled attention | 5 | `tiled_online.py`, no fused backend | 5 |
| E3 Validation | 5 | Max and mean error, honest performance | 5, 7 |
| F1 Draft and target | 5 | Tokenizer test, memory table | 6 |
| F2 Algorithm | 5 | `speculative.py`, state tests | 6 |
| F3 Analysis | 5 | Equality, acceptance, speedup or slowdown | 6, 7 |
| G1 Browser app | 5 | Launcher, validation, error handling | 9 |
| G2 Workflow | 5 | Unseen notes, evidence, overall result | 9 |
| G3 Inference view | 5 | Live comparison | 9 |
| G4 Usability and export | 5 | Report export, recovery from bad input | 9 |
| H1 Paper | 5 | IEEE PDF | 10 |
| H2 Reproducibility | 5 | Clean-clone test, video | 11 |
| H3 Defense | 5 | `docs/DEFENSE.md`, your own practice | 11 |
