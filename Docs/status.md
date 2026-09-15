# Project Status Report

## Completed Phases
* **Phase 0 (Environment & Repo Setup):**
  * Repository structure initialized, GitHub repo created, `.gitignore` setup.
  * Environment setup and `requirements.txt` generated.
  * Random seed utility implemented (`src/utils/seed.py`).
  * `README.md` stubbed with project summary and novelty statement.
* **Phase 1 (Problem framing & dataset decision):**
  * Dataset chosen: IEEE-CIS (documented in `README.md`).
  * Problem and novelty statements defined.
  * **[Completed]** EDA notebook (`notebooks/eda.ipynb`) created analyzing class imbalance and feature cardinality.
* **Phase 2 (Graph construction pipeline):**
  * Graph construction logic implemented in `src/data/build_graph.py`.
  * **[Completed]** `addr2` excluded from both node features and edge relations (low signal: only 74 distinct values across 590,540 transactions; contributed ~52M low-information edges).
  * **[Completed]** `DEGREE_CAP` lowered from 100 to 20 to keep total edge count tractable for a single T4 GPU (capped/sampled groups instead of dropping high-degree groups entirely).
  * **[Fixed]** Numeric NaN imputation changed from a raw `0` fill to each column's train-split median, computed before `StandardScaler` fitting. The previous raw-0 fill created artificial outlier values post-scaling for columns with non-zero natural means and substantial missingness, disproportionately harming attention-based models (GAT/camouflage-GAT) whose attention logits are more outlier-sensitive than SAGE's mean aggregation.
* **Phase 3 (GraphSAGE from scratch):**
  * SAGEConv layer implemented (`src/models/layers/sage.py`), including an explicit zero-degree self-loop fix (via `torch.where`, autograd-safe) for isolated nodes.
  * Unit tests written (`tests/test_sage.py`).
* **Phase 4 (GAT from scratch):**
  * GATConv layer implemented (`src/models/layers/gat.py`), including a numerically-stable softmax (max-subtraction via `scatter_reduce_`).
  * Unit tests written (`tests/test_gat.py`).
* **Phase 5 (Baseline training & evaluation harness):**
  * **[Completed]** `train.py` and `evaluate.py` created.
  * **[Completed]** Logging and `configs/` structure established for ablations.
  * **[Completed]** Class imbalance handling via `pos_weight` in BCE loss.
  * **[Completed]** Checkpoint/resume support added to `train.py`: `--resume_dir` flag, config-compatibility validation before resuming, `last_model.pt`/`best_model.pt` saved every epoch with full optimizer/epoch/early-stopping state, `metrics.json` rewritten every epoch (not just at the end) so a mid-run interruption preserves progress. Built and tested against Kaggle's 12-hour session limit.
  * **[Fixed]** `load_config()` bug: `config.update(cli_args)` was applied unconditionally after YAML values, but `cli_args` always contained every argparse field (since every flag has a default), silently reverting YAML-set values (e.g. `model`, `lr`) back to argparse defaults on every run. Fixed using a second `argparse.SUPPRESS`-based parse to unambiguously detect which flags were actually passed on the command line, including the edge case where an explicit override coincides with the parser's own default value.
  * **[Fixed]** Training instability in GAT: no gradient clipping existed in the training loop, and `GATConv`'s exponential-based attention softmax is more sensitive to outlier gradients than SAGE's mean aggregation, causing sharp single-epoch performance collapses (e.g. PR-AUC 0.148 → 0.049 in one epoch). Fixed by adding `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` (applies to all three models), combined with lowering `gat_baseline.yaml`'s learning rate from 0.005 to 0.001.
  * **[Fixed]** `evaluate.py` bug: `torch.load(model_path)`'s full checkpoint dict (`{'model', 'optimizer', 'epoch', ...}`) was passed directly into `model.load_state_dict()` instead of unwrapping the `'model'` key first, causing a `RuntimeError` on every invocation. This silently blocked all test-set evaluation for every model, since `run_experiments.py` calls `evaluate.py` immediately after every training run. Fixed; test-set evaluation is now unblocked but has not yet actually been run (see Missing Tasks).
  * Debug print of the fully-resolved config and `exp_dir` added to `train.py`'s startup, to make config-resolution bugs immediately visible in logs going forward.
* **Phase 6 (Literature deep-dive: camouflage-resistant GNNs):**
  * **[Completed]** `report/related_work.md` written, including mechanism summaries and our proposed novelty.
* **Phase 7 (Design & implement the novel module):**
  * Camouflage GAT module implemented (`src/models/layers/camouflage_gat.py`), using a label-supervised score-agreement trust signal (following the CARE-GNN approach, Dou et al. 2020) rather than raw feature similarity, specifically to resist fraud rings that deliberately shape features to look legitimate.
  * Unit tests written and passing (`tests/test_camouflage.py`).
  * **[Noted, not yet fixed]** `CamouflageGATConv` has a second, additive instability risk beyond what the Phase 5 gradient-clipping fix addresses: `trust = torch.exp(-score_diff)` is a second, unbounded exponential stacked on top of the attention softmax's own exponential, fed by an untrained `score_head` early in training. Gradient clipping should help but has not been confirmed sufficient specifically for this term, since camouflage-GAT has not been trained yet.

## Kaggle Training Runs Completed To Date
* **SAGE baseline (`sage_baseline.yaml`):** trained twice.
  * First run (pre-NaN-imputation-fix): best PR-AUC 0.4106 / ROC-AUC 0.8591 at epoch 6, early-stopped at epoch 16.
  * Second run (post-NaN-imputation-fix, current baseline): best PR-AUC 0.4147 / ROC-AUC 0.8618 at epoch 15, early-stopped at epoch 25. Results committed to `experiments/sage_baseline/`.
* **GAT baseline (`gat_baseline.yaml`):** trained three times.
  * First attempt (Version 5): invalidated by user error — the training cell still had `sage_baseline.yaml` configured, so this was actually a second SAGE run, not GAT. (Initially misattributed to the `load_config()` bug before the actual cell content was checked directly; corrected once verified.)
  * Second attempt: genuine GAT run at `lr: 0.005` (unmodified default), exhibited severe training instability — sharp single-epoch collapses (e.g. ROC-AUC dropping to ~0.54-0.59 in isolated epochs), best PR-AUC only 0.1478, well below SAGE. Diagnosed as caused by absent gradient clipping combined with an aggressive learning rate for an attention-based architecture.
  * Third attempt (post gradient-clipping + `lr: 0.001` fix, pre-NaN-imputation-fix): stable training, no collapses, best PR-AUC 0.3157 / ROC-AUC 0.8103 at epoch 6, early-stopped at epoch 16. Verified via direct checkpoint inspection (`torch.load` + epoch/PR-AUC cross-check against logs). Results committed to `experiments/gat_baseline/`.
  * **Not yet rerun against the NaN-imputation fix** — the committed GAT result above predates that fix, unlike the current SAGE result. This is an inconsistency that should be resolved before treating the two as a fair comparison (see Missing Tasks).
* **Camouflage-GAT (`camo_gat.yaml`):** not yet trained. Config still has `lr: 0.005` (the same value that caused GAT's instability) and has not been lowered yet, despite this being flagged as needed.

## Missing / Incomplete Tasks
* **Phase 8 (Experiments & ablations) — in progress, not complete:**
  * Rerun GAT baseline against the NaN-imputation-corrected graph, so SAGE and GAT results are directly comparable (both post-fix). Currently only SAGE has been re-run since that fix.
  * Update `configs/camo_gat.yaml`'s learning rate from 0.005 to 0.001, matching the GAT fix, before training camouflage-GAT.
  * Do a real hyperparameter pass for SAGE and GAT (at minimum: 2-3 `hidden_channels` values, a small learning-rate grid) rather than relying on single, largely-default-hyperparameter runs. Current results are not the product of any tuning beyond the one GAT learning-rate correction.
  * Train camouflage-GAT for the first time, watching closely for the untested `torch.exp(-score_diff)` instability risk noted above during its first several epochs.
  * Run the ablation configurations (`sage_baseline.yaml`, `gat_baseline.yaml`, `camo_gat.yaml`) across 3 random seeds each via `run_experiments.py`, once reasonable hyperparameters have been chosen per model. Not yet run at all — all results to date are single-seed.
  * Actually execute test-set evaluation (`evaluate.py`) for every trained model. This has never been run successfully — it was silently broken until the checkpoint-unwrapping fix above, and has not been invoked since being fixed. All results reported to date are validation-set metrics only; `test_mask` in `graph.pt` has not been used at all yet.
  * `run_experiments.py`'s multi-seed sweep depends on `evaluate.py` producing `test_results.json` correctly — this dependency chain is now unblocked in principle but has not been exercised end-to-end even once.
* **Report:**
  * The final report synthesizing results, figures, and limitations still needs to be written.
  * Should explicitly document the bugs found and fixed during Phase 8 (config-resolution, training instability, evaluation-blocking bug, NaN-imputation), since they materially affected which results are valid and comparable.

## What is Next
1. **Close the gap between SAGE and GAT baselines:** rerun GAT against the current (NaN-imputation-fixed) graph so both baselines reflect the same data pipeline.
2. **Hyperparameter pass:** run a small `hidden_channels` / learning-rate sweep for SAGE and GAT before locking in "final" baseline configs.
3. **Fix and train camouflage-GAT:** lower its learning rate to 0.001 first, then run it, watching for the flagged secondary instability risk.
4. **Run the full multi-seed sweep:** execute `run_experiments.py` across all three configs × 3 seeds, now that `evaluate.py` is fixed.
5. **Evaluate on the held-out test set** for every model, not just validation — report both.
6. **Compile Results:** record mean ± standard deviation for PR-AUC and ROC-AUC across the 3 seeds for each configuration, on both validation and test sets.
7. **Write the Final Report:** combine the novelty statement, related work, experiment results, and a candid account of the pipeline issues found and fixed along the way into a final deliverable report.