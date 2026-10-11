# Project Status Report

## Current Architecture (2026-10-12): library layers, from-scratch dropped

**Decision:** the from-scratch constraint (roadmap Appendix C) was dropped by the project owner to pursue library-based models. The roadmap lists this as a professor requirement for graded work. Confirm with the professor before any result is used for grading; the report must state this change.

**What changed**
* `src/models/models.py` now uses PyG layers: `GraphSAGEModel` (`SAGEConv`, mean, ReLU + dropout), `FraudGAT` (`GATConv`), and `FraudCamouflageGNN` (`GATConv` with a trust edge feature).
* **Camouflage mechanism, redesigned:** one `score_head` (MLP on input features) predicts a per-node fraud score, supervised by the auxiliary loss on seed nodes. Trust per edge = exp(-|s_u - s_v|) is passed to each `GATConv` as an edge feature. `GATConv` adds a learned per-head term linear in trust to the attention logits before LeakyReLU. The sign is learned, not constrained: check `trust_gains()`. Self-loops receive the mean trust (PyG default), not the maximum.
* **LightGBM baseline added:** `src/train_tabular.py`, `configs/lgbm_tabular.yaml`, LightGBM 4.7.0 pinned. Node features only, same time-based masks, `scale_pos_weight` on the train split, early stopping on validation PR-AUC.
* **Node features changed (`src/data/build_graph.py`):** `TransactionAmt` is log1p-transformed; categoricals are train-split frequencies instead of ordinal codes; five label-free per-card features (`card_prior_count`, `card_prior_amt_mean`, `card_prior_amt_std`, `card_amt_dev`, `card_log_gap`) use only earlier transactions on the same `card1`. The time-based masks and edge construction are unchanged.
* **Training objective changed (`src/train.py`, `src/utils/losses.py`):** the loss is still weighted BCE on the seed nodes, plus two surrogates of the metrics we report: a pairwise logistic ROC-AUC term (`rank_loss_weight`, default 1.0) and a smooth average-precision PR-AUC term (`ap_loss_weight`, default 1.0, sigmoid temperature `ap_temperature`, default 1.0). Checkpoint selection and early stopping now use `selection_score` = mean of val PR-AUC and val ROC-AUC, not PR-AUC alone. Setting both weights to 0 gives BCE alone; the selection rule still changes. The weights and temperature are untuned defaults, not results.
* Hand-written layers in `src/models/layers/` are no longer used by the models. Kept as reference, not deleted.

**Consequences**
* **All earlier SAGE/GAT/camouflage numbers are from old architectures and are invalid for comparison.** The feature change also invalidates the LightGBM baseline run from before it, because the feature matrix changed. Rerun every baseline on the new code and graph. The earlier SAGE model had no BatchNorm or dropout, despite what earlier status text said.
* `NeighborLoader` needs `pyg-lib` or `torch-sparse` on the training machine. Kaggle's image should have them; the local sandbox does not, so `train.py` was not run end-to-end here.

**Verification (2026-10-12, CPU, torch 2.14.1, PyG 2.8.0.post1, LightGBM 4.7.0)**
* `pytest tests/`: 34 passed (legacy layer tests, PyG model tests incl. trust-feature check, LightGBM tests, 5 `build_graph` tests: hand-computed card features, a no-look-ahead check, missing-card handling, train-only frequency encoding, and an end-to-end run on synthetic CSVs; and 12 objective tests: the pairwise loss matches a brute-force pair average, the smooth-AP loss converges to exact `average_precision_score` at low temperature, both surrogates rank perfect orderings low and reversed ones high, both return zero with gradients when a batch lacks a class, the selection score, and resume refusing a checkpoint written under a different objective). A deliberately leaky variant of the card features fails the no-look-ahead test.
* `train.py` loop run end to end on synthetic graphs with a stand-in for `NeighborLoader` (the sandbox has no `pyg-lib`): camouflage and SAGE with the new objective, resume from a checkpoint (renamed key `best_val_score`), and both new weights set to 0. On a synthetic graph where fraud depends on amount, both objectives learn the signal at similar rates (one seed, 15 epochs; val ROC-AUC about 0.81 for both). This checks the wiring, not whether the objective helps on IEEE-CIS.
* Full-batch sanity run on a synthetic graph: all three models train without NaN; camouflage trust gains positive for every head. **Synthetic only; not evidence about IEEE-CIS.**

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
  * **[Corrected 2026-10-12]** The previously noted "unbounded exponential" risk does not exist. `trust = torch.exp(-score_diff)` with `score_diff = |s_src - s_dst| >= 0` lies in (0, 1], so it cannot explode; at worst it underflows toward 0 for very large disagreements. Nothing here needs fixing for this reason.
  * **[Fixed 2026-10-12, not yet verified by a run]** Label leak in the auxiliary loss (`src/train.py`): `aux_loss` was computed over every node in the NeighborLoader batch, including sampled neighbors that can be validation or test nodes, so their labels supervised `score_head`. It is now restricted to the seed nodes (`[:batch.batch_size]`), matching the main loss. No camouflage-GAT result existed before this fix, so no reported number was affected.
  * **[Known design choice, not changed]** Self-loop edges always receive trust = exp(0) = 1, the maximum bonus, so each node is favored over neighbors whose predicted scores disagree. This may suppress the neighborhood signal the mechanism is meant to filter. Left as-is for the first run; candidate for an ablation.
  * **[Fixed 2026-10-12, not yet verified by a run]** `tests/test_camouflage.py` referenced `sim_weight`, which the layer no longer defines, so `test_camouflage_gradient_flow` and `test_camouflage_behavior` raised `AttributeError`. `test_camouflage_behavior` also tested the abandoned cosine-similarity mechanism. Both now test the score-agreement trust term. The earlier claim "unit tests passing" was incorrect.

## Kaggle Training Runs Completed To Date
* **SAGE baseline (`sage_baseline.yaml`):** trained twice.
  * First run (pre-NaN-imputation-fix): best PR-AUC 0.4106 / ROC-AUC 0.8591 at epoch 6, early-stopped at epoch 16.
  * Second run (post-NaN-imputation-fix, current baseline): best PR-AUC 0.4147 / ROC-AUC 0.8618 at epoch 15, early-stopped at epoch 25. Results committed to `experiments/sage_baseline/`.
* **GAT baseline (`gat_baseline.yaml`):** trained three times.
  * First attempt (Version 5): invalidated by user error — the training cell still had `sage_baseline.yaml` configured, so this was actually a second SAGE run, not GAT. (Initially misattributed to the `load_config()` bug before the actual cell content was checked directly; corrected once verified.)
  * Second attempt: genuine GAT run at `lr: 0.005` (unmodified default), exhibited severe training instability — sharp single-epoch collapses (e.g. ROC-AUC dropping to ~0.54-0.59 in isolated epochs), best PR-AUC only 0.1478, well below SAGE. Diagnosed as caused by absent gradient clipping combined with an aggressive learning rate for an attention-based architecture.
  * Third attempt (post gradient-clipping + `lr: 0.001` fix, pre-NaN-imputation-fix): stable training, no collapses, best PR-AUC 0.3157 / ROC-AUC 0.8103 at epoch 6, early-stopped at epoch 16. Verified via direct checkpoint inspection (`torch.load` + epoch/PR-AUC cross-check against logs). Results committed to `experiments/gat_baseline/`.
  * **Not yet rerun against the NaN-imputation fix** — the committed GAT result above predates that fix, unlike the current SAGE result. This is an inconsistency that should be resolved before treating the two as a fair comparison (see Missing Tasks).
* **Camouflage-GAT (`camo_gat.yaml`):** not yet trained. **[Corrected 2026-10-12]** Config is already at `lr: 0.001`, and `aux_loss_weight: 0.3` is now written explicitly into the YAML so it is recorded in `config.json`. Previously it was only a hardcoded default in `train.py`.

## Missing / Incomplete Tasks
* **Cleanup pending (owner action):**
  * `report/_archive/` still contains `final_report_FABRICATED_2026-08-26.md` and `results_FABRICATED_2026-08-26.md`. Their 3-seed numbers (e.g. camouflage PR-AUC 0.3907 ± 0.0144) do not correspond to any run in this repo; `run_experiments.py` has never completed end-to-end. Delete this folder. Do not cite any number from it. (The deletion was not performed from the working session.)
  * `src/agent.py` is outside the roadmap. Fixed on 2026-10-12 to use the `camo_gat_baseline` checkpoint path and to unwrap the checkpoint's `'model'` key. Not tested (needs `GOOGLE_API_KEY` and a trained camouflage checkpoint). Keep or remove; decide and document.
  * `report/related_work.md` twist paragraph rewritten on 2026-10-12 to describe score-agreement gating rather than cosine similarity.
  * Metrics: `recall` added to `compute_metrics`; `run_experiments.py` now reports sample std (ddof=1).
  * `train.py` default fanout is `[25, 10]`; the roadmap and earlier status text say `[15, 15]`. Use the real value in the report.
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