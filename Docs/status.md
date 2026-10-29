# Project status

Last updated: 2026-10-29. Decisions are in `Docs/decisions.md`.

## Summary
The project is a graph-based fraud detector on IEEE-CIS data, with LightGBM as a non-graph baseline and a camouflage-aware GAT as the proposed model. The comparison protocol is pre-registered in `Docs/decisions.md` (commit `cdebe6d`, with the tuning-grid line added in commit `522d999`). No valid final results exist yet. Earlier numbers came from a pipeline that has since changed.

## Phase status
- **Phase 0 (setup, repo):** done. `.gitignore` anchors the data rule to `/data/`, ignores checkpoints by pattern, and ignores `/*.zip`.
- **Phase 1 (dataset, EDA):** done. IEEE-CIS chosen. `notebooks/eda.ipynb` covers class imbalance and entity sharing.
- **Phase 2 (graph construction):** code rewritten and tested locally. Built on Kaggle: `graph_causal.pt` and `graph_full.pt` written, 590,540 transactions, 436 node features. Causal direction check passed (0 edges from later to earlier transactions).
  - Provenance: the graph was built by Kaggle notebook `notebookec7033b03b` (2026-10-28), from code version 21366554, which matches `src/data/build_graph.py` at commit `2e80f6f` or later.
- **Phase 3 (layers):** models use PyTorch Geometric layers. Hand-written SAGE and GAT remain as reference code. Their equivalence test with the library layers is not yet written.
- **Phase 4 (baselines):** LightGBM with node features and with causal entity-history features is written, not yet run on the current graph. GAT has not been rerun on the current pipeline.
- **Phase 5 (training harness):** rewritten. Checkpoints are selected on validation PR-AUC only. Validation sampling is repeatable. Test evaluation runs once per run, saves per-row test scores, and reports precision and recall at review budgets of 1%, 5% and 10%.
- **Phase 6 (literature):** `report/related_work.md` is written.
- **Phase 7 (camouflage model):** `FraudCamouflageGNN` in `src/models/models.py` is the canonical version, with `detach_trust` (default true). Tests cover the detached and undetached gradient paths. Its only earlier full-graph run, of the removed hand-written version, collapsed to ROC-AUC about 0.5. The new version has not been trained.
- **Phase 8 (experiments, report):** not started on the current pipeline. `report/results.md` says so.
- **Phase 9 (comparison):** `scripts/paired_bootstrap.py` is written. It compares GNN and LightGBM test scores on identical rows.

## Commit history for this stage
| Commit | Content |
|---|---|
| `bc8c061` | `detach_trust` option, per-row test scores, causal LightGBM features, updated gradient tests |
| `eb187ad` | Camouflage nodetach config, LightGBM configs, paired bootstrap script |
| `cdebe6d` | Pre-registered comparison protocol and review budgets (`Docs/decisions.md`) |
| `b47b614` | Status update with the decisions commit hash |
| `522d999` | Tuning-grid line added to `Docs/decisions.md` (committed alone, before any final run) |

## Smoke tests
- Not yet run. Each model should run one epoch on the causal graph in the Kaggle sweep notebook. The smoke folders (`experiments/smoke_*`) must be deleted afterwards and never reported.

## Fraud rate per split (current causal graph, Kaggle build)
| Split | Rows | Fraud rate |
|---|---|---|
| Train (first 70%) | 413,378 | 0.0352 |
| Validation (next 15%) | 88,581 | 0.0343 |
| Test (last 15%) | 88,581 | 0.0348 |

The rates are close, so a difference in fraud rate between periods does not explain the earlier validation-to-test gap. The gap is still open and must be explained before the final runs.

## Feature-drift check
- Not yet run. Compare validation and test feature means with the training split, in training-SD units, on `graph_causal.pt`. Shifts above about 0.5 SD in many features would point to drift. If few features shift, drift is recorded as unlikely to explain the gap.

## Results from the friend's Kaggle run (old pipeline)
These logs are the only record of a training run. They used the earlier code: the non-causal graph (45.5 million edges, 430 features), the old selection rule (mean of PR-AUC and ROC-AUC), and random neighbour sampling at evaluation. They are a sanity check of the pipeline, not a result.

| Item | Value |
|---|---|
| Model / commit | SAGE, commit `3bb56af` |
| Environment | Kaggle, Tesla T4, Python 3.13, torch 2.11.0+cu128, PyG 2.8.0.post1 |
| Best validation PR-AUC | 0.4199 (epoch 9) |
| Early stop | epoch 19 (patience 10) |
| Test PR-AUC | 0.2268 |
| Test ROC-AUC | 0.8404 |
| Test recall at 0.5 threshold | 0.597 |

Two things the logs show:
- **Training is deterministic.** The run appears twice in the log with identical numbers.
- **The validation-to-test gap is large.** Validation PR-AUC is 0.42 and test PR-AUC is 0.23, while ROC-AUC barely moves (0.85 to 0.84). The split fraud rates do not explain it, so the cause is still open.

## Known open issues
1. **Validation-to-test gap** (above). The fraud rates match across splits. The feature-drift check and the old pipeline's evaluation path are the next checks.
2. **Lost training notebook.** `gnn_kaggle_train_v2.ipynb` and `gnn_fraud_full_run.ipynb` are not available. The replacement is a short runner notebook that reads the code from a Kaggle dataset.
3. **Git history.** Four commits on `feat/rank-objective` carry a real name and email. They are unmerged, so they can be rewritten, but only with the friend's agreement.
4. **Camouflage score-head choice.** `detach_trust` is decided by validation PR-AUC over the sweep, with a tie default of detach. Not yet run.
5. **Review budget is an assumption.** The 1%, 5% and 10% budgets were chosen on 2026-10-28, before any budget-level result existed. At 1% of the test split (about 886 reviews, roughly 31 fraud on average), precision is noisy, so intervals must be reported with it.
6. **Loss weights and `aux_loss_weight` are not tuned.** This is a deliberate limit for the deadline, stated in the decisions file. It should appear in the report as a limitation.
7. **Commit identity.** `Docs/decisions.md` asks for a placeholder identity on new commits, but recent commits use the identity in the local git config. Either reword that entry or use the placeholder from now on.

## Next steps
1. Run the smoke tests in the Kaggle sweep notebook, then delete the `experiments/smoke_*` folders.
2. Run the feature-drift check on `graph_causal.pt`.
3. Run LightGBM with node features and with causal features, once each. These are the reference the GNNs must beat.
4. Run the camouflage sweep (both `detach_trust` settings) and the SAGE and GAT sweeps on validation PR-AUC, using the same learning-rate grid.
5. Run the final experiments: frozen configs, 3 seeds each, `evaluate.py` once per run.
6. Run the paired bootstrap for each GNN against the causal LightGBM baseline, per seed and on the seed average.
7. Run the hand-written versus library layer comparison on validation PR-AUC.
8. Write the report from `experiments/*/test_results.json` and the bootstrap output only.