# Project status

Last updated: 2026-10-29. Decisions are in `Docs/decisions.md`.

## Summary
The project is a graph-based fraud detector on IEEE-CIS data, with LightGBM as a non-graph baseline and a camouflage-aware GAT as the proposed model. The comparison protocol is pre-registered in `Docs/decisions.md` (commit `cdebe6d`, with the tuning-grid line added in commit `522d999`). Both LightGBM baselines have now been run on the causal graph. No GNN has been trained on the full protocol yet. Earlier numbers came from a pipeline that has since changed.

## Phase status
- **Phase 0 (setup, repo):** done. `.gitignore` anchors the data rule to `/data/`, ignores checkpoints by pattern, and ignores `/*.zip`.
- **Phase 1 (dataset, EDA):** done. IEEE-CIS chosen. `notebooks/eda.ipynb` covers class imbalance and entity sharing.
- **Phase 2 (graph construction):** done. Built on Kaggle: `graph_causal.pt` and `graph_full.pt` written, 590,540 transactions, 436 node features. Causal direction check passed (0 edges from later to earlier transactions).
  - Provenance: the graph was built by Kaggle notebook `notebookec7033b03b` (2026-10-28), from code version 21366554, which matches `src/data/build_graph.py` at commit `2e80f6f` or later.
- **Phase 3 (layers):** models use PyTorch Geometric layers. Hand-written SAGE and GAT remain as reference code. Their equivalence test with the library layers is not yet written.
- **Phase 4 (baselines):** LightGBM done for both feature sets (see table below). GAT and SAGE sweeps are not yet run on the current pipeline.
- **Phase 5 (training harness):** rewritten. Checkpoints are selected on validation PR-AUC only. Validation sampling is repeatable. Test evaluation runs once per run, saves per-row test scores, and reports precision and recall at review budgets of 1%, 5% and 10%.
- **Phase 6 (literature):** `report/related_work.md` is written.
- **Phase 7 (camouflage model):** `FraudCamouflageGNN` in `src/models/models.py` is the canonical version, with `detach_trust` (default true). Tests cover the detached and undetached gradient paths. Its only earlier full-graph run, of the removed hand-written version, collapsed to ROC-AUC about 0.5. The new version has been smoke-tested (one epoch) but not trained on the protocol.
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
| `<to fill>` | Merge identity table before causal entity features (`src/train_tabular.py`) |
| `<to fill>` | This update: LightGBM results, primary comparator, smoke outcome |

## LightGBM baselines (first valid runs on the causal graph)
Run on Kaggle, seed 42, `graph_causal.pt`, `train_tabular.py`. Configs were not changed after these runs.

| Baseline | Features | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test precision at 1% | Test recall at 1% | Test recall at 5% | Test recall at 10% |
|---|---|---|---|---|---|---|---|---|
| `lgbm_tabular` (primary comparator) | 436 | 0.6149 | 0.5369 | 0.8919 | 0.8929 | 0.2488 | 0.5955 | 0.7178 |
| `lgbm_causal` (secondary) | 452 | 0.5990 | 0.5288 | 0.8973 | 0.8887 | 0.2744 | 0.5900 | 0.7126 |

Notes:
- `lgbm_causal` adds 16 causal entity-history features. They did not raise validation or test PR-AUC. They raised test ROC-AUC and recall at 1%.
- Both models show a validation-to-test drop of about 0.07 to 0.08 PR-AUC. The drop is shared across feature sets, which points to the test period rather than to either model.
- Before the GNNs are compared, the test numbers must be copied into `experiments/lgbm_tabular/` and `experiments/lgbm_causal/` from the Kaggle session, and committed.

## Smoke tests
- Run on 2026-10-29 in the sweep notebook: one epoch each on the causal graph. All four completed and wrote checkpoints.
- Validation PR-AUC after one epoch (not reported as results): SAGE 0.4019, GAT 0.1325, camouflage detached 0.2519, camouflage nodetach 0.2552.
- The `experiments/smoke_*` folders must be deleted and never reported. Confirm deletion before the sweep.

## Fraud rate per split (current causal graph, Kaggle build)
| Split | Rows | Fraud rate |
|---|---|---|
| Train (first 70%) | 413,378 | 0.0352 |
| Validation (next 15%) | 88,581 | 0.0343 |
| Test (last 15%) | 88,581 | 0.0348 |

The rates are close, so a difference in fraud rate between periods does not explain the validation-to-test gap. The gap is still open and must be explained before the final runs.

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
- **The validation-to-test gap is large.** Validation PR-AUC is 0.42 and test PR-AUC is 0.23, while ROC-AUC barely moves (0.85 to 0.84). The fraud rates do not explain it. The same kind of gap appears in both LightGBM runs, so it is likely a property of the test period.

## Known open issues
1. **Validation-to-test gap** (above). Seen in both LightGBM runs and in the old SAGE run. The feature-drift check is the next check.
2. **Screen for GNNs.** A GNN family whose best validation PR-AUC is below 0.6149 cannot be the primary winner (see `Docs/decisions.md`). The sweep will show whether any family clears that bar.
3. **Lost training notebook.** `gnn_kaggle_train_v2.ipynb` and `gnn_fraud_full_run.ipynb` are not available. The replacement is a short runner notebook that reads the code from a Kaggle dataset.
4. **Git history.** Four commits on `feat/rank-objective` carry a real name and email. They are unmerged, so they can be rewritten, but only with the friend's agreement.
5. **Camouflage score-head choice.** `detach_trust` is decided by validation PR-AUC over the sweep, with a tie default of detach. Not yet run on the protocol.
6. **Review budget is an assumption.** The 1%, 5% and 10% budgets were chosen on 2026-10-28, before any budget-level result existed. At 1% of the test split (about 886 reviews, roughly 31 fraud on average), precision is noisy, so intervals must be reported with it.
7. **Loss weights and `aux_loss_weight` are not tuned.** This is a deliberate limit for the deadline, stated in the decisions file. It should appear in the report as a limitation.
8. **Commit identity.** `Docs/decisions.md` asks for a placeholder identity on new commits, but recent commits use the identity in the local git config. Either reword that entry or use the placeholder from now on.
9. **Kaggle outputs are not in the repo.** LightGBM and GNN results exist only in the Kaggle session until copied. Copy each `test_results.json` and `metrics.json` into the matching `experiments/` folder before committing.

## Next steps
1. Commit this status and the decisions update (two commits, see the instructions in the reply).
2. Copy `experiments/lgbm_tabular/` and `experiments/lgbm_causal/` (JSON files only) from the sweep notebook's output into the local repo, then commit them.
3. Run the feature-drift check in the `gnn-drift-check` notebook and record the result here.
4. Delete the `experiments/smoke_*` folders in the sweep notebook.
5. Run the validation sweep in the sweep notebook: SAGE, GAT and camouflage (both `detach_trust` settings), each at learning rates {0.0003, 0.001, 0.003, 0.01}. Select the best per family by validation PR-AUC only, and apply the screen against 0.6149.
6. For each family that passes the screen, run the frozen configuration with seeds 42, 123 and 456, then `evaluate.py` once per run.
7. Run `scripts/paired_bootstrap.py` for each GNN against `lgbm_tabular`, per seed and on the seed average. Run it against `lgbm_causal` as the secondary comparison.
8. Run the hand-written versus library layer comparison on validation PR-AUC.
9. Write the report from `experiments/*/test_results.json` and the bootstrap output only.