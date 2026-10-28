# Project status

Last updated: 2026-10-28. Decisions are in `Docs/decisions.md`.

## Summary
The project is a graph-based fraud detector on IEEE-CIS data, with LightGBM as a non-graph baseline and a camouflage-aware GAT as the proposed model. No valid final results exist yet. Earlier numbers came from a pipeline that has since changed.

## Phase status
- **Phase 0 (setup, repo):** done. The repo, `.gitignore` (data rule anchored to `/data/`), and environment files are in place.
- **Phase 1 (dataset, EDA):** done. IEEE-CIS chosen. `notebooks/eda.ipynb` covers class imbalance and entity sharing.
- **Phase 2 (graph construction):** code rewritten, not yet run. `src/data/build_graph.py` builds two graphs from one feature matrix:
  - `graph_causal.pt` (main result): each transaction links only to earlier transactions.
  - `graph_full.pt` (upper bound): links across all time.
  - New features: frequency of card, address, email domain and device; hour of day.
- **Phase 3 (layers):** models use PyTorch Geometric layers. Hand-written SAGE and GAT remain as reference code. Their equivalence test with the library layers is not yet written.
- **Phase 4 (baselines):** the SAGE run below is the only baseline result available. GAT has not been rerun on the current pipeline.
- **Phase 5 (training harness):** rewritten. Checkpoints are selected on validation PR-AUC only. Validation sampling is repeatable. Test evaluation runs once per run and reports precision and recall at review budgets of 1%, 5% and 10%.
- **Phase 6 (literature):** `report/related_work.md` is written.
- **Phase 7 (camouflage model):** `FraudCamouflageGNN` in `src/models/models.py` is the canonical version. The earlier hand-written version has been removed. Its only full-graph run before removal collapsed to ROC-AUC about 0.5. The new version has not been trained.
- **Phase 8 (experiments, report):** not started on the current pipeline. `report/results.md` says so.

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
- **The validation-to-test gap is large.** Validation PR-AUC is 0.42 and test PR-AUC is 0.23, while ROC-AUC barely moves (0.85 to 0.84). This usually means the fraud rate or fraud pattern differs between the two periods. It must be explained before any result is reported. The fraud rate per split is the first thing to check.

## Known open issues
1. **Validation-to-test gap** (above). Check the fraud rate per split, then decide whether the time split is the right protocol.
2. **Pipeline version mismatch.** The Kaggle environment (Python 3.13, torch 2.11) differs from the local environment (Python 3.14, torch 2.14). Pin to Kaggle's versions.
3. **Failing test.** `tests/test_build_graph.py` still refers to the removed `OUTPUT_FILE` setting and to the old single `graph.pt` file.
4. **Lost training notebook.** `gnn_kaggle_train_v2.ipynb` and `gnn_fraud_full_run.ipynb` are not available. The replacement is a short runner notebook that reads the code from a Kaggle dataset.
5. **Git history.** Four commits on `feat/rank-objective` carry a real name and email. They are unmerged, so they can be rewritten, but only with the friend's agreement.
6. **Reporting.** The archived reports that could not be traced to runs have been deleted.

## Next steps
1. Fix the failing test.
2. Pin the environment to Kaggle's versions. Recreate the local venv with Python 3.13.
3. Compute the fraud rate per split and record it in the status.
4. Build both graphs on Kaggle using the competition data.
5. Run a 2-epoch smoke test of each model and of LightGBM.
6. Commit the decisions in `Docs/decisions.md`, including the review budgets, before any test-set results are seen.
7. Run the final experiments: 3 seeds per model and graph.
8. Run the hand-written versus library layer comparison on validation PR-AUC.
9. Run the camouflage stress test.
10. Write the report from `experiments/*/test_results.json` only.