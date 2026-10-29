# Project decisions

Date format: YYYY-MM-DD. Each decision is final until a newer entry overrides it.

## Pre-registered comparison (fixed before the final runs)
- 2026-10-29: Primary metric: test PR-AUC, averaged over seeds 42, 123 and 456.
- 2026-10-29: Hypothesis: the best GNN (selected on validation PR-AUC) has higher test PR-AUC than the best LightGBM, selected the same way.
- 2026-10-29: Primary comparator: the best LightGBM by validation PR-AUC. Of the two baselines, this is `lgbm_tabular` (node features, validation PR-AUC 0.6149) rather than `lgbm_causal` (0.5990). This applies the rule above to the two runs already made. It does not change the rule. Both baseline runs were complete when this was written, and their validation numbers were used to choose. Their test numbers were also visible when this was written, so the report must say so.
- 2026-10-29: Secondary hypothesis: the GNN also beats `lgbm_causal` (`feature_set: causal`). Reported in full, but it is not the basis of the decision.
- 2026-10-29: Decision rule: a GNN "wins" only if the paired bootstrap 95% interval for the PR-AUC difference (`scripts/paired_bootstrap.py`) against the primary comparator lies entirely above zero.
- 2026-10-29: Screen before the test comparison: a GNN family whose best validation PR-AUC is below the primary comparator's validation PR-AUC (0.6149) is reported as not competitive. It is not evaluated on test as a candidate winner, but its runs are still reported.
- 2026-10-29: Tuning budget is equal for each model family: the same learning-rate grid and the same number of configurations, selected on validation PR-AUC only.
- 2026-10-29: Tuning grid, selected on validation PR-AUC only: learning rate {0.0003, 0.001, 0.003, 0.01} for each of SAGE, GAT and camouflage; camouflage `detach_trust` {true, false}. `aux_loss_weight` is fixed at 0.3 and the loss weights `rank_loss_weight` and `ap_loss_weight` are fixed at 1.0. Neither is tuned. Changing this grid after any final run is not allowed.
- 2026-10-29: Breakdown analysis is reported whatever its result: cards seen in training versus unseen, and the first versus second half of the test period.
- 2026-10-29: If the GNN does not win, the report says so. A negative result is a valid result.
- 2026-10-29: This block must be committed before the final runs. Its commit hash is recorded in `Docs/status.md`.

## Modelling
- 2026-10-28: Layers. PyTorch Geometric layers are the default for all final models. Hand-written SAGE and GAT are kept as tested reference code. Switch to hand-written only if validation PR-AUC over 3 seeds is clearly higher than the library's, by more than the larger of the two standard deviations.
- 2026-10-28: Camouflage mechanism is kept. The canonical version is `FraudCamouflageGNN` in `src/models/models.py`. The old hand-written camouflage layer is removed.
- 2026-10-28: Node features add frequency of card, address, email domain and device, and hour of day. All are computed without labels.
- 2026-10-29: Camouflage trust is computed from detached scores by default (`detach_trust: true`). The alternative is compared on validation PR-AUC only.
- 2026-10-29: Entity-history features for LightGBM (`feature_set: causal`) use only earlier transactions for the same value. In the first run they did not raise validation PR-AUC (see `Docs/status.md`).

## Evaluation protocol
- 2026-10-28: Time-based split: first 70% train, next 15% validation, last 15% test, ordered by `TransactionDT`.
- 2026-10-28: Checkpoint selection and early stopping use validation PR-AUC only. ROC-AUC is reported but does not select checkpoints.
- 2026-10-28: The causal graph (each transaction linked only to earlier transactions) is the main result. The full graph is an upper bound and is never the headline.
- 2026-10-28: Review budgets: 1% headline, with 5% and 10% also reported. The budgets were chosen by the project members on 2026-10-28, before any budget-level result existed. The earlier SAGE run printed no budget metrics, so no budget-level test number has been seen. The budget is an assumption, not a measured review capacity: "we assume a review team can check about 1% of transactions." Intervals are reported alongside precision at each budget. The report must state this.
- 2026-10-28: The test split is evaluated once per run. `evaluate.py` refuses to overwrite `test_results.json` unless `--force` is given. Smoke-run numbers are never reported.
- 2026-10-28: Before the final runs, the fraud rate of each split is recorded in `Docs/status.md`. The validation-to-test gap seen in the earlier SAGE run must be explained first.
- 2026-10-28: Results files are generated only from `experiments/*/test_results.json`. Archived reports that could not be traced to runs are deleted.

## Environment and code delivery
- 2026-10-28: Training environment is the Kaggle image: Python 3.13.15, torch 2.11.0+cu128, torch-geometric 2.8.0.post1, lightgbm 4.6.0, pandas 2.3.3, numpy 2.1.3, scikit-learn 1.6.1, Tesla T4 GPU. `requirements.txt` pins these versions. Local tests use the same versions.
- 2026-10-28: Code reaches Kaggle as a zip made from a local commit with `git archive`. Each run therefore matches one commit. The commit hash is recorded in `Docs/status.md` with each run.
- 2026-10-28: Commits are made locally once tests pass. They are pushed to GitHub only after the team agrees.
- 2026-10-28: Kaggle notebooks must not run `pip install` for torch. The image already provides it. `torch-geometric` and `pyg-lib` are installed with the pinned versions in the setup cell, as the image does not include them.
- 2026-10-29: Run outputs stay in the Kaggle session until copied. Each `test_results.json` used in the report must be copied into `experiments/<run_name>/` and committed.

## Demo and presentation
- 2026-10-28: `agent.py` is an optional demo in `demo/`, not part of the results.

## Repository hygiene
- 2026-10-28: Commits and project files must not contain user names or email addresses. Commit identity is a generic placeholder agreed by the team.
- 2026-10-28: Old checkpoints in `experiments/` are untracked once the team agrees. Their `.json` files are kept.
- 2026-10-28: Existing commit identities on `feat/rank-objective` are left unchanged. New commits use the placeholder identity.
- 2026-10-28: Local development uses Python 3.14.3. Training and reported results come from Kaggle (Python 3.13.15).