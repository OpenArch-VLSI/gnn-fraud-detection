# Project decisions

Date format: YYYY-MM-DD. Each decision is final until a newer entry overrides it.

## Modelling
- 2026-10-28: Layers. PyTorch Geometric layers are the default for all final models. Hand-written SAGE and GAT are kept as tested reference code. Switch to hand-written only if validation PR-AUC over 3 seeds is clearly higher than the library's, by more than the larger of the two standard deviations.
- 2026-10-28: Camouflage mechanism is kept. The canonical version is `FraudCamouflageGNN` in `src/models/models.py`. The old hand-written camouflage layer is removed.
- 2026-10-28: Node features add frequency of card, address, email domain and device, and hour of day. All are computed without labels.

## Evaluation protocol
- 2026-10-28: Time-based split: first 70% train, next 15% validation, last 15% test, ordered by `TransactionDT`.
- 2026-10-28: Checkpoint selection and early stopping use validation PR-AUC only. ROC-AUC is reported but does not select checkpoints.
- 2026-10-28: The causal graph (each transaction linked only to earlier transactions) is the main result. The full graph is an upper bound and is never the headline.
- 2026-10-28: Review budgets. Headline: 1% of transactions reviewed. Also reported: 5% and 10%. Confirmed by the team on 2026-10-28, before any test-set result was seen. The team is the two project members, and the budget is an assumption: "we assume a review team can check about 1% of transactions." The report must state this.
- 2026-10-28: The test split is evaluated once per run. `evaluate.py` refuses to overwrite `test_results.json` unless `--force` is given. Smoke-run numbers are never reported.
- 2026-10-28: Before the final runs, the fraud rate of each split is recorded in `docs/status.md`. The validation-to-test gap seen in the earlier SAGE run must be explained first.
- 2026-10-28: Results files are generated only from `experiments/*/test_results.json`. Archived reports that could not be traced to runs are deleted.

## Environment and code delivery
- 2026-10-28: Training environment is the Kaggle image: Python 3.13.15, torch 2.11.0+cu128, torch-geometric 2.8.0.post1, lightgbm 4.6.0, pandas 2.3.3, numpy 2.1.3, scikit-learn 1.6.1, Tesla T4 GPU. `requirements.txt` pins these versions. Local tests use the same versions.
- 2026-10-28: Code reaches Kaggle as a zip made from a local commit with `git archive`. Each run therefore matches one commit. The commit hash is recorded in `docs/status.md` with each run.
- 2026-10-28: Commits are made locally once tests pass. They are pushed to GitHub only after the team agrees.
- 2026-10-28: Kaggle notebooks must not run `pip install` cells for torch or PyG. The image already provides them, and extra installs can mix versions.

## Demo and presentation
- 2026-10-28: `agent.py` is an optional demo in `demo/`, not part of the results.

## Repository hygiene
- 2026-10-28: Commits and project files must not contain user names or email addresses. Commit identity is a generic placeholder agreed by the team.
- 2026-10-28: Old checkpoints in `experiments/` are untracked once the team agrees. Their `.json` files are kept.
- 2026-10-28: Existing commit identities on `feat/rank-objective` are left unchanged. New commits use the placeholder identity.
- 2026-10-28: Local development uses Python 3.14.3. Training and reported results come from Kaggle (Python 3.13.15).