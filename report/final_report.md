# Final Report (draft, results pending)

Status: no final results yet. This file is completed after the final runs.

## Method
- Graph: transactions are linked when they share a card, address, email domain, or device. The main graph (causal) links each transaction only to earlier transactions. A second graph (full) links across all time and is reported only as an upper bound.
- Features: transaction amount (log scale), hour of day, frequency of each card, address, email domain, and device in the training data, and per-card history using only earlier transactions.
- Models: LightGBM (no graph) as the reference; GraphSAGE and GAT as baselines; camouflage GAT as the proposed model. Layers are PyTorch Geometric's. Hand-written reference layers are included and tested against them.
- Camouflage mechanism: a score head predicts a fraud score for each node from its own features. Edges whose endpoints get similar scores receive more attention. The learned weight on this term is checked after training.

## Protocol
- Time-based split, 70/15/15.
- Model selection and early stopping on validation PR-AUC only.
- Test split evaluated once per run.
- Headline metric: PR-AUC. Also reported: precision and recall at review budgets of 1%, 5%, and 10%. ROC-AUC for comparison only.

## Known limitations
- Relation types are combined into one edge set for the model.
- The non-causal graph uses future transactions and is an upper bound only.
- Results from earlier experiments were produced on an older feature set and graph, and are not reported.
- The LLM explanation demo is not part of the results.

## Results
To be added from `experiments/*/test_results.json` after the final runs. Negative or small gains will be reported as measured.