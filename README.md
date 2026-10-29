# Camouflage-Robust Graph Neural Networks for Fraud Detection

## Problem
Fraud rings link fraudulent accounts to ordinary-looking ones to avoid detection. This project builds a transaction-graph fraud detector and tests whether a camouflage-aware attention model improves detection under that pattern.

## Models
- LightGBM on node features only (no graph): reference baseline.
- GraphSAGE and GAT: PyTorch Geometric layers.
- Camouflage GAT (`FraudCamouflageGNN`, `src/models/models.py`): GAT whose attention is adjusted by a learned fraud-score agreement between neighbours.
- Hand-written SAGE and GAT layers (`src/models/layers/`): reference implementations, tested against the library layers.

## Graphs
- `graph_causal.pt` (main result): each transaction is linked only to earlier transactions that share a card, address, email domain, or device. A prediction uses only information available at that time.
- `graph_full.pt` (upper bound only): links in both time directions. Its scores use future transactions, so they are not deployable.

## Evaluation protocol
- Split: time-based, 70% train, 15% validation, 15% test (by `TransactionDT`).
- Checkpoint selection and early stopping: validation PR-AUC only.
- Headline metric: PR-AUC. Also reported: precision and recall at review budgets of 1%, 5%, and 10% (threshold chosen on validation). ROC-AUC is for comparison only.
- The test split is evaluated once per run. `evaluate.py` refuses to overwrite an existing `test_results.json` unless `--force` is given.

## Setup
1. Create a Python 3.11 or 3.12 virtual environment.
2. Install PyTorch for your CUDA version from https://pytorch.org, then run `python -m pip install -r requirements.txt`.
3. Put `train_transaction.csv` and `train_identity.csv` from the IEEE-CIS Fraud Detection competition into `data/raw/`.
4. Build both graphs: `python src/data/build_graph.py`

Exact package versions used in the working environment are in `requirements-lock.txt`.

## Running
```
python -m pytest tests -q
python src/train.py --config configs/sage_baseline.yaml --run_name sage_baseline
python src/evaluate.py --exp_dir experiments/sage_baseline
python src/train_tabular.py --config configs/lgbm_tabular.yaml
```

## Layout
- `src/data/build_graph.py`: features, label-free per-card history, relation edges, time masks
- `src/models/models.py`: models used for training
- `src/models/layers/`: hand-written reference layers
- `src/utils/`: losses, metrics, seeding
- `src/train.py`, `src/evaluate.py`, `src/train_tabular.py`: training, test evaluation, LightGBM baseline
- `configs/`: experiment settings
- `experiments/`: run outputs
- `demo/`: optional LLM explanation demo (not working yet; not part of the results)
- `Docs/decisions.md`: log of project decisions