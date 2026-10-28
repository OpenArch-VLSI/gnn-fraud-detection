"""
Builds the IEEE-CIS transaction graph.

Two graphs are written from the same node features:
  graph_causal.pt  edges only from EARLIER to LATER transactions. Main result:
                   a transaction is scored using only transactions that
                   happened before it.
  graph_full.pt    edges across all time. Its scores use future transactions,
                   so they are an upper bound, not a deployable result.

Node features use only information known at a transaction's own time, plus
statistics computed on the training split. No labels are read.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

PROCESSED_DATA_DIR = os.path.join('data', 'processed')

# Columns that define graph links. Their raw values are not node features,
# but how common each value is (frequency) is.
RELATION_COLS = ['card1', 'card2', 'addr1', 'P_emaildomain', 'DeviceInfo']
EXCLUDE_COLS = ['TransactionID', 'isFraud', 'TransactionDT', 'addr2'] + RELATION_COLS

# Maximum neighbours per node from one relation. Larger groups are sampled.
DEGREE_CAP = 20
RANDOM_SEED = 42

CARD_HISTORY_COLS = ['card_prior_count', 'card_prior_amt_mean', 'card_prior_amt_std',
                     'card_amt_dev', 'card_log_gap']


def card_history_features(df):
    """
    Label-free, look-back-only features per card1. df must be sorted by time.
    Each row uses only EARLIER rows with the same card1; the current row is excluded.
    """
    out = pd.DataFrame(np.nan, index=df.index, columns=CARD_HISTORY_COLS)
    valid = df['card1'].notna()
    if not valid.any():
        return out

    card = df.loc[valid, 'card1']
    t = df.loc[valid, 'TransactionDT'].astype(float)
    amt = np.log1p(df.loc[valid, 'TransactionAmt'].astype(float).clip(lower=0))

    has_amt = amt.notna().astype(float)
    amt0 = amt.fillna(0.0)
    n_prior = has_amt.groupby(card).cumsum() - has_amt
    s_prior = amt0.groupby(card).cumsum() - amt0
    sq_prior = (amt0 ** 2).groupby(card).cumsum() - amt0 ** 2

    mean = (s_prior / n_prior).where(n_prior > 0)
    var = (sq_prior / n_prior - mean ** 2).clip(lower=0)
    gap = t.groupby(card).diff().clip(lower=0)

    out.loc[valid, 'card_prior_count'] = np.log1p(card.groupby(card).cumcount())
    out.loc[valid, 'card_prior_amt_mean'] = mean
    out.loc[valid, 'card_prior_amt_std'] = np.sqrt(var)
    out.loc[valid, 'card_amt_dev'] = amt - mean
    out.loc[valid, 'card_log_gap'] = np.log1p(gap)
    return out


def frequency_encode(series, train_end):
    """
    Replace each value with its share among the first train_end rows (training
    split). Values not seen in training map to 0.0. Uses features only, never labels.
    """
    freq = series.iloc[:train_end].value_counts(normalize=True)
    return series.map(freq).fillna(0.0).astype(float)


def check_files(transaction_file, identity_file, raw_data_dir):
    if not os.path.exists(transaction_file) or not os.path.exists(identity_file):
        print("Error: IEEE-CIS dataset files not found.")
        print("Download 'train_transaction.csv' and 'train_identity.csv' from the Kaggle "
              "IEEE-CIS Fraud Detection competition and place them in "
              f"'{raw_data_dir}'.")
        return False
    return True


def load_frame(transaction_file, identity_file, limit=None):
    """Transactions sorted by time, merged with identity columns."""
    df_trans = pd.read_csv(transaction_file).sort_values('TransactionDT').reset_index(drop=True)
    if limit is not None:
        df_trans = df_trans.head(limit)
    df_id = pd.read_csv(identity_file)
    return df_trans.merge(df_id, on='TransactionID', how='left').reset_index(drop=True)


def node_features(df, train_end):
    feats = df[[c for c in df.columns if c not in EXCLUDE_COLS]].copy()

    # Heavy-tailed amount: use log(1 + amount).
    feats['TransactionAmt'] = np.log1p(feats['TransactionAmt'].clip(lower=0))
    # Hour of day from the transaction time (label-free).
    feats['tx_hour'] = (df['TransactionDT'] // 3600) % 24
    # How common this card, address, email domain, or device is in the training data.
    for col in RELATION_COLS:
        if col in df.columns:
            feats[f'{col}_freq'] = frequency_encode(df[col].astype(str), train_end)

    feats = pd.concat([feats, card_history_features(df)], axis=1)

    numeric = feats.select_dtypes(include=[np.number]).columns
    categorical = feats.select_dtypes(exclude=[np.number]).columns

    # Impute numeric gaps with training-split medians (0 if a column is all missing).
    medians = feats.iloc[:train_end][numeric].median().fillna(0.0)
    feats[numeric] = feats[numeric].fillna(medians)

    for col in categorical:
        feats[col] = frequency_encode(feats[col].fillna('MISSING'), train_end)

    scaler = StandardScaler().fit(feats.iloc[:train_end])
    return torch.tensor(scaler.transform(feats), dtype=torch.float)


def relation_edges(df, col, causal, rng):
    """
    Edges for one relation column.
    causal=True : each transaction is linked FROM up to DEGREE_CAP earlier
                  transactions with the same value. Direction is earlier -> later.
    causal=False: all pairs in small groups; random samples in large groups, both ways.
    """
    sub = df.loc[df[col].notna(), [col]]
    rows = sub.index.to_numpy()          # df row ids; increasing = time order
    src, dst = [], []

    for pos in sub.groupby(col).indices.values():
        members = rows[pos]
        if len(members) < 2:
            continue
        if causal:
            for k in range(1, len(members)):
                earlier = members[:k]
                if len(earlier) > DEGREE_CAP:
                    earlier = rng.choice(earlier, size=DEGREE_CAP, replace=False)
                src.append(earlier)
                dst.append(np.full(len(earlier), members[k]))
        elif len(members) <= DEGREE_CAP:
            i, j = np.meshgrid(members, members)
            keep = i != j
            src.append(i[keep])
            dst.append(j[keep])
        else:
            for i in members:
                others = members[members != i]
                sampled = rng.choice(others, size=min(DEGREE_CAP, len(others)), replace=False)
                src.append(np.full(len(sampled), i))
                dst.append(sampled)

    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    edges = np.vstack([np.concatenate(src), np.concatenate(dst)])
    return torch.tensor(edges, dtype=torch.long)


def build(transaction_file, identity_file, limit=None, modes=('causal', 'full')):
    print("Loading data...")
    df = load_frame(transaction_file, identity_file, limit)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    print(f"Total transactions: {n}")

    y = torch.tensor(df['isFraud'].values, dtype=torch.long)
    print("Building node features...")
    x = node_features(df, train_end)
    print(f"Node feature count: {x.shape[1]}")

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    for mode in modes:
        print(f"Building {mode} graph...")
        rng = np.random.default_rng(RANDOM_SEED)
        relation_ei = {}
        for col in RELATION_COLS:
            if col in df.columns:
                relation_ei[col] = relation_edges(df, col, causal=(mode == 'causal'), rng=rng)
                print(f"  {col}: {relation_ei[col].size(1)} edges")

        if relation_ei:
            combined = torch.cat(list(relation_ei.values()), dim=1)
        else:
            combined = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=combined, y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        data.graph_mode = mode
        for col, ei in relation_ei.items():
            setattr(data, f'edge_index_{col}', ei)

        path = os.path.join(PROCESSED_DATA_DIR, f'graph_{mode}.pt')
        torch.save(data, path)
        print(f"Saved {path} with {combined.size(1)} edges")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help="Use only the first N rows (testing)")
    parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'raw'))
    parser.add_argument('--modes', nargs='+', default=['causal', 'full'], choices=['causal', 'full'])
    args = parser.parse_args()

    # Run from the project root regardless of where the script was started.
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(proj_root)

    transaction_file = os.path.join(args.data_dir, 'train_transaction.csv')
    identity_file = os.path.join(args.data_dir, 'train_identity.csv')
    if check_files(transaction_file, identity_file, args.data_dir):
        build(transaction_file, identity_file, limit=args.limit, modes=args.modes)