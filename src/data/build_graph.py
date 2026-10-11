import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import argparse

# Configuration
PROCESSED_DATA_DIR = 'data/processed'
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'graph.pt')

# Relations to build and degree cap (to prevent hub explosion)
# NOTE: 'addr2' was removed -- it only has 74 distinct values across 590,540
# transactions (avg. group size ~175,000), meaning it groups transactions by
# broad region/country rather than by any fraud-relevant shared entity. It
# contributed ~52M low-information edges while adding little discriminative
# signal, so it's excluded rather than capped.
RELATION_COLS = ['card1', 'card2', 'addr1', 'P_emaildomain', 'DeviceInfo']

# NOTE: DEGREE_CAP lowered from 100 to 20. At 100, capped/sampled groups for
# high-cardinality-but-still-large relations (e.g. addr1, P_emaildomain)
# pushed total edges to ~273M (~9GB graph.pt), which is impractical to
# download/load/train on a single T4 GPU. 20 preserves meaningfully more
# connectivity signal than a more aggressive cap (e.g. 10) while cutting
# total edges from capped groups by ~5x compared to 100.
DEGREE_CAP = 20  # Max number of edges any single node can get from one relation group
RANDOM_SEED = 42  # For reproducible sampling when capping large groups

# Names of the per-card history columns produced by card_history_features.
CARD_HISTORY_COLS = ['card_prior_count', 'card_prior_amt_mean', 'card_prior_amt_std',
                     'card_amt_dev', 'card_log_gap']


def card_history_features(df):
    """
    Label-free, look-back-only per-card features, keyed on card1.

    df must already be sorted by TransactionDT. For each row, the statistics
    use only the EARLIER rows that share its card1 value. The current row is
    never included and no label is read, so these features cannot leak the
    target or any later transaction.

    Rows without card1, and rows where a statistic has no history yet, are NaN.
    build_graph fills those NaNs with the train-split median like any other
    numeric column.

    Returns a DataFrame indexed like df with columns CARD_HISTORY_COLS:
      card_prior_count     log1p(number of earlier transactions on this card)
      card_prior_amt_mean  mean of log1p(amount) over earlier transactions
      card_prior_amt_std   std (ddof=0) of log1p(amount) over earlier transactions
      card_amt_dev         log1p(amount) minus card_prior_amt_mean
      card_log_gap         log1p(seconds since the previous transaction on this card)
    """
    out = pd.DataFrame(np.nan, index=df.index, columns=CARD_HISTORY_COLS)
    valid = df['card1'].notna()
    if not valid.any():
        return out

    card = df.loc[valid, 'card1']
    t = df.loc[valid, 'TransactionDT'].astype(float)
    amt = np.log1p(df.loc[valid, 'TransactionAmt'].astype(float).clip(lower=0))

    # Running totals within each card, minus the current row, so only earlier
    # rows contribute. Missing amounts count as "no observation", not as zero.
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
    Replace each category with its share of the train split (the first
    train_end rows, in time order). Categories not seen in train map to 0.0.

    Unlike ordinal codes, this does not impose an arbitrary numeric order on
    unordered categories. The trade-off is that two categories with the same
    share become indistinguishable. Frequencies are computed from features
    only, never from labels.
    """
    freq = series.iloc[:train_end].value_counts(normalize=True)
    return series.map(freq).fillna(0.0).astype(float)


def check_files(transaction_file, identity_file, raw_data_dir):
    if not os.path.exists(transaction_file) or not os.path.exists(identity_file):
        print("Error: IEEE-CIS dataset files not found.")
        print(f"Please download 'train_transaction.csv' and 'train_identity.csv' from Kaggle (IEEE-CIS Fraud Detection)")
        print(f"and place them in the '{raw_data_dir}' directory.")
        return False
    return True

def build_graph(transaction_file, identity_file, limit=None):
    print("Loading data...")
    # Read data
    df_trans = pd.read_csv(transaction_file)
    df_id = pd.read_csv(identity_file)
    
    # Sort transactions by time to allow time-based split
    df_trans = df_trans.sort_values('TransactionDT').reset_index(drop=True)
    
    if limit is not None:
        df_trans = df_trans.head(limit)
    
    # Merge on TransactionID
    df = df_trans.merge(df_id, on='TransactionID', how='left')
    
    print(f"Total transactions: {len(df)}")
    
    labels = df['isFraud'].values
    y = torch.tensor(labels, dtype=torch.long)
    
    print("Processing node features...")
    # Feature processing: log-scale amounts, add label-free per-card history,
    # impute numeric NaNs with train medians, frequency-encode categoricals
    # (train shares), then scale (train statistics). Nothing here reads labels.
    # Exclude IDs, target, and relation columns from node features
    exclude_cols = ['TransactionID', 'isFraud', 'TransactionDT', 'addr2'] + RELATION_COLS
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    features_df = df[feature_cols].copy()

    # Amounts are heavy-tailed, so the model sees log(1 + amount), not raw dollars.
    features_df['TransactionAmt'] = np.log1p(features_df['TransactionAmt'].clip(lower=0))

    # Per-card history: how this transaction compares with the same card's
    # EARLIER activity (amount deviation, spending level, time since last use).
    features_df = pd.concat([features_df, card_history_features(df)], axis=1)

    # Identify numeric and categorical columns. Numeric is any number dtype;
    # everything else is categorical. This keeps string columns categorical
    # under pandas >= 3, which stores them in a dedicated string dtype.
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    categorical_cols = features_df.select_dtypes(exclude=[np.number]).columns
    
    train_end = int(len(df) * 0.7)

    # BUGFIX: numeric NaNs were previously filled with a raw 0 *before*
    # StandardScaler was fit/applied. For columns with a non-zero natural
    # mean and substantial missingness (common among the anonymized V1-V339
    # features in this dataset), a raw 0 fill becomes an artificial outlier
    # once scaled -- e.g. a column with train-mean 50 and std 10 turns every
    # NaN into a scaled value of -5.0, a value the model has no way to tell
    # apart from a genuine extreme case. This is especially harmful for
    # attention-based models (GAT/camouflage-GAT), whose attention logits
    # are more sensitive to input-feature outliers than SAGE's mean
    # aggregation.
    #
    # Fix: impute each numeric column's NaNs with that column's own
    # TRAIN-SPLIT median (a robust, distribution-appropriate fill value)
    # computed only from the train portion, consistent with how the scaler
    # and categorical encoder below are already fit only on train_end to
    # avoid leaking val/test statistics into imputation.
    numeric_medians = features_df.iloc[:train_end][numeric_cols].median()
    features_df[numeric_cols] = features_df[numeric_cols].fillna(numeric_medians)
    
    # Encode categoricals as train-split frequencies, not ordinal codes.
    # Ordinal codes give unordered categories an arbitrary numeric order,
    # which the scaler then treats as magnitude.
    for col in categorical_cols:
        features_df[col] = frequency_encode(features_df[col].fillna('MISSING'), train_end)
    
    # Scale features
    scaler = StandardScaler()
    scaler.fit(features_df.iloc[:train_end])
    x_scaled = scaler.transform(features_df)
    x = torch.tensor(x_scaled, dtype=torch.float)
    
    print("Building edges...")
    edge_indices = {}
    rng = np.random.default_rng(RANDOM_SEED)
    
    for col in RELATION_COLS:
        if col not in df.columns:
            continue
            
        print(f"  Processing relation: {col}")
        # Drop missing values for the relation
        valid_nodes = df[df[col].notna()][['TransactionID', col]]
        valid_nodes['node_idx'] = valid_nodes.index
        
        # Group by the entity value
        grouped = valid_nodes.groupby(col)['node_idx'].apply(list)
        
        # Need at least 2 nodes to make an edge
        grouped = grouped[grouped.apply(len) > 1]
        
        # Build edges. Small groups (<= DEGREE_CAP) are fully connected.
        # Large groups (> DEGREE_CAP) are NOT dropped anymore -- instead,
        # each node in the group is connected to a random sample of up to
        # DEGREE_CAP other members, so popular entities (e.g. a widely-used
        # card or common email domain) still contribute edges without
        # creating an all-pairs "hub explosion" (which for a group of size
        # N would otherwise create N*(N-1) edges).
        src = []
        dst = []
        num_capped_groups = 0
        num_full_groups = 0
        
        for indices in grouped:
            group_size = len(indices)
            
            if group_size <= DEGREE_CAP:
                # Small enough: fully connect every pair, as before
                num_full_groups += 1
                for i in indices:
                    for j in indices:
                        if i != j:
                            src.append(i)
                            dst.append(j)
            else:
                # Large group: cap each node's edges via random sampling
                # instead of discarding the group entirely
                num_capped_groups += 1
                indices_arr = np.array(indices)
                for i in indices_arr:
                    # Sample up to DEGREE_CAP other members (excluding i itself)
                    others = indices_arr[indices_arr != i]
                    sample_size = min(DEGREE_CAP, len(others))
                    sampled = rng.choice(others, size=sample_size, replace=False)
                    for j in sampled:
                        src.append(i)
                        dst.append(int(j))
        
        if len(src) > 0:
            edge_indices[col] = torch.tensor([src, dst], dtype=torch.long)
            print(f"    Created {len(src)} edges for {col} "
                  f"({num_full_groups} fully-connected groups, "
                  f"{num_capped_groups} capped/sampled groups)")
        else:
            print(f"    No edges created for {col} (no groups of size > 1)")
    
    # Create single combined edge_index for basic GraphSAGE/GAT
    # (Advanced model in Phase 7 might use the separate relations)
    all_src = []
    all_dst = []
    for ei in edge_indices.values():
        all_src.extend(ei[0].tolist())
        all_dst.extend(ei[1].tolist())
        
    if len(all_src) > 0:
        combined_edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    else:
        combined_edge_index = torch.empty((2, 0), dtype=torch.long)
        
    print(f"Total combined edges: {combined_edge_index.shape[1]}")
    
    # Create train/val/test masks based on time (TransactionDT sorting)
    # E.g., 70% train, 15% val, 15% test
    n_nodes = len(df)
    train_end = int(n_nodes * 0.7)
    val_end = int(n_nodes * 0.85)
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    
    data = Data(x=x, edge_index=combined_edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    # Store individual relation edge indices as extra attributes
    for col, ei in edge_indices.items():
        setattr(data, f'edge_index_{col}', ei)
        
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    torch.save(data, OUTPUT_FILE)
    print(f"Graph saved to {OUTPUT_FILE}")
    print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help="Limit number of rows to load")
    parser.add_argument('--data_dir', type=str, default='data/raw', help="Directory containing Kaggle CSVs")
    args = parser.parse_args()

    # Change working directory to project root if executed from elsewhere
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(proj_root)
    
    transaction_file = os.path.join(args.data_dir, 'train_transaction.csv')
    identity_file = os.path.join(args.data_dir, 'train_identity.csv')
    
    if check_files(transaction_file, identity_file, args.data_dir):
        build_graph(transaction_file, identity_file, limit=args.limit)