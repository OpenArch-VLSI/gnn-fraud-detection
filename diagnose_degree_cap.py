import pandas as pd

RELATION_COLS = ['card1', 'card2', 'addr1', 'addr2', 'P_emaildomain', 'DeviceInfo']
DEGREE_CAP = 100

df_trans = pd.read_csv("data/raw/train_transaction.csv")
df_id = pd.read_csv("data/raw/train_identity.csv")
df = df_trans.merge(df_id, on='TransactionID', how='left')

for col in RELATION_COLS:
    if col not in df.columns:
        continue

    valid = df[df[col].notna()]
    group_sizes = valid.groupby(col).size()

    dropped_groups = group_sizes[group_sizes > DEGREE_CAP]
    kept_groups = group_sizes[(group_sizes > 1) & (group_sizes <= DEGREE_CAP)]

    transactions_in_dropped = dropped_groups.sum()
    transactions_in_kept = kept_groups.sum()

    print(f"\n{col}:")
    print(f"  Total groups: {len(group_sizes)}")
    print(f"  Groups dropped (size > {DEGREE_CAP}): {len(dropped_groups)}")
    print(f"  Transactions in dropped groups: {transactions_in_dropped}")
    print(f"  Transactions in kept groups (size 2-{DEGREE_CAP}): {transactions_in_kept}")