import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.data.build_graph as bg


def _card_frame(card, amounts, times):
    return pd.DataFrame({
        'card1': np.asarray(card, dtype=float),
        'TransactionAmt': np.asarray(amounts, dtype=float),
        'TransactionDT': np.asarray(times, dtype=float),
    })


def test_card_history_matches_hand_computed_values():
    # One card: amounts 10, 100, 1000 at t = 0, 60, 180 seconds.
    df = _card_frame([1, 1, 1], [10.0, 100.0, 1000.0], [0.0, 60.0, 180.0])
    out = bg.card_history_features(df)

    a, b, c = np.log1p([10.0, 100.0, 1000.0])

    # First transaction on the card: no history yet.
    assert out.loc[0, 'card_prior_count'] == 0.0
    assert out.loc[0, ['card_prior_amt_mean', 'card_prior_amt_std',
                       'card_amt_dev', 'card_log_gap']].isna().all()

    # Second transaction: history is the first amount only.
    assert out.loc[1, 'card_prior_count'] == pytest.approx(np.log1p(1))
    assert out.loc[1, 'card_prior_amt_mean'] == pytest.approx(a)
    assert out.loc[1, 'card_prior_amt_std'] == pytest.approx(0.0)
    assert out.loc[1, 'card_amt_dev'] == pytest.approx(b - a)
    assert out.loc[1, 'card_log_gap'] == pytest.approx(np.log1p(60.0))

    # Third transaction: history is the first two amounts.
    m = (a + b) / 2
    assert out.loc[2, 'card_prior_count'] == pytest.approx(np.log1p(2))
    assert out.loc[2, 'card_prior_amt_mean'] == pytest.approx(m)
    assert out.loc[2, 'card_prior_amt_std'] == pytest.approx(abs(b - a) / 2)
    assert out.loc[2, 'card_amt_dev'] == pytest.approx(c - m)
    assert out.loc[2, 'card_log_gap'] == pytest.approx(np.log1p(120.0))


def test_card_history_ignores_later_transactions():
    """Changing any row must not change features of earlier rows (no look-ahead)."""
    rng = np.random.default_rng(0)
    n = 60
    df = _card_frame(rng.integers(0, 4, size=n),
                     rng.lognormal(3, 1, size=n),
                     np.arange(n) * 30.0)
    base = bg.card_history_features(df)

    changed = df.copy()
    changed.loc[41:, 'TransactionAmt'] = 1e6   # alter every row after index 40
    after = bg.card_history_features(changed)

    np.testing.assert_allclose(base.loc[:40].to_numpy(),
                               after.loc[:40].to_numpy(), equal_nan=True)
    # Sanity check that the test can fail: later rows must actually change.
    assert not np.array_equal(base.loc[41:].fillna(-1).to_numpy(),
                              after.loc[41:].fillna(-1).to_numpy())


def test_missing_card_gets_no_history_and_does_not_join_other_cards():
    df = _card_frame([1, np.nan, 1], [10.0, 500.0, 20.0], [0.0, 10.0, 20.0])
    out = bg.card_history_features(df)

    assert out.loc[1].isna().all()
    # Row 2 sees card 1's history (row 0) only, not the card-less row 1.
    assert out.loc[2, 'card_prior_count'] == pytest.approx(np.log1p(1))
    assert out.loc[2, 'card_log_gap'] == pytest.approx(np.log1p(20.0))


def test_frequency_encoding_uses_train_split_only():
    s = pd.Series(['a', 'a', 'b', 'a', 'c'])
    out = bg.frequency_encode(s, train_end=4)
    # Shares come from the first 4 rows; 'c' appears only after the train split.
    np.testing.assert_allclose(out.to_numpy(), [0.75, 0.75, 0.25, 0.75, 0.0])


def _relation_frame(values):
    return pd.DataFrame({'card1': np.asarray(values, dtype=float)})


def _edge_set(ei):
    return set(zip(ei[0].tolist(), ei[1].tolist()))


def test_causal_edges_point_from_earlier_to_later(monkeypatch):
    monkeypatch.setattr(bg, 'DEGREE_CAP', 20)
    # Rows 0, 2 and 5 share a value. Causal links: 0->2, 0->5, 2->5.
    df = _relation_frame([1, np.nan, 1, np.nan, np.nan, 1])
    ei = bg.relation_edges(df, 'card1', causal=True, rng=np.random.default_rng(0))
    assert _edge_set(ei) == {(0, 2), (0, 5), (2, 5)}


def test_full_edges_link_both_directions_in_small_groups(monkeypatch):
    monkeypatch.setattr(bg, 'DEGREE_CAP', 20)
    df = _relation_frame([1, 1, 1])
    ei = bg.relation_edges(df, 'card1', causal=False, rng=np.random.default_rng(0))
    assert _edge_set(ei) == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}


def test_causal_in_degree_is_capped(monkeypatch):
    monkeypatch.setattr(bg, 'DEGREE_CAP', 3)
    df = _relation_frame(np.ones(30))
    ei = bg.relation_edges(df, 'card1', causal=True, rng=np.random.default_rng(0))
    src, dst = ei[0].numpy(), ei[1].numpy()

    assert (src < dst).all()                 # earlier -> later only
    in_deg = np.bincount(dst, minlength=30)
    assert in_deg.max() <= 3                 # cap respected
    assert in_deg[0] == 0                    # first transaction has no earlier partner
    assert in_deg[1:].min() >= 1             # every later transaction has at least one


def _write_synthetic_ieee(tmp_path, n=40, seed=0):
    rng = np.random.default_rng(seed)
    tx = pd.DataFrame({
        'TransactionID': np.arange(n),
        'isFraud': (rng.random(n) < 0.2).astype(int),
        'TransactionDT': np.sort(rng.integers(0, 10_000, size=n)),
        'TransactionAmt': rng.lognormal(3, 1, size=n).round(2),
        'ProductCD': rng.choice(['W', 'C', 'H'], size=n),
        'V1': np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
        'card1': rng.integers(0, 5, size=n),
        'card2': rng.integers(100, 105, size=n).astype(float),
        'addr1': rng.integers(0, 3, size=n).astype(float),
        'addr2': 87.0,
        'P_emaildomain': rng.choice(['gmail.com', 'yahoo.com'], size=n),
    })
    ident = pd.DataFrame({
        'TransactionID': np.arange(n),
        'DeviceInfo': rng.choice(['Windows', 'iOS Device'], size=n),
        'DeviceType': [None if u < 0.5 else 'desktop' for u in rng.random(n)],
    })
    tx_path = tmp_path / 'train_transaction.csv'
    id_path = tmp_path / 'train_identity.csv'
    tx.to_csv(tx_path, index=False)
    ident.to_csv(id_path, index=False)
    return tx_path, id_path


def test_build_writes_causal_and_full_graphs_on_synthetic_csvs(tmp_path, monkeypatch):
    tx_path, id_path = _write_synthetic_ieee(tmp_path)
    out_dir = tmp_path / 'processed'
    monkeypatch.setattr(bg, 'PROCESSED_DATA_DIR', str(out_dir))

    bg.build(str(tx_path), str(id_path))

    causal = torch.load(out_dir / 'graph_causal.pt', weights_only=False)
    full = torch.load(out_dir / 'graph_full.pt', weights_only=False)

    assert causal.graph_mode == 'causal'
    assert full.graph_mode == 'full'
    assert causal.num_nodes == 40 and full.num_nodes == 40

    # Node features: TransactionAmt, ProductCD, V1, DeviceType (4),
    # hour of day (1), five relation frequencies (5), five card-history columns (5) = 15.
    assert causal.x.shape == (40, 15)
    assert torch.isfinite(causal.x).all()
    # Both graphs share the same node features.
    assert torch.equal(causal.x, full.x)

    # Causal graph: every edge goes from an earlier row to a later row.
    assert causal.edge_index.size(1) > 0
    assert (causal.edge_index[0] < causal.edge_index[1]).all()
    # The full graph has at least as many edges.
    assert full.edge_index.size(1) >= causal.edge_index.size(1)

    # Time-ordered masks: 70% / 15% / 15%, disjoint, covering every row.
    assert (causal.train_mask | causal.val_mask | causal.test_mask).all()
    assert not (causal.train_mask & causal.test_mask).any()
    assert causal.train_mask[:28].all()
    assert causal.test_mask[34:].all()