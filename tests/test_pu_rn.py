import numpy as np
import pandas as pd

from pu_rn import build_reliable_negative_mask


def _make_data():
    rng = np.random.RandomState(0)
    n = 20
    m = 4
    X_mod = rng.rand(n, m)
    active_mask = np.zeros(n, dtype=bool)
    active_mask[[1, 5, 9]] = True
    df_pool = pd.DataFrame({"id": np.arange(n)})
    return df_pool, active_mask, X_mod


def test_rn_mask_basic():
    df_pool, active_mask, X_mod = _make_data()
    rn_mask = build_reliable_negative_mask(df_pool, active_mask, X_mod, seed=123, smiles_col=None)
    assert rn_mask.dtype == bool
    assert rn_mask.sum() <= 10 * active_mask.sum()
    assert not np.any(rn_mask & active_mask)


def test_rn_mask_deterministic():
    df_pool, active_mask, X_mod = _make_data()
    rn_a = build_reliable_negative_mask(df_pool, active_mask, X_mod, seed=42, smiles_col=None)
    rn_b = build_reliable_negative_mask(df_pool, active_mask, X_mod, seed=42, smiles_col=None)
    assert np.array_equal(rn_a, rn_b)


def test_missing_smiles_column():
    df_pool, active_mask, X_mod = _make_data()
    rn_mask = build_reliable_negative_mask(
        df_pool,
        active_mask,
        X_mod,
        seed=7,
        smiles_col="smiles",
        use_rdkit=True,
    )
    assert rn_mask.sum() <= 10 * active_mask.sum()
    assert not np.any(rn_mask & active_mask)
