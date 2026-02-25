"""Tests for the core CWRA algorithm (cwra.py)."""

import numpy as np
import pandas as pd

from cwra import (
    CWRAConfig,
    compute_bedroc,
    compute_ef,
    evaluate_weights,
    normalize_modalities,
)

# Also import helpers that aren't re-exported through the package.
import importlib, sys, pathlib
_cwra_path = pathlib.Path(__file__).resolve().parent.parent / "cwra.py"
_spec = importlib.util.spec_from_file_location("_cwra_direct", str(_cwra_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
project_to_capped_simplex = _mod.project_to_capped_simplex
normalize_weights = _mod.normalize_weights
filter_modalities = _mod.filter_modalities
objective_fair = _mod.objective_fair


# ---- project_to_capped_simplex -----------------------------------------

def test_simplex_uniform():
    """Equal input -> equal weights summing to 1."""
    v = np.array([1.0, 1.0, 1.0, 1.0])
    w = project_to_capped_simplex(v, lo=0.0, hi=1.0)
    assert abs(w.sum() - 1.0) < 1e-10
    assert np.allclose(w, 0.25)


def test_simplex_respects_bounds():
    v = np.array([10.0, 0.0, 0.0])
    w = project_to_capped_simplex(v, lo=0.05, hi=0.6)
    assert abs(w.sum() - 1.0) < 1e-10
    assert np.all(w >= 0.05 - 1e-12)
    assert np.all(w <= 0.60 + 1e-12)


def test_simplex_infeasible_raises():
    # sum(lo) > 1  ->  infeasible
    try:
        project_to_capped_simplex(np.array([1.0, 1.0]), lo=0.6, hi=1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---- normalize_weights --------------------------------------------------

def test_normalize_weights_sums_to_one():
    w = normalize_weights(np.array([3.0, 1.0, 1.0]), 0.0, 1.0)
    assert abs(w.sum() - 1.0) < 1e-10
    assert w[0] > w[1]  # preserves relative order


def test_normalize_weights_with_bounds():
    w = normalize_weights(np.array([10.0, 0.0, 0.0]), 0.05, 0.6)
    assert abs(w.sum() - 1.0) < 1e-10
    assert np.all(w >= 0.05 - 1e-12)


# ---- CWRAConfig --------------------------------------------------------

def test_cwra_config_defaults():
    cfg = CWRAConfig()
    assert len(cfg.modalities) == 11
    assert "vina_score" in cfg.modalities
    assert cfg.modalities["vina_score"][0] == "low"


def test_cwra_config_seed_field():
    cfg = CWRAConfig()
    cfg.seed = 123
    assert cfg.seed == 123


# ---- normalize_modalities -----------------------------------------------

def _make_df(n=200, seed=42):
    """Small synthetic dataset with 3 modalities and 20 actives."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "smiles": [f"C{'C' * i}" for i in range(n)],
        "source": ["initial_370"] * 20 + ["G1"] * (n - 20),
        "mod_high": rng.rand(n),
        "mod_low": rng.rand(n),
        "mod_flat": np.ones(n) * 5.0,
    })
    return df


def test_normalize_modalities_minmax():
    df = _make_df()
    mods = {
        "mod_high": ("high", "HighMod"),
        "mod_low": ("low", "LowMod"),
        "mod_flat": ("high", "FlatMod"),
    }
    X, cols, names = normalize_modalities(df, mods, "minmax")
    assert X.shape == (200, 3)
    # minmax should produce [0, 1] range
    assert X[:, 0].min() >= -1e-10
    assert X[:, 0].max() <= 1.0 + 1e-10
    # 'low' direction column should be flipped
    assert cols[1] == "mod_low"
    # constant column -> all zeros
    assert np.allclose(X[:, 2], 0.0)


def test_normalize_modalities_rank():
    df = _make_df()
    mods = {"mod_high": ("high", "H")}
    X, _, _ = normalize_modalities(df, mods, "rank")
    assert X.shape == (200, 1)
    assert X.min() >= -1e-10
    assert X.max() <= 1.0 + 1e-10


def test_normalize_modalities_skips_missing():
    df = _make_df()
    mods = {
        "mod_high": ("high", "H"),
        "nonexistent_col": ("high", "Ghost"),
    }
    X, cols, names = normalize_modalities(df, mods, "minmax")
    assert len(cols) == 1
    assert "nonexistent_col" not in cols


# ---- compute_ef ---------------------------------------------------------

def test_ef_perfect_ranking():
    """All actives at the top -> EF@k% = N/A_total."""
    n, n_active = 1000, 10
    scores = np.zeros(n)
    scores[:n_active] = 10.0  # actives get highest score
    active = np.zeros(n, dtype=bool)
    active[:n_active] = True

    ef, hits, k = compute_ef(scores, active, cutoff_pct=1.0)
    # top 1% = 10 compounds, all 10 actives => EF = (10/10) / (10/1000) = 100
    assert ef == 100.0
    assert hits == 10


def test_ef_random_ranking():
    """Random scores -> EF close to 1.0 with large N."""
    rng = np.random.RandomState(0)
    n = 50000
    active = np.zeros(n, dtype=bool)
    active[:500] = True
    scores = rng.rand(n)

    ef, _, _ = compute_ef(scores, active, cutoff_pct=10.0)
    assert 0.7 < ef < 1.3  # within 30% of expected value 1.0


def test_ef_no_actives():
    scores = np.array([1.0, 2.0, 3.0])
    active = np.zeros(3, dtype=bool)
    ef, hits, _ = compute_ef(scores, active, cutoff_pct=50.0)
    assert ef == 0
    assert hits == 0


# ---- evaluate_weights ---------------------------------------------------

def test_evaluate_weights_equal():
    rng = np.random.RandomState(7)
    n, m = 500, 3
    X = rng.rand(n, m)
    active = np.zeros(n, dtype=bool)
    active[:25] = True  # 5% active rate

    w = np.array([1 / 3, 1 / 3, 1 / 3])
    result = evaluate_weights(w, X, active, cutoffs=[1, 5, 10])
    assert 1 in result
    assert result[1]["hits"] >= 0
    assert result[1]["ef"] >= 0


# ---- compute_bedroc -----------------------------------------------------

def test_bedroc_perfect():
    n = 200
    scores = np.arange(n, dtype=float)
    active = np.zeros(n, dtype=bool)
    active[-10:] = True  # top 10 are active
    b = compute_bedroc(scores, active, alpha=20.0)
    assert b > 0.9  # near-perfect ranking


def test_bedroc_no_actives():
    scores = np.array([1.0, 2.0])
    active = np.zeros(2, dtype=bool)
    assert compute_bedroc(scores, active) == 0.0


# ---- filter_modalities --------------------------------------------------

def test_filter_modalities_drops():
    mods = {
        "a": ("high", "A"),
        "b": ("low", "B"),
        "c": ("high", "C"),
    }
    filtered = filter_modalities(mods, ["B"])
    assert "b" not in filtered
    assert len(filtered) == 2


# ---- objective_fair (smoke) ---------------------------------------------

def test_objective_fair_returns_negative():
    """Fair objective should return a negative value (minimization)."""
    rng = np.random.RandomState(0)
    n, m = 100, 4
    X = rng.rand(n, m)
    active = np.zeros(n, dtype=bool)
    active[:10] = True
    w = np.array([0.25, 0.25, 0.25, 0.25])
    cutoff_weights = {1: 5, 5: 2, 10: 1}

    val = objective_fair(w, X, active, cutoff_weights, 0.0, 1.0)
    assert isinstance(val, float)
    assert val < 0  # negative because it's -sum(ef)


# ---- end-to-end: normalize -> weight -> evaluate -------------------------

def test_end_to_end_pipeline():
    """Full mini pipeline: normalize -> score -> evaluate."""
    df = _make_df(n=500, seed=99)
    mods = {
        "mod_high": ("high", "H"),
        "mod_low": ("low", "L"),
    }
    X, cols, names = normalize_modalities(df, mods, "minmax")
    active = (df["source"] == "initial_370").values

    # equal weights
    w = np.array([0.5, 0.5])
    result = evaluate_weights(w, X, active, cutoffs=[1, 5, 10])

    for c in [1, 5, 10]:
        assert c in result
        assert result[c]["ef"] >= 0
        assert result[c]["hits"] >= 0
        assert result[c]["k"] > 0
