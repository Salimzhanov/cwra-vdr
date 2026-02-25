import numpy as np

from shift_weights import compute_importance_weights


def test_shift_weights_detects_shift():
    rng = np.random.RandomState(0)
    X_cal = rng.normal(0, 1, size=(200, 2))
    X_tgt = rng.normal(2.0, 1, size=(200, 2))
    w_cal, _ = compute_importance_weights(X_cal, X_tgt, seed=0, clip=20)
    assert w_cal.std() > 0.05


def test_shift_weights_no_shift_mean_near_one():
    rng = np.random.RandomState(1)
    X_cal = rng.normal(0, 1, size=(200, 2))
    X_tgt = rng.normal(0, 1, size=(200, 2))
    w_cal, _ = compute_importance_weights(X_cal, X_tgt, seed=1, clip=20)
    assert abs(w_cal.mean() - 1.0) < 0.2
