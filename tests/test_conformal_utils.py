import numpy as np

from conformal_utils import conformal_pvalues, nonconformity_from_scores


def test_nonconformity_from_scores():
    scores = np.array([0.0, 0.2, 1.0])
    alpha = nonconformity_from_scores(scores)
    assert np.allclose(alpha, np.array([0.0, 0.2, 1.0]))


def test_conformal_pvalues_simple():
    alpha_cal = np.array([0.1, 0.2, 0.5])
    alpha_all = np.array([0.2, 0.3])
    # For 0.2: count >= 0.2 = 2 (0.2, 0.5) -> (2+1)/(3+1)=0.75
    # For 0.3: count >= 0.3 = 1 (0.5) -> (1+1)/(3+1)=0.5
    p = conformal_pvalues(alpha_cal, alpha_all)
    assert np.allclose(p, np.array([0.75, 0.5]))
    assert np.all(p > 0) and np.all(p <= 1)
