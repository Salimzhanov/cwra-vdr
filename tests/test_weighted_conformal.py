import numpy as np

from weighted_conformal import weighted_conformal_pvalues


def test_weighted_equals_unweighted_when_all_one():
    alpha_cal = np.array([0.1, 0.2, 0.5])
    w_cal = np.ones_like(alpha_cal)
    alpha_all = np.array([0.2, 0.3])
    p_w, p_u = weighted_conformal_pvalues(alpha_cal, w_cal, alpha_all)
    assert np.allclose(p_w, p_u)


def test_monotonic_sanity():
    alpha_cal = np.array([0.1, 0.2, 0.5, 0.9])
    w_cal = np.ones_like(alpha_cal)
    alpha_all = np.array([0.01, 0.9])
    p_w, _ = weighted_conformal_pvalues(alpha_cal, w_cal, alpha_all)
    assert p_w[0] >= p_w[1]
