#!/usr/bin/env python3
"""
Step C: conformal utilities (unweighted).
"""

from __future__ import annotations

import numpy as np


def nonconformity_from_scores(scores):
    """
    Convert scores in [0,1] (higher = more active-like) to nonconformity.
    Nonconformity increases with active-likeness; large nonconformity => small
    p-value under RN-null.
    """
    scores = np.asarray(scores, dtype=float)
    scores = np.clip(scores, 0.0, 1.0)
    return scores


def conformal_pvalues(alpha_cal, alpha_all):
    """
    Compute conformal p-values for all points.
    p(x) = (count(alpha_cal >= alpha_x) + 1) / (n_cal + 1)
    """
    alpha_cal = np.asarray(alpha_cal, dtype=float)
    alpha_all = np.asarray(alpha_all, dtype=float)
    if alpha_cal.ndim != 1 or alpha_all.ndim != 1:
        raise ValueError("alpha_cal and alpha_all must be 1D arrays.")

    n_cal = alpha_cal.size
    if n_cal == 0:
        raise ValueError("alpha_cal must be non-empty.")

    # Use sorting + search to vectorize: count >= alpha_x
    cal_sorted = np.sort(alpha_cal)
    # number >= x = n - index of first >? We want >=, so use left for x
    idx = np.searchsorted(cal_sorted, alpha_all, side="left")
    counts_ge = n_cal - idx
    pvals = (counts_ge + 1.0) / (n_cal + 1.0)
    return pvals
