#!/usr/bin/env python3
"""
Step D: weighted conformal p-values.
"""

from __future__ import annotations

import numpy as np

from conformal_utils import conformal_pvalues


def weighted_conformal_pvalues(alpha_cal, w_cal, alpha_all, w_all=None):
    """
    Weighted conformal p-values.
    """
    alpha_cal = np.asarray(alpha_cal, dtype=float)
    alpha_all = np.asarray(alpha_all, dtype=float)
    w_cal = np.asarray(w_cal, dtype=float)

    if alpha_cal.ndim != 1 or alpha_all.ndim != 1 or w_cal.ndim != 1:
        raise ValueError("alpha_cal, w_cal, and alpha_all must be 1D arrays.")
    if alpha_cal.size != w_cal.size:
        raise ValueError("alpha_cal and w_cal must have the same length.")

    if w_all is None:
        w_all = np.ones_like(alpha_all, dtype=float)
    else:
        w_all = np.asarray(w_all, dtype=float)
        if w_all.shape != alpha_all.shape:
            raise ValueError("w_all must have the same shape as alpha_all.")

    # Sort calibration alphas with weights for cumulative sums
    order = np.argsort(alpha_cal)
    cal_sorted = alpha_cal[order]
    w_sorted = w_cal[order]
    w_cum = np.cumsum(w_sorted)
    w_total = w_cum[-1]

    # For each alpha_x: sum weights with alpha_cal >= alpha_x
    idx = np.searchsorted(cal_sorted, alpha_all, side="left")
    w_ge = w_total - np.where(idx == 0, 0.0, w_cum[idx - 1])

    numerator = w_ge + w_all
    denominator = w_total + w_all
    p_weighted = numerator / denominator

    p_unweighted = conformal_pvalues(alpha_cal, alpha_all)
    return p_weighted, p_unweighted
