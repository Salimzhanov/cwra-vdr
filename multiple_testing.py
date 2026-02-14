#!/usr/bin/env python3
"""
Multiple testing utilities.
"""

from __future__ import annotations

import numpy as np


def bh_qvalues(pvals):
    """
    Benjamini-Hochberg q-values for FDR control.
    NaNs are preserved and excluded from the correction.
    """
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)

    valid_mask = np.isfinite(pvals)
    if not valid_mask.any():
        return qvals

    p = pvals[valid_mask]
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)
    q_sorted = p_sorted * m / ranks

    # enforce monotone non-increasing from largest to smallest
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(p)
    q[order] = q_sorted
    qvals[valid_mask] = q
    return qvals
