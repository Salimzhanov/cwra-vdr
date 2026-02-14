#!/usr/bin/env python3
"""
Step D: density-ratio estimation under covariate shift.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_domain_classifier(
    X_cal, X_tgt, seed: int = 42, use_balanced: bool = False
) -> LogisticRegression:
    """
    Fit a domain classifier to distinguish calibration (0) vs target (1).
    """
    X_cal = np.asarray(X_cal)
    X_tgt = np.asarray(X_tgt)
    X = np.vstack([X_cal, X_tgt])
    y = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_tgt))])

    class_weight = "balanced" if use_balanced else None
    model = LogisticRegression(
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
        class_weight=class_weight,
    )
    model.fit(X, y)
    return model


def importance_weights_from_domain_proba(
    p_domain, n_cal: int, n_tgt: int, clip: float = 20
):
    """
    Convert p_domain = P(domain=1|x) to importance weights.
    """
    p_domain = np.asarray(p_domain, dtype=float)
    eps = 1e-12
    p = np.clip(p_domain, eps, 1 - eps)
    # Density ratio via logistic odds.
    w = (p / (1 - p)) * (n_cal / n_tgt)
    w = np.clip(w, 1.0 / clip, clip)
    return w


def compute_importance_weights(
    X_cal,
    X_tgt,
    X_all: Optional[np.ndarray] = None,
    seed: int = 42,
    clip: float = 20,
    use_balanced: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fit domain model and compute importance weights for calibration (and optionally all).
    """
    X_cal = np.asarray(X_cal)
    X_tgt = np.asarray(X_tgt)
    n_cal = len(X_cal)
    n_tgt = len(X_tgt)
    if n_cal == 0 or n_tgt == 0:
        raise ValueError("X_cal and X_tgt must be non-empty.")

    model = fit_domain_classifier(X_cal, X_tgt, seed=seed, use_balanced=use_balanced)

    n_cal_eff, n_tgt_eff = (1, 1) if use_balanced else (n_cal, n_tgt)

    p_cal = model.predict_proba(X_cal)[:, 1]
    w_cal = importance_weights_from_domain_proba(p_cal, n_cal_eff, n_tgt_eff, clip=clip)

    w_all = None
    if X_all is not None:
        X_all = np.asarray(X_all)
        p_all = model.predict_proba(X_all)[:, 1]
        w_all = importance_weights_from_domain_proba(p_all, n_cal_eff, n_tgt_eff, clip=clip)

    return w_cal, w_all
