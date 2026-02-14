#!/usr/bin/env python3
"""
Step B: PU meta-model training utilities (no CLI).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import logging

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_feature_matrix(
    df_pool: pd.DataFrame,
    X_mod: np.ndarray,
    desc_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Concatenate modality features and descriptor features.
    """
    X_mod = np.asarray(X_mod)
    if X_mod.ndim != 2:
        raise ValueError("X_mod must be a 2D array.")

    n, m = X_mod.shape
    if len(df_pool) != n:
        raise ValueError("df_pool and X_mod must have the same number of rows.")

    feature_names = [f"mod:{i}" for i in range(m)]

    if desc_df is None or desc_df.shape[1] == 0:
        logger.info("No descriptors provided; using only modality features.")
        X = X_mod.astype(np.float32, copy=False)
        return X, feature_names

    desc_df = desc_df.reindex(df_pool.index)
    desc_values = desc_df.to_numpy()
    X = np.concatenate([X_mod, desc_values], axis=1).astype(np.float32, copy=False)
    feature_names.extend([f"desc:{c}" for c in desc_df.columns])
    logger.info("Built feature matrix: %d rows, %d features", X.shape[0], X.shape[1])
    return X, feature_names


def train_pu_model(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """
    Train a weighted logistic regression PU model on labeled data only.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative labels are required.")

    n = len(y)
    w_pos = n / (2 * n_pos)
    w_neg = n / (2 * n_neg)
    sample_weight = np.where(y == 1, w_pos, w_neg)
    logger.info("Training PU model: n=%d pos=%d neg=%d w_pos=%.3f w_neg=%.3f", n, n_pos, n_neg, w_pos, w_neg)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_pu_model_gbt(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """
    Train a weighted GradientBoosting PU model on labeled data only.
    Conservative hyperparameters to avoid overfitting with 366 positives.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative labels are required.")

    n = len(y)
    w_pos = n / (2 * n_pos)
    w_neg = n / (2 * n_neg)
    sample_weight = np.where(y == 1, w_pos, w_neg)
    logger.info(
        "Training PU GBT model: n=%d pos=%d neg=%d w_pos=%.3f w_neg=%.3f",
        n,
        n_pos,
        n_neg,
        w_pos,
        w_neg,
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=seed,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def calibrate_model(model, X_cal: np.ndarray, y_cal: np.ndarray, method: str = "sigmoid"):
    """
    Calibrate a prefit model using held-out labeled data.
    """
    logger.info("Calibrating model with method=%s on %d samples", method, len(y_cal))
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method, cv=5)
    calibrator.fit(X_cal, y_cal)
    return calibrator


def standardize_features(X_train: np.ndarray, X_all: np.ndarray):
    """
    Standardize features using statistics from X_train only.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)
    logger.info("Standardized features using train-only statistics.")
    return scaler, X_train_scaled, X_all_scaled
