import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pu_meta_model import (
    build_feature_matrix,
    calibrate_model,
    standardize_features,
    train_pu_model,
)


def test_build_feature_matrix_and_names():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"id": [1, 2, 3]})
    X_mod = rng.rand(3, 2)
    desc_df = pd.DataFrame({"MW": ["1.0", "2.0", "3.0"], "cLogP": [0.1, 0.2, 0.3]})
    X, names = build_feature_matrix(df, X_mod, desc_df)
    assert X.shape == (3, 4)
    assert names == ["mod:0", "mod:1", "desc:MW", "desc:cLogP"]
    assert X.dtype == np.float32


def test_train_and_calibrate():
    rng = np.random.RandomState(1)
    X = rng.randn(50, 4)
    y = np.array([1] * 25 + [0] * 25)
    model = train_pu_model(X, y, seed=7)

    X_cal = rng.randn(20, 4)
    y_cal = np.array([1] * 10 + [0] * 10)
    cal = calibrate_model(model, X_cal, y_cal, method="sigmoid")
    proba = cal.predict_proba(X_cal)[:, 1]
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_standardize_features_train_only():
    rng = np.random.RandomState(2)
    X_train = rng.randn(10, 3)
    X_extra = rng.randn(5, 3) + 5.0
    X_all = np.vstack([X_train, X_extra])
    scaler, X_train_s, X_all_s = standardize_features(X_train, X_all)

    assert np.allclose(scaler.mean_, X_train.mean(axis=0))
    assert np.allclose(X_train_s.mean(axis=0), 0.0, atol=1e-6)
    assert not np.allclose(X_all_s.mean(axis=0), 0.0, atol=1e-3)
