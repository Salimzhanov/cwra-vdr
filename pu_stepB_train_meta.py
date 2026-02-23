#!/usr/bin/env python3
"""
Step B CLI: train, calibrate, and score PU meta-model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from cwra_compat import CWRAConfig, normalize_modalities
from pu_descriptors import compute_physchem_descriptors
from pu_meta_model import (
    build_feature_matrix,
    calibrate_model,
    standardize_features,
    train_pu_model,
    train_pu_model_gbt,
)

logger = logging.getLogger(__name__)


def _choose_smiles_col(df: pd.DataFrame) -> Optional[str]:
    if "smiles" in df.columns:
        return "smiles"
    if "SMILES" in df.columns:
        return "SMILES"
    return None


def _configure_active_sources(config: CWRAConfig, include_newref_137_as_active: bool) -> None:
    if not include_newref_137_as_active:
        return
    if "newRef_137" not in config.active_sources:
        config.active_sources.append("newRef_137")
    config.exclude_sources = [s for s in config.exclude_sources if s != "newRef_137"]


def _align_labels(labels_df: pd.DataFrame, df_pool: pd.DataFrame) -> pd.Series:
    if "index" in labels_df.columns:
        labels_df = labels_df.set_index("index")
        labels_df = labels_df.reindex(df_pool.index)
        if labels_df["pu_label"].isna().any():
            raise ValueError("pu_labels index mismatch with df_pool.")
        return labels_df["pu_label"].astype(int)
    if "id" in labels_df.columns and "id" in df_pool.columns:
        labels_df = labels_df.set_index("id")
        labels_df = labels_df.reindex(df_pool["id"].values)
        if labels_df["pu_label"].isna().any():
            raise ValueError("pu_labels id mismatch with df_pool.")
        return labels_df["pu_label"].astype(int).reset_index(drop=True)
    if len(labels_df) != len(df_pool):
        raise ValueError("pu_labels length does not match df_pool.")
    return labels_df["pu_label"].astype(int)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step B: train and score PU meta-model.")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--pu-labels", required=True, help="PU labels CSV from Step A")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-frac", type=float, default=0.3)
    parser.add_argument(
        "--model-type",
        choices=["lr", "gbt"],
        default="lr",
        help="Model type: lr (logistic regression) or gbt (gradient boosting)",
    )
    parser.add_argument(
        "--include-newref-137-as-active",
        action="store_true",
        default=False,
        help="Treat source 'newRef_137' as active (remove from exclude_sources).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = CWRAConfig()
    _configure_active_sources(config, args.include_newref_137_as_active)

    df = pd.read_csv(args.input)
    df_pool = df[~df["source"].isin(config.exclude_sources)].copy()
    logger.info("Loaded %d rows, pool after exclude_sources: %d", len(df), len(df_pool))
    if args.include_newref_137_as_active:
        logger.info("Including 'newRef_137' as active source.")

    labels_df = pd.read_csv(args.pu_labels)
    pu_label = _align_labels(labels_df, df_pool)

    X_mod, _, _ = normalize_modalities(df_pool, config.modalities)
    logger.info("X_mod shape: %s", X_mod.shape)

    smiles_col = _choose_smiles_col(df_pool)
    desc_df = compute_physchem_descriptors(df_pool, smiles_col=smiles_col)

    X_all, feature_names = build_feature_matrix(df_pool, X_mod, desc_df)
    logger.info("Total features: %d", len(feature_names))

    # Drop rows with any NaN/inf in features
    finite_mask = np.isfinite(X_all).all(axis=1)
    if not np.all(finite_mask):
        dropped = int((~finite_mask).sum())
        logger.warning("Dropping %d rows with NaN/inf in features.", dropped)
        X_all = X_all[finite_mask]
        df_pool = df_pool.loc[finite_mask].copy()
        pu_label = pu_label.iloc[finite_mask].reset_index(drop=True)

    labeled_mask = pu_label.isin([0, 1]).values
    X_labeled = X_all[labeled_mask]
    y_labeled = pu_label.values[labeled_mask]
    logger.info("Labeled set size: %d (pos=%d neg=%d)", len(y_labeled), int((y_labeled == 1).sum()), int((y_labeled == 0).sum()))

    if len(np.unique(y_labeled)) < 2:
        raise ValueError("Both positive and negative labels are required for training.")

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_labeled,
        y_labeled,
        test_size=args.calib_frac,
        random_state=args.seed,
        stratify=y_labeled,
    )
    logger.info("Train size: %d, Calib size: %d", len(y_train), len(y_cal))

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_cal_i = imputer.transform(X_cal)
    X_all_i = imputer.transform(X_all)

    scaler, X_train_s, X_all_s = standardize_features(X_train_i, X_all_i)
    X_cal_s = scaler.transform(X_cal_i)

    logger.info("Model type: %s", args.model_type)
    if args.model_type == "gbt":
        model = train_pu_model_gbt(X_train_s, y_train, seed=args.seed)
    else:
        model = train_pu_model(X_train_s, y_train, seed=args.seed)
    cal_model = calibrate_model(model, X_cal_s, y_cal, method="sigmoid")

    meta_score = cal_model.predict_proba(X_all_s)[:, 1]
    logger.info("Scored all rows: %d", len(meta_score))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(cal_model, output_dir / "meta_model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")

    schema = {"feature_names": feature_names}
    with open(output_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    meta_scores = pd.DataFrame(
        {
            "meta_score": meta_score,
            "pu_label": pu_label.values,
            "source": df_pool["source"].values,
        }
    )
    if "id" in df_pool.columns:
        meta_scores.insert(0, "id", df_pool["id"].values)
    else:
        meta_scores.insert(0, "index", df_pool.index.values)
    if smiles_col:
        meta_scores[smiles_col] = df_pool[smiles_col].values

    meta_scores.to_csv(output_dir / "meta_scores.csv", index=False)

    logger.info("Wrote outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
