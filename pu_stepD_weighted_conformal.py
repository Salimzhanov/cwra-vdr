#!/usr/bin/env python3
"""
Step D CLI: compute weighted conformal p-values end-to-end.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from cwra_compat import CWRAConfig, normalize_modalities
from pu_descriptors import compute_physchem_descriptors
from pu_meta_model import build_feature_matrix, standardize_features
from shift_weights import compute_importance_weights
from weighted_conformal import weighted_conformal_pvalues
from conformal_utils import nonconformity_from_scores

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


def _select_calibration_indices(
    pu_label: pd.Series,
    calib_set: str,
    calib_frac: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if calib_set == "labeled":
        pool_idx = np.where(np.isin(pu_label.values, [0, 1]))[0]
    elif calib_set == "rn_only":
        pool_idx = np.where(pu_label.values == 0)[0]
    elif calib_set == "labeled_balanced":
        pos_idx = np.where(pu_label.values == 1)[0]
        rn_idx = np.where(pu_label.values == 0)[0]
        n = min(len(pos_idx), len(rn_idx))
        if n == 0:
            pool_idx = np.array([], dtype=int)
        else:
            pos_s = rng.choice(pos_idx, size=n, replace=False)
            rn_s = rng.choice(rn_idx, size=n, replace=False)
            pool_idx = np.concatenate([pos_s, rn_s])
    else:
        raise ValueError("calib_set must be one of: labeled, rn_only, labeled_balanced.")

    if pool_idx.size == 0:
        raise ValueError("Calibration pool is empty.")

    if calib_set == "rn_only":
        return pool_idx
    n_cal = max(1, int(np.floor(calib_frac * pool_idx.size)))
    if n_cal > pool_idx.size:
        n_cal = pool_idx.size
    return rng.choice(pool_idx, size=n_cal, replace=False)


def _align_is_calib(conf_df: pd.DataFrame, df_pool: pd.DataFrame) -> Optional[np.ndarray]:
    if "is_calib" not in conf_df.columns:
        return None

    if "id" in conf_df.columns and "id" in df_pool.columns:
        aligned = conf_df.set_index("id").reindex(df_pool["id"].values)
    elif "index" in conf_df.columns:
        if "index" in df_pool.columns:
            key = df_pool["index"].values
        else:
            key = df_pool.index.values
        aligned = conf_df.set_index("index").reindex(key)
    elif "row" in conf_df.columns:
        if len(conf_df) != len(df_pool):
            raise ValueError("conformal_pvalues row alignment requires matching length.")
        aligned = conf_df
    else:
        return None

    if aligned["is_calib"].isna().any():
        raise ValueError("conformal_pvalues is_calib alignment mismatch with df_pool.")

    return aligned["is_calib"].astype(int).to_numpy()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step D: weighted conformal p-values.")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--pu-labels", required=True, help="PU labels CSV")
    parser.add_argument("--meta-scores", required=True, help="meta_scores.csv from Step B")
    parser.add_argument("--conformal", default=None, help="Optional conformal_pvalues.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--calib-frac", type=float, default=0.3)
    parser.add_argument(
        "--calib-set",
        choices=["rn_only", "labeled", "labeled_balanced"],
        default="rn_only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip", type=float, default=20)
    parser.add_argument(
        "--target",
        choices=["unlabeled", "generated_only"],
        default="unlabeled",
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
    if args.include_newref_137_as_active:
        logger.info("Including 'newRef_137' as active source.")

    labels_df = pd.read_csv(args.pu_labels)
    pu_label = _align_labels(labels_df, df_pool)

    meta_scores = pd.read_csv(args.meta_scores)
    if len(meta_scores) != len(df_pool):
        if "index" in meta_scores.columns and meta_scores["index"].isin(df_pool.index).all():
            meta_scores = meta_scores.set_index("index")
            df_pool = df_pool.loc[meta_scores.index].copy()
            pu_label = _align_labels(labels_df, df_pool).reset_index(drop=True)
            meta_scores = meta_scores.reset_index()
            logger.warning("Aligned df_pool to meta_scores using index column.")
        elif "id" in meta_scores.columns and "id" in df_pool.columns:
            df_pool = df_pool.set_index("id")
            meta_scores = meta_scores.set_index("id")
            df_pool = df_pool.loc[meta_scores.index].copy()
            df_pool = df_pool.reset_index()
            pu_label = _align_labels(labels_df, df_pool).reset_index(drop=True)
            meta_scores = meta_scores.reset_index()
            logger.warning("Aligned df_pool to meta_scores using id column.")
        else:
            raise ValueError("meta_scores length does not match df_pool and no align key found.")

    X_mod, _, _ = normalize_modalities(df_pool, config.modalities)
    smiles_col = _choose_smiles_col(df_pool)
    desc_df = compute_physchem_descriptors(df_pool, smiles_col=smiles_col)
    X_all, feature_names = build_feature_matrix(df_pool, X_mod, desc_df)

    # Drop rows with NaN/inf to match Step B behavior
    finite_mask = np.isfinite(X_all).all(axis=1)
    if not np.all(finite_mask):
        dropped = int((~finite_mask).sum())
        logger.warning("Dropping %d rows with NaN/inf in features.", dropped)
        X_all = X_all[finite_mask]
        df_pool = df_pool.loc[finite_mask].copy()
        pu_label = pu_label.iloc[finite_mask].reset_index(drop=True)
        meta_scores = meta_scores.iloc[finite_mask].reset_index(drop=True)

    if len(meta_scores) != len(df_pool):
        raise ValueError("meta_scores length does not match df_pool after filtering.")

    # Standardize using saved scaler if available
    scaler_path = Path(args.meta_scores).parent / "scaler.joblib"
    labeled_mask = pu_label.isin([0, 1]).values
    if labeled_mask.sum() == 0:
        raise ValueError("No labeled data available to fit scaler.")

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if getattr(scaler, "n_features_in_", X_all.shape[1]) != X_all.shape[1]:
            logger.warning(
                "Scaler feature count (%d) does not match X_all (%d); refitting on labeled set.",
                getattr(scaler, "n_features_in_", -1),
                X_all.shape[1],
            )
            _, X_train_s, X_all_s = standardize_features(X_all[labeled_mask], X_all)
        else:
            X_all_s = scaler.transform(X_all)
            logger.info("Loaded scaler from %s", scaler_path)
    else:
        logger.warning("Scaler not found; fitting on labeled set only.")
        _, X_train_s, X_all_s = standardize_features(X_all[labeled_mask], X_all)

    # Calibration indices
    cal_idx = None
    is_calib = None
    if args.conformal:
        conf_df = pd.read_csv(args.conformal)
        if "is_calib" in conf_df.columns:
            try:
                is_calib = _align_is_calib(conf_df, df_pool)
            except ValueError as exc:
                raise ValueError(f"Failed to align is_calib from Step C: {exc}") from exc

            if is_calib is None:
                raise ValueError("Failed to align is_calib from Step C.")

            cal_idx = np.where(is_calib == 1)[0]
            if cal_idx.size == 0:
                raise ValueError("No calibration indices found in Step C is_calib.")
            logger.info("Using calibration indices from Step C (n=%d).", cal_idx.size)
        else:
            logger.warning("conformal_pvalues.csv missing is_calib; sampling anew.")

    if cal_idx is None:
        cal_idx = _select_calibration_indices(
            pu_label,
            calib_set=args.calib_set,
            calib_frac=args.calib_frac,
            seed=args.seed,
        )
        logger.info("Sampled calibration indices (n=%d).", cal_idx.size)
        is_calib = np.zeros(len(df_pool), dtype=int)
        is_calib[cal_idx] = 1

    # Target indices
    if args.target == "generated_only":
        gen_sources = {"G1", "G2", "G3", "G4"}
        tgt_mask = (pu_label.values == -1) & df_pool["source"].isin(gen_sources).values
        if not np.any(tgt_mask):
            logger.warning("No generated_only targets found; falling back to unlabeled.")
            tgt_mask = pu_label.values == -1
    else:
        tgt_mask = pu_label.values == -1

    if not np.any(tgt_mask):
        raise ValueError("Target set is empty.")

    X_cal = X_all_s[cal_idx]
    X_tgt = X_all_s[tgt_mask]

    w_cal, w_all = compute_importance_weights(
        X_cal, X_tgt, X_all=X_all_s, seed=args.seed, clip=args.clip
    )

    alpha_cal = nonconformity_from_scores(meta_scores.loc[cal_idx, "meta_score"].values)
    alpha_all = nonconformity_from_scores(meta_scores["meta_score"].values)

    p_w, p_u = weighted_conformal_pvalues(alpha_cal, w_cal, alpha_all, w_all=w_all)

    out_df = pd.DataFrame(
        {
            "meta_score": meta_scores["meta_score"].values,
            "pu_label": meta_scores["pu_label"].values,
            "is_calib": is_calib,
            "pval_unweighted": p_u,
            "pval_weighted": p_w,
            "weight": w_all,
        }
    )
    if "id" in df_pool.columns:
        out_df.insert(0, "id", df_pool["id"].values)
    else:
        out_df.insert(0, "index", df_pool.index.values)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "weighted_pvalues.csv", index=False)
    logger.info("Wrote outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
