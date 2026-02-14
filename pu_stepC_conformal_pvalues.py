#!/usr/bin/env python3
"""
Step C CLI: compute unweighted conformal p-values.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from conformal_utils import conformal_pvalues, nonconformity_from_scores

logger = logging.getLogger(__name__)


def _select_calibration_indices(
    df: pd.DataFrame,
    calib_set: str,
    calib_frac: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)

    if calib_set == "labeled":
        pool_idx = df.index[df["pu_label"].isin([0, 1])].to_numpy()
    elif calib_set == "rn_only":
        pool_idx = df.index[df["pu_label"] == 0].to_numpy()
    elif calib_set == "labeled_balanced":
        pos_idx = df.index[df["pu_label"] == 1].to_numpy()
        rn_idx = df.index[df["pu_label"] == 0].to_numpy()
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

    # For rn_only: use ALL reliable negatives (no subsampling).
    # For other modes: subsample at calib_frac.
    if calib_set == "rn_only":
        return pool_idx

    n_cal = max(1, int(np.floor(calib_frac * pool_idx.size)))
    if n_cal > pool_idx.size:
        n_cal = pool_idx.size
    return rng.choice(pool_idx, size=n_cal, replace=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step C: unweighted conformal p-values.")
    parser.add_argument("--meta-scores", required=True, help="meta_scores.csv from Step B")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--calib-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calib-indices",
        default=None,
        help="Optional calib_negatives.csv with an index column to define calibration set",
    )
    parser.add_argument(
        "--calib-set",
        choices=["labeled", "rn_only", "labeled_balanced"],
        default="rn_only",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(args.meta_scores)
    if "meta_score" not in df.columns or "pu_label" not in df.columns:
        raise ValueError("meta_scores.csv must contain meta_score and pu_label columns.")

    if args.calib_indices:
        cal_df = pd.read_csv(args.calib_indices)
        if "index" not in cal_df.columns:
            raise ValueError("calib_negatives.csv must contain an index column.")
        if "index" not in df.columns:
            raise ValueError("meta_scores.csv must contain 'index' when using --calib-indices")
        df = df.set_index("index", drop=False)
        cal_idx = cal_df["index"].values
        if not np.isin(cal_idx, df.index.values).all():
            raise ValueError("calib_negatives indices do not align with meta_scores.csv.")
    else:
        cal_idx = _select_calibration_indices(df, args.calib_set, args.calib_frac, args.seed)
    logger.info("Calibration set size: %d", len(cal_idx))
    logger.info("Min achievable p-value: %.6f", 1.0 / (len(cal_idx) + 1))

    alpha_cal = nonconformity_from_scores(df.loc[cal_idx, "meta_score"].values)
    alpha_all = nonconformity_from_scores(df["meta_score"].values)

    pvals = conformal_pvalues(alpha_cal, alpha_all)

    is_calib = df.index.isin(cal_idx).astype(int)

    if "id" in df.columns:
        align_key = df["id"].values
        align_key_name = "id"
    elif "index" in df.columns:
        align_key = df["index"].values
        align_key_name = "index"
    else:
        align_key = np.arange(len(df))
        align_key_name = "row"

    out_df = pd.DataFrame(
        {
            align_key_name: align_key,
            "meta_score": df["meta_score"].values,
            "pu_label": df["pu_label"].values,
            "is_calib": is_calib,
            "pval_unweighted": pvals,
        }
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "conformal_pvalues.csv", index=False)

    if "pu_label" in df.columns:
        pos_med = df.loc[df["pu_label"] == 1, "meta_score"].size
        rn_med = df.loc[df["pu_label"] == 0, "meta_score"].size
        unl_med = df.loc[df["pu_label"] == -1, "meta_score"].size
        if pos_med > 0:
            pos_m = float(np.median(pvals[df["pu_label"] == 1]))
        else:
            pos_m = float("nan")
        if rn_med > 0:
            rn_m = float(np.median(pvals[df["pu_label"] == 0]))
        else:
            rn_m = float("nan")
        if unl_med > 0:
            unl_m = float(np.median(pvals[df["pu_label"] == -1]))
        else:
            unl_m = float("nan")
        logger.info(
            "P-value medians: pos=%.6f rn=%.6f unlabeled=%.6f",
            pos_m,
            rn_m,
            unl_m,
        )

    logger.info("Wrote outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
