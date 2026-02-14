#!/usr/bin/env python3
"""
Step E: final selection + report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from multiple_testing import bh_qvalues

logger = logging.getLogger(__name__)


def _choose_smiles_col(df: pd.DataFrame) -> Optional[str]:
    if "smiles" in df.columns:
        return "smiles"
    if "SMILES" in df.columns:
        return "SMILES"
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step E: final selection + report.")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--pvalues", required=True, help="weighted_pvalues.csv or conformal_pvalues.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["alpha", "bh"], default="bh")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--select-over", choices=["unlabeled", "all"], default="unlabeled")
    parser.add_argument(
        "--select-mode",
        choices=["bh", "pval_cutoff"],
        default="bh",
        help="Selection mode: bh (Benjamini-Hochberg) or pval_cutoff.",
    )
    parser.add_argument("--pval-cutoff", type=float, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help=(
            "Pre-filter: apply BH only to top-k compounds by meta_score within the "
            "universe. 0 = no filter. Reduces multiple-testing burden."
        ),
    )
    parser.add_argument(
        "--pval-type",
        choices=["weighted", "unweighted", "auto"],
        default="auto",
        help="Which p-value column to use. 'auto' prefers weighted if available.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(args.input)
    pvals_df = pd.read_csv(args.pvalues)

    if "meta_score" not in pvals_df.columns:
        raise ValueError("pvalues file must contain meta_score column.")

    if args.pval_type == "weighted":
        if "pval_weighted" not in pvals_df.columns:
            raise ValueError("--pval-type=weighted but pval_weighted column not found.")
        pval_col = "pval_weighted"
    elif args.pval_type == "unweighted":
        if "pval_unweighted" not in pvals_df.columns:
            raise ValueError("--pval-type=unweighted but pval_unweighted column not found.")
        pval_col = "pval_unweighted"
    else:  # auto
        if "pval_weighted" in pvals_df.columns:
            pval_col = "pval_weighted"
        elif "pval_unweighted" in pvals_df.columns:
            pval_col = "pval_unweighted"
        else:
            raise ValueError("pvalues file must contain pval_weighted or pval_unweighted.")

    logger.info("Using p-value column: %s", pval_col)

    if len(pvals_df) != len(df):
        if "index" in pvals_df.columns and pvals_df["index"].isin(df.index).all():
            idx = pvals_df["index"].values
            df = df.loc[idx].reset_index(drop=True)
            pvals_df = pvals_df.set_index("index").loc[idx].reset_index(drop=True)
            logger.warning("Aligned input to pvalues using index column.")
        elif "id" in pvals_df.columns and "id" in df.columns and pvals_df["id"].isin(df["id"]).all():
            idx = pvals_df["id"].values
            df = df.set_index("id").loc[idx].reset_index()
            pvals_df = pvals_df.set_index("id").loc[idx].reset_index()
            logger.warning("Aligned input to pvalues using id column.")
        else:
            raise ValueError("pvalues length must match input CSV length or include index/id for alignment.")

    df = df.copy()
    df["meta_score"] = pvals_df["meta_score"].values
    df["pval"] = pvals_df[pval_col].values
    if "pval_weighted" in pvals_df.columns:
        df["pval_weighted"] = pvals_df["pval_weighted"].values
    if "pval_unweighted" in pvals_df.columns:
        df["pval_unweighted"] = pvals_df["pval_unweighted"].values
    if "pu_label" in pvals_df.columns:
        df["pu_label"] = pvals_df["pu_label"].values

    if args.select_over == "unlabeled":
        universe_mask = df["pu_label"] == -1
    else:
        universe_mask = pd.Series(True, index=df.index)

    # Exclude positives from universe (always)
    if "pu_label" in df.columns:
        universe_mask = universe_mask & (df["pu_label"] != 1)

    n_unlabeled_total = int((df["pu_label"] == -1).sum()) if "pu_label" in df.columns else 0
    n_calib_neg = 0
    calib_path = Path(args.output).parent / "B2" / "calib_negatives.csv"
    if calib_path.exists():
        calib_df = pd.read_csv(calib_path)
        if "index" in calib_df.columns:
            calib_idx = calib_df["index"].astype(int).values
        elif "idx" in calib_df.columns:
            calib_idx = calib_df["idx"].astype(int).values
        else:
            raise ValueError("calib_negatives.csv must contain an index column.")
        universe_mask = universe_mask & (~df.index.isin(calib_idx))
        n_calib_neg = int(len(calib_idx))

    logger.info(
        "Universe exclusion: n_unlabeled_total=%d n_calib_neg=%d n_universe_after_exclusion=%d",
        n_unlabeled_total,
        n_calib_neg,
        int(universe_mask.sum()),
    )

    if args.top_k > 0 and universe_mask.sum() > args.top_k:
        universe_indices = df.index[universe_mask]
        top_k_indices = df.loc[universe_indices].nlargest(args.top_k, "meta_score").index
        prefilter_mask = pd.Series(False, index=df.index)
        prefilter_mask.loc[top_k_indices] = True
        universe_mask = universe_mask & prefilter_mask
        logger.info(
            "Pre-filtered BH universe to top %d by meta_score (was %d)",
            args.top_k,
            len(universe_indices),
        )

    if universe_mask.sum() > 0:
        pvals_universe = df.loc[universe_mask, "pval"].values
        p_min = float(np.min(pvals_universe))
        p_q95, p_q99 = np.quantile(pvals_universe, [0.95, 0.99]).tolist()
        p_max = float(np.max(pvals_universe))
        logger.info(
            "P-value stats in universe: p_min=%.6f p_95=%.6f p_99=%.6f p_max=%.6f",
            p_min,
            p_q95,
            p_q99,
            p_max,
        )
        if args.select_mode == "bh" and p_max <= args.q:
            logger.info("DIAG: pval_max <= q, so BH will select ALL in this universe.")

    qvals = None
    if args.select_mode == "bh":
        qvals = pd.Series(index=df.index, dtype=float)
        qvals.loc[universe_mask] = bh_qvalues(df.loc[universe_mask, "pval"].values)
        selected = (qvals <= args.q) & universe_mask
        df["qval"] = qvals
        threshold_used = args.q
    else:
        if args.pval_cutoff is None:
            raise ValueError("--select-mode=pval_cutoff requires --pval-cutoff.")
        selected = (df["pval"] <= args.pval_cutoff) & universe_mask
        threshold_used = args.pval_cutoff

    df["selected"] = selected

    cols = ["source"]
    smiles_col = _choose_smiles_col(df)
    if smiles_col:
        cols.append(smiles_col)
    cols += ["meta_score", "pval"]
    for extra_col in ["pval_weighted", "pval_unweighted"]:
        if extra_col in df.columns and extra_col != "pval":
            cols.append(extra_col)
    if args.select_mode == "bh":
        cols.append("qval")
    cols.append("selected")

    if "pu_label" in df.columns:
        cols.append("pu_label")

    out_df = df[cols].sort_values("meta_score", ascending=False)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "final_selected.csv", index=False)

    n_universe = int(universe_mask.sum())
    n_selected = int(selected.sum())
    report = {
        "mode": args.select_mode,
        "alpha": float(args.alpha),
        "q": float(args.q),
        "pval_column": pval_col,
        "select_over": args.select_over,
        "n_universe": n_universe,
        "n_selected": n_selected,
        "top_k": int(args.top_k),
        "pval_type": args.pval_type,
    }
    if "pu_label" in df.columns:
        n_pos_selected = int(((df["pu_label"] == 1) & selected).sum())
        report["n_pos_selected"] = n_pos_selected
    if args.top_k > 0 and universe_mask.sum() > 0:
        score_threshold = float(df.loc[universe_mask, "meta_score"].min())
        report["meta_score_threshold"] = score_threshold
        logger.info("Meta-score threshold at top-%d: %.6f", args.top_k, score_threshold)

    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Selection mode: %s", args.select_mode)
    logger.info("Selection threshold: %.6f", float(threshold_used))
    logger.info("Selected %d / %d rows", n_selected, n_universe)
    logger.info("Wrote outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
