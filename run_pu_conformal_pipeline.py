#!/usr/bin/env python3
"""
Orchestrator: run Steps A->E end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pu_stepA_reliable_negatives import main as stepA_main
from pu_stepB_train_meta import main as stepB_main
from pu_stepB_cwra_bypass import main as stepB_cwra_main
from pu_stepB2_build_calib_negatives import main as stepB2_main
from pu_stepC_conformal_pvalues import main as stepC_main
from pu_stepD_weighted_conformal import main as stepD_main
from pu_stepE_select import main as stepE_main

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _log_csv_shape(path: Path, label: str) -> None:
    if not path.exists():
        logger.warning("%s not found: %s", label, path)
        return
    df = pd.read_csv(path)
    logger.info("%s rows=%d cols=%d", label, len(df), df.shape[1])


def _log_value_counts(path: Path, col: str, label: str, top: int = 10) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if col not in df.columns:
        return
    counts = df[col].value_counts(dropna=False).head(top)
    logger.info("%s %s counts (top %d): %s", label, col, top, counts.to_dict())


def _log_top_k(path: Path, score_col: str, label: str, k: int = 5) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if score_col not in df.columns:
        return
    top_df = df.sort_values(score_col, ascending=False).head(k)
    cols = [c for c in ["source", "pu_label", score_col, "pval_weighted", "pval_unweighted"] if c in top_df.columns]
    logger.info("%s top-%d by %s:\n%s", label, k, score_col, top_df[cols].to_string(index=False))


def _apply_druglike_filter(
    input_csv: str,
    pu_labels_path: Path,
    max_mw: float,
    max_rotb: int,
    min_qed: float,
) -> dict:
    """
    Pre-filter non-drug-like unlabeled compounds by setting pu_label = -2.

    Only modifies unlabeled (pu_label == -1) compounds.
    Positives (1) and reliable negatives (0) are NEVER touched.

    Returns a report dict.
    """
    df_input = pd.read_csv(input_csv)
    labels_df = pd.read_csv(pu_labels_path)

    # Align input rows with labels
    if "index" in labels_df.columns:
        labels_df = labels_df.set_index("index")
        df_input = df_input.loc[labels_df.index]
    elif len(labels_df) == len(df_input):
        labels_df.index = df_input.index
    else:
        raise ValueError("Cannot align pu_labels with input CSV.")

    unlabeled_mask = labels_df["pu_label"] == -1
    n_unlabeled_before = int(unlabeled_mask.sum())

    # Build exclusion mask: True = fails drug-likeness
    exclude = pd.Series(False, index=labels_df.index)

    filters_applied = []

    if max_mw > 0 and "MW" in df_input.columns:
        mw = pd.to_numeric(df_input["MW"], errors="coerce")
        mw_fail = unlabeled_mask & (mw > max_mw)
        n_mw = int(mw_fail.sum())
        exclude = exclude | mw_fail
        filters_applied.append(f"MW>{max_mw:.0f}: {n_mw}")
        logger.info("Drug-likeness filter MW>%.0f: %d unlabeled excluded", max_mw, n_mw)
    elif max_mw > 0:
        logger.warning("MW column not found in input CSV; skipping MW filter.")

    if max_rotb > 0 and "RotB" in df_input.columns:
        rotb = pd.to_numeric(df_input["RotB"], errors="coerce")
        rotb_fail = unlabeled_mask & (rotb > max_rotb)
        n_rotb = int(rotb_fail.sum())
        exclude = exclude | rotb_fail
        filters_applied.append(f"RotB>{max_rotb}: {n_rotb}")
        logger.info("Drug-likeness filter RotB>%d: %d unlabeled excluded", max_rotb, n_rotb)
    elif max_rotb > 0:
        logger.warning("RotB column not found in input CSV; skipping RotB filter.")

    if min_qed > 0 and "QED" in df_input.columns:
        qed = pd.to_numeric(df_input["QED"], errors="coerce")
        qed_fail = unlabeled_mask & ((qed < min_qed) | qed.isna())
        n_qed = int(qed_fail.sum())
        exclude = exclude | qed_fail
        filters_applied.append(f"QED<{min_qed:.2f}: {n_qed}")
        logger.info("Drug-likeness filter QED<%.2f: %d unlabeled excluded", min_qed, n_qed)
    elif min_qed > 0:
        logger.warning("QED column not found in input CSV; skipping QED filter.")

    n_excluded = int(exclude.sum())
    n_unlabeled_after = n_unlabeled_before - n_excluded

    # Relabel excluded compounds: -1 → -2
    labels_df.loc[exclude, "pu_label"] = -2

    # Write back
    labels_df = labels_df.reset_index()
    labels_df.to_csv(pu_labels_path, index=False)

    logger.info(
        "Drug-likeness pre-filter: %d / %d unlabeled excluded (%.1f%%), %d remain",
        n_excluded,
        n_unlabeled_before,
        100 * n_excluded / max(n_unlabeled_before, 1),
        n_unlabeled_after,
    )

    return {
        "n_unlabeled_before": n_unlabeled_before,
        "n_excluded": n_excluded,
        "n_unlabeled_after": n_unlabeled_after,
        "max_mw": max_mw,
        "max_rotb": max_rotb,
        "min_qed": min_qed,
        "filters_applied": filters_applied,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run PU conformal pipeline (A->E).")
    parser.add_argument("--input", default="data/composed_modalities_with_rdkit.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-pos-ratio", type=int, default=10)
    parser.add_argument("--bottom-q", type=float, default=0.2)
    parser.add_argument("--sim-max", type=float, default=0.35)
    parser.add_argument("--calib-frac", type=float, default=0.3)
    parser.add_argument("--clip", type=float, default=20)
    parser.add_argument("--mode", choices=["alpha", "bh"], default="bh")
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--select-mode",vchoices=["bh", "pval_cutoff"], default="bh", help="Selection mode for Step E: bh or pval_cutoff.")
    parser.add_argument("--pval-cutoff", type=float, default=None)
    parser.add_argument("--calib-negatives", choices=["rn_only", "unlabeled_random"], default="rn_only", help="Calibration negatives source for Step C.")
    parser.add_argument("--calib-neg-n", type=int, default=0, help="Number of calibration negatives for unlabeled_random (0=auto).")
    parser.add_argument("--calib-neg-sim-max", type=float, default=None, help="Similarity cutoff for unlabeled calibration negatives (default: --sim-max).")
    parser.add_argument(
        "--score-source",
        choices=["cwra", "meta_lr", "meta_gbt"],
        default="cwra",
        help=(
            "Score source for conformal selection. "
            "'cwra': CWRA consensus scores (primary, recommended). "
            "'meta_lr': logistic regression meta-model. "
            "'meta_gbt': gradient boosting meta-model (ablation)."
        ),
    )
    parser.add_argument(
        "--cwra-weights-csv",
        default=None,
        help="Path to pre-computed CWRA weights CSV (optional).",
    )
    parser.add_argument(
        "--cwra-norm",
        choices=["minmax", "rank", "robust"],
        default="minmax",
        help="CWRA normalization method (must match CWRA CV configuration).",
    )
    parser.add_argument(
        "--top-k", type=int, default=1000, help="Pre-filter top-k for BH selection (0=no filter)"
    )
    parser.add_argument(
        "--pval-type",
        choices=["weighted", "unweighted", "auto"],
        default="unweighted",
        help="P-value type for Step E selection. Default 'unweighted' for best power.",
    )
    parser.add_argument(
        "--max-mw",
        type=float,
        default=600.0,
        help="Drug-likeness pre-filter: max molecular weight for unlabeled compounds. "
             "Set to 0 to disable. Default 800 (match CWRA CV if using cwra scoring).",
    )
    parser.add_argument(
        "--max-rotb",
        type=int,
        default=15,
        help="Drug-likeness pre-filter: max rotatable bonds for unlabeled compounds. "
             "Set to 0 to disable. Default 15.",
    )
    parser.add_argument(
        "--min-qed",
        type=float,
        default=0.0,
        help="Drug-likeness pre-filter: min QED score for unlabeled compounds. "
             "Set to 0 to disable. Default 0 (off).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dir_a = outdir / "A"
    dir_b = outdir / "B"
    dir_c = outdir / "C"
    dir_d = outdir / "D"
    dir_e = outdir / "E"

    for d in [dir_a, dir_b, dir_c, dir_d, dir_e]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Input: %s", args.input)
    logger.info("Output root: %s", outdir)
    logger.info(
        "Params: seed=%d neg_pos_ratio=%d bottom_q=%.3f sim_max=%.3f "
        "calib_frac=%.3f clip=%.1f mode=%s q=%.3f alpha=%.3f "
        "select_mode=%s pval_cutoff=%s calib_negatives=%s calib_neg_n=%d calib_neg_sim_max=%s "
        "score_source=%s cwra_weights_csv=%s cwra_norm=%s top_k=%d pval_type=%s "
        "max_mw=%.0f max_rotb=%d min_qed=%.2f",
        args.seed,
        args.neg_pos_ratio,
        args.bottom_q,
        args.sim_max,
        args.calib_frac,
        args.clip,
        args.mode,
        args.q,
        args.alpha,
        args.select_mode,
        str(args.pval_cutoff),
        args.calib_negatives,
        args.calib_neg_n,
        str(args.calib_neg_sim_max),
        args.score_source,
        str(args.cwra_weights_csv),
        args.cwra_norm,
        args.top_k,
        args.pval_type,
        args.max_mw,
        args.max_rotb,
        args.min_qed,
    )

    try:
        logger.info("=== Step A: Reliable negatives ===")
        step_a_args = [
            "--input",
            args.input,
            "--output",
            str(dir_a),
            "--neg-pos-ratio",
            str(args.neg_pos_ratio),
            "--bottom-q",
            str(args.bottom_q),
            "--sim-max",
            str(args.sim_max),
            "--seed",
            str(args.seed),
            "--include-boltz-confidence",
            "--include-unimol",
        ]
        stepA_main(step_a_args)
    except Exception as exc:
        logger.error("Step A failed: %s", exc)
        return 1
    report_a = _load_json(dir_a / "pu_stepA_report.json")
    if report_a:
        logger.info("Step A report: %s", report_a)
    _log_csv_shape(dir_a / "pu_labels.csv", "Step A pu_labels")
    _log_value_counts(dir_a / "pu_labels.csv", "pu_label", "Step A", top=5)

    # === Drug-likeness pre-filter ===
    dl_report = None
    any_filter = (args.max_mw > 0) or (args.max_rotb > 0) or (args.min_qed > 0)
    if any_filter:
        try:
            logger.info("=== Drug-likeness pre-filter ===")
            dl_report = _apply_druglike_filter(
                input_csv=args.input,
                pu_labels_path=dir_a / "pu_labels.csv",
                max_mw=args.max_mw,
                max_rotb=args.max_rotb,
                min_qed=args.min_qed,
            )
            _log_value_counts(dir_a / "pu_labels.csv", "pu_label", "Post-filter", top=5)
        except Exception as exc:
            logger.error("Drug-likeness filter failed: %s", exc)
            return 1
    else:
        logger.info("Drug-likeness pre-filter: disabled (all thresholds = 0)")

    try:
        logger.info("=== Step B: Scoring (%s) ===", args.score_source)
        if args.score_source == "cwra":
            cwra_args = [
                "--input",
                args.input,
                "--pu-labels",
                str(dir_a / "pu_labels.csv"),
                "--output",
                str(dir_b),
                "--method",
                "fair",
                "--seed",
                str(args.seed),
                "--norm",
                args.cwra_norm,
            ]
            if args.cwra_weights_csv:
                cwra_args += ["--weights-csv", args.cwra_weights_csv]
            stepB_cwra_main(cwra_args)
        else:
            model_type = "gbt" if args.score_source == "meta_gbt" else "lr"
            stepB_main(
                [
                    "--input",
                    args.input,
                    "--pu-labels",
                    str(dir_a / "pu_labels.csv"),
                    "--output",
                    str(dir_b),
                    "--seed",
                    str(args.seed),
                    "--calib-frac",
                    str(args.calib_frac),
                    "--model-type",
                    model_type,
                ]
            )
    except Exception as exc:
        logger.error("Step B failed: %s", exc)
        return 1

    calib_indices_path = None
    if args.calib_negatives == "unlabeled_random":
        dir_b2 = outdir / "B2"
        dir_b2.mkdir(parents=True, exist_ok=True)
        n_rn = int(report_a.get("n_rn", 2690)) if report_a else 2690
        n_cal = args.calib_neg_n if args.calib_neg_n > 0 else n_rn
        sim_max = args.calib_neg_sim_max if args.calib_neg_sim_max is not None else args.sim_max
        try:
            stepB2_main(
                [
                    "--input",
                    args.input,
                    "--pu-labels",
                    str(dir_a / "pu_labels.csv"),
                    "--meta-scores",
                    str(dir_b / "meta_scores.csv"),
                    "--output",
                    str(dir_b2),
                    "--n-cal",
                    str(n_cal),
                    "--seed",
                    str(args.seed),
                    "--sim-max",
                    str(sim_max),
                ]
            )
        except Exception as exc:
            logger.error("Step B2 failed: %s", exc)
            return 1
        calib_indices_path = dir_b2 / "calib_negatives.csv"

    _log_csv_shape(dir_b / "meta_scores.csv", "Step B meta_scores")
    _log_value_counts(dir_b / "meta_scores.csv", "pu_label", "Step B", top=5)
    _log_top_k(dir_b / "meta_scores.csv", "meta_score", "Step B", k=5)
    # Score distribution diagnostic
    _meta_path = dir_b / "meta_scores.csv"
    if _meta_path.exists():
        _m = pd.read_csv(_meta_path)
        for _label, _name in [(1, "positives"), (0, "RN"), (-1, "unlabeled")]:
            _subset = _m.loc[_m["pu_label"] == _label, "meta_score"]
            if len(_subset) > 0:
                logger.info(
                    "Score dist (%s, %s): median=%.4f mean=%.4f std=%.4f min=%.4f max=%.4f",
                    args.score_source,
                    _name,
                    _subset.median(),
                    _subset.mean(),
                    _subset.std(),
                    _subset.min(),
                    _subset.max(),
                )
        _rn_scores = _m.loc[_m["pu_label"] == 0, "meta_score"]
        if len(_rn_scores) > 0:
            _rn_p99 = float(_rn_scores.quantile(0.99))
            logger.info("RN 99th percentile meta_score: %.6f", _rn_p99)
        if len(_m) >= 1000:
            _top1000_thresh = float(_m["meta_score"].nlargest(1000).min())
            logger.info("Meta_score at rank 1000: %.6f", _top1000_thresh)
        calib_neg_scores = None
        if calib_indices_path is not None and calib_indices_path.exists():
            try:
                _cneg = pd.read_csv(calib_indices_path)
                if "index" in _cneg.columns and "index" in _m.columns:
                    _m_idx = _m.set_index("index", drop=False)
                    calib_neg_scores = _m_idx.reindex(_cneg["index"])["meta_score"]
                    if calib_neg_scores.isna().any():
                        logger.warning("Calibration negatives not fully aligned to meta_scores.")
                else:
                    logger.warning("Calibration negatives require index column in meta_scores.")
            except Exception as exc:
                logger.warning("Failed to read calibration negatives for diagnostics: %s", exc)

        if calib_neg_scores is not None and len(calib_neg_scores) > 0:
            logger.info(
                "Score dist (%s, calib_neg): median=%.4f mean=%.4f std=%.4f min=%.4f max=%.4f",
                args.score_source,
                calib_neg_scores.median(),
                calib_neg_scores.mean(),
                calib_neg_scores.std(),
                calib_neg_scores.min(),
                calib_neg_scores.max(),
            )

        _unl_med = _m.loc[_m["pu_label"] == -1, "meta_score"].median()
        if calib_neg_scores is not None and len(calib_neg_scores) > 0:
            _calib_max = calib_neg_scores.max()
            if _unl_med <= _calib_max:
                logger.info(
                    "Score separation OK: unlabeled median (%.4f) <= calib_neg max (%.4f) — "
                    "conformal selection will be discriminative.",
                    _unl_med,
                    _calib_max,
                )
            else:
                logger.warning(
                    "POOR SEPARATION: unlabeled median (%.4f) > calib_neg max (%.4f) — "
                    "most unlabeled will get minimum p-value, limiting discrimination.",
                    _unl_med,
                    _calib_max,
                )
        else:
            # Check score separation: is there a gap between RN and unlabeled?
            _rn_max = _m.loc[_m["pu_label"] == 0, "meta_score"].max()
            if _unl_med <= _rn_max:
                logger.info(
                    "Score separation OK: unlabeled median (%.4f) <= RN max (%.4f) — "
                    "conformal selection will be discriminative.",
                    _unl_med,
                    _rn_max,
                )
            else:
                logger.warning(
                    "POOR SEPARATION: unlabeled median (%.4f) > RN max (%.4f) — "
                    "most unlabeled will get minimum p-value, limiting discrimination.",
                    _unl_med,
                    _rn_max,
                )

    try:
        logger.info("=== Step C: Unweighted conformal ===")
        step_c_args = [
            "--meta-scores",
            str(dir_b / "meta_scores.csv"),
            "--output",
            str(dir_c),
            "--calib-frac",
            str(args.calib_frac),
            "--seed",
            str(args.seed),
        ]
        if calib_indices_path is not None:
            step_c_args += ["--calib-indices", str(calib_indices_path)]
        else:
            step_c_args += ["--calib-set", "rn_only"]
        stepC_main(step_c_args)
    except Exception as exc:
        logger.error("Step C failed: %s", exc)
        return 1
    _log_csv_shape(dir_c / "conformal_pvalues.csv", "Step C conformal_pvalues")
    # Sanity check: positives should have SMALLER p-values than negatives
    _conf_path = dir_c / "conformal_pvalues.csv"
    if _conf_path.exists():
        _c = pd.read_csv(_conf_path)
        if "pval_unweighted" in _c.columns and "pu_label" in _c.columns:
            _pos_med = _c.loc[_c["pu_label"] == 1, "pval_unweighted"].median()
            _neg_med = _c.loc[_c["pu_label"] == 0, "pval_unweighted"].median()
            _unl_med = _c.loc[_c["pu_label"] == -1, "pval_unweighted"].median()
            logger.info(
                "P-value sanity check (unweighted): pos_median=%.4f neg_median=%.4f unlabeled_median=%.4f",
                _pos_med,
                _neg_med,
                _unl_med,
            )
            if _pos_med >= _neg_med:
                logger.warning(
                    "SANITY CHECK FAILED: positive median p-value >= negative median. "
                    "P-value direction may be wrong."
                )

    try:
        logger.info("=== Step D: Weighted conformal ===")
        stepD_main(
            [
                "--input",
                args.input,
                "--pu-labels",
                str(dir_a / "pu_labels.csv"),
                "--meta-scores",
                str(dir_b / "meta_scores.csv"),
                "--conformal",
                str(dir_c / "conformal_pvalues.csv"),
                "--output",
                str(dir_d),
                "--calib-frac",
                str(args.calib_frac),
                "--calib-set",
                "rn_only",
                "--seed",
                str(args.seed),
                "--clip",
                str(args.clip),
                "--target",
                "unlabeled",
            ]
        )
    except Exception as exc:
        logger.error("Step D failed: %s", exc)
        return 1
    _log_csv_shape(dir_d / "weighted_pvalues.csv", "Step D weighted_pvalues")
    _log_top_k(dir_d / "weighted_pvalues.csv", "meta_score", "Step D", k=5)
    _wp_path = dir_d / "weighted_pvalues.csv"
    if _wp_path.exists():
        _w = pd.read_csv(_wp_path)
        if all(c in _w.columns for c in ["pval_weighted", "pval_unweighted", "pu_label"]):
            _pos = _w[_w["pu_label"] == 1]
            _unl = _w[_w["pu_label"] == -1]
            logger.info(
                "Weighted vs unweighted (positives): weighted_med=%.4f unweighted_med=%.4f",
                _pos["pval_weighted"].median(),
                _pos["pval_unweighted"].median(),
            )
            logger.info(
                "Weighted vs unweighted (unlabeled): weighted_med=%.4f unweighted_med=%.4f",
                _unl["pval_weighted"].median(),
                _unl["pval_unweighted"].median(),
            )
            # Effective weight diagnostic
            if "weight" in _w.columns:
                w = _w["weight"].values
                n_eff = (w.sum() ** 2) / (w**2).sum()
                logger.info(
                    "Importance weights: n_eff=%.1f / n=%d (ratio=%.2f)",
                    n_eff,
                    len(w),
                    n_eff / len(w),
                )

    try:
        logger.info("=== Step E: Final selection ===")
        step_e_args = [
            "--input",
            args.input,
            "--pvalues",
            str(dir_d / "weighted_pvalues.csv"),
            "--output",
            str(dir_e),
            "--mode",
            args.mode,
            "--q",
            str(args.q),
            "--alpha",
            str(args.alpha),
            "--select-over",
            "unlabeled",
            "--top-k",
            str(args.top_k),
            "--pval-type",
            args.pval_type,
            "--select-mode",
            args.select_mode,
        ]
        if args.pval_cutoff is not None:
            step_e_args += ["--pval-cutoff", str(args.pval_cutoff)]
        stepE_main(step_e_args)
    except Exception as exc:
        logger.error("Step E failed: %s", exc)
        return 1
    report_e = _load_json(dir_e / "selection_summary.json")
    if report_e:
        logger.info("Step E report: %s", report_e)
    _log_csv_shape(dir_e / "final_selected.csv", "Step E final_selected")
    _log_value_counts(dir_e / "final_selected.csv", "selected", "Step E", top=2)
    _log_value_counts(dir_e / "final_selected.csv", "source", "Step E selected-by-source", top=10)

    # Pipeline summary
    if dl_report is not None and report_e:
        logger.info(
            "Pipeline summary: %d unlabeled → %d after drug-likeness filter → "
            "%d in selection universe → %d selected",
            dl_report["n_unlabeled_before"],
            dl_report["n_unlabeled_after"],
            report_e.get("n_universe", "?"),
            report_e.get("n_selected", "?"),
        )

    # Combined report
    rows = []
    for step, report_path in [
        ("A", dir_a / "pu_stepA_report.json"),
        ("E", dir_e / "selection_summary.json"),
    ]:
        report = _load_json(report_path)
        for k, v in report.items():
            rows.append({"step": step, "key": k, "value": v})

    if dl_report is not None:
        for k, v in dl_report.items():
            rows.append({"step": "DL_filter", "key": k, "value": str(v)})

    report_df = pd.DataFrame(rows)
    report_df.to_csv(outdir / "combined_report.csv", index=False)
    logger.info("Wrote combined report to %s", outdir / "combined_report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
