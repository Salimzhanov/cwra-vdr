#!/usr/bin/env python3
"""
Generate LaTeX tables from CWRA CV result CSVs.

Example:
  python scripts/cwra_cv_to_latex.py --input-dir cwra_cv_results_cv5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[warn] missing file: {path}")
        return None
    return pd.read_csv(path)


def _latex(df: pd.DataFrame, caption: Optional[str] = None, label: Optional[str] = None) -> str:
    return df.to_latex(index=False, escape=False, caption=caption, label=label)


def _write_tex(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def build_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    df["Train EF"] = df.apply(lambda r: _pm(r.train_ef_mean, r.train_ef_std), axis=1)
    df["Test EF"] = df.apply(lambda r: _pm(r.test_ef_mean, r.test_ef_std), axis=1)
    df["Full EF"] = df.apply(lambda r: _pm(r.full_ef_mean, r.full_ef_std), axis=1)
    df["Train hits"] = df["train_hits_mean"].map(lambda v: f"{v:.1f}")
    df["Test hits"] = df["test_hits_mean"].map(lambda v: f"{v:.1f}")
    df["Full hits"] = df["full_hits_mean"].map(lambda v: f"{v:.1f}")
    out = df[["cutoff_pct", "Train EF", "Test EF", "Full EF", "Train hits", "Test hits", "Full hits"]]
    out = out.rename(columns={"cutoff_pct": "Cutoff (%)"})
    return out


def build_baseline_table(baseline_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["test_ef_1", "test_ef_5", "test_ef_10"]
    g = baseline_df.groupby("baseline", as_index=False)
    rows = []
    for _, row in g:
        entry = {"Baseline": row["baseline"].iloc[0]}
        for c in cols:
            entry[c] = _pm(row[c].mean(), row[c].std())
        rows.append(entry)
    out = pd.DataFrame(rows)
    out = out.rename(
        columns={
            "test_ef_1": "Test EF@1%",
            "test_ef_5": "Test EF@5%",
            "test_ef_10": "Test EF@10%",
        }
    )
    return out


def build_weights_table(weights_df: pd.DataFrame, top_k: Optional[int]) -> pd.DataFrame:
    g = weights_df.groupby(["modality", "column"], as_index=False)
    agg = g["weight_pct"].agg(["mean", "std"]).reset_index()
    agg = agg.rename(columns={"mean": "weight_pct_mean", "std": "weight_pct_std"})
    agg["Weight (%)"] = agg.apply(lambda r: _pm(r.weight_pct_mean, r.weight_pct_std), axis=1)
    out = agg.sort_values("weight_pct_mean", ascending=False)[["modality", "column", "Weight (%)"]]
    out = out.rename(columns={"modality": "Modality", "column": "Column"})
    if top_k is not None:
        out = out.head(top_k)
    return out


def build_fold_info_table(info_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["fold", "n_actives_train", "n_actives_test", "n_total"]
    out = info_df[cols].copy()
    out = out.rename(
        columns={
            "fold": "Fold",
            "n_actives_train": "Train actives",
            "n_actives_test": "Test actives",
            "n_total": "Total compounds",
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CWRA CV CSV outputs as LaTeX tables.")
    parser.add_argument("--input-dir", default="cwra_cv_results_cv5", help="Directory with CSV outputs")
    parser.add_argument("--output-dir", default=None, help="Directory to write .tex files (default: input dir)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k modalities in weights table")
    args = parser.parse_args()

    base = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _read_csv(base / "cwra_cv_cv_summary.csv")
    baseline_df = _read_csv(base / "cwra_cv_cv_folds_baselines.csv")
    weights_df = _read_csv(base / "cwra_cv_cv_folds_weights.csv")
    info_df = _read_csv(base / "cwra_cv_cv_folds_info.csv")

    if summary_df is not None:
        print("\n% CV Summary")
        summary_tex = _latex(build_summary_table(summary_df), caption="CWRA CV Summary", label="tab:cwra_cv_summary")
        print(summary_tex)
        _write_tex(out_dir / "cwra_cv_summary.tex", summary_tex)

    if baseline_df is not None:
        print("\n% Baselines (test set)")
        baseline_tex = _latex(
            build_baseline_table(baseline_df),
            caption="Baseline Test-Set Performance (CV)",
            label="tab:cwra_cv_baselines",
        )
        print(baseline_tex)
        _write_tex(out_dir / "cwra_cv_baselines.tex", baseline_tex)

    if weights_df is not None:
        print("\n% Weights")
        weights_tex = _latex(
            build_weights_table(weights_df, args.top_k),
            caption="CWRA Weights Across Folds",
            label="tab:cwra_cv_weights",
        )
        print(weights_tex)
        _write_tex(out_dir / "cwra_cv_weights.tex", weights_tex)

    if info_df is not None:
        print("\n% Fold Info")
        info_tex = _latex(
            build_fold_info_table(info_df),
            caption="Cross-Validation Fold Composition",
            label="tab:cwra_cv_folds",
        )
        print(info_tex)
        _write_tex(out_dir / "cwra_cv_folds.tex", info_tex)


if __name__ == "__main__":
    main()
