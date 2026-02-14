#!/usr/bin/env python3
"""
Analyze CWRA ablation results and produce paper-ready tables.

Usage:
    python analyze_ablations.py                          # analyze all
    python analyze_ablations.py --results-root results   # custom root
    python analyze_ablations.py --format csv             # CSV instead of LaTeX

Reads the ablation_manifest.json and per-variant output CSVs,
produces comparison tables for the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# Data loading
# ============================================================================

def load_manifest(results_root: Path) -> list[dict]:
    """Load the ablation manifest."""
    path = results_root / "ablation_manifest.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_ablations.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_variant_data(results_root: Path, variant: dict) -> dict:
    """Load all CSV outputs for one variant. Returns dict of DataFrames."""
    cwra_dir = results_root / variant["folder"] / "cwra"
    pipe_dir = results_root / variant["folder"] / "pipeline"

    data = {"tag": variant["tag"], "label": variant["label"],
            "tier": variant["tier"], "folder": variant["folder"]}

    # --- CWRA CV outputs ---
    files = {
        "paper_table":     cwra_dir / "cwra_cv_paper_table.csv",
        "summary":         cwra_dir / "cwra_cv_summary.csv",
        "mean_weights":    cwra_dir / "cwra_cv_mean_weights.csv",
        "folds_perf":      cwra_dir / "cwra_cv_folds_performance.csv",
        "folds_weights":   cwra_dir / "cwra_cv_folds_weights.csv",
        "folds_methods":   cwra_dir / "cwra_cv_folds_methods.csv",
        "folds_baselines": cwra_dir / "cwra_cv_folds_baselines.csv",
        "folds_individual": cwra_dir / "cwra_cv_folds_individual.csv",
        "extra_metrics":   cwra_dir / "cwra_cv_folds_extra_metrics.csv",
        "filter_report":   cwra_dir / "cwra_filter_report.csv",
    }
    for key, path in files.items():
        if path.exists():
            try:
                data[key] = pd.read_csv(path)
            except Exception as e:
                print(f"  WARNING: Could not read {path}: {e}")
                data[key] = None
        else:
            data[key] = None

    # --- Pipeline outputs ---
    pipe_report = pipe_dir / "combined_report.csv"
    pipe_selection = pipe_dir / "E" / "selection_summary.json"
    pipe_final = pipe_dir / "E" / "final_selected.csv"

    if pipe_report.exists():
        try:
            data["pipe_report"] = pd.read_csv(pipe_report)
        except Exception:
            data["pipe_report"] = None
    else:
        data["pipe_report"] = None

    if pipe_selection.exists():
        try:
            with open(pipe_selection) as f:
                data["pipe_selection"] = json.load(f)
        except Exception:
            data["pipe_selection"] = None
    else:
        data["pipe_selection"] = None

    if pipe_final.exists():
        try:
            df = pd.read_csv(pipe_final)
            data["pipe_n_selected"] = int(df["selected"].sum())
            data["pipe_n_total"] = len(df)
            # Source breakdown of selected
            sel = df[df["selected"] == True]
            data["pipe_source_counts"] = sel["source"].value_counts().to_dict() if len(sel) > 0 else {}
        except Exception:
            pass

    return data


def extract_cwra_metrics(data: dict) -> dict:
    """Extract key CWRA metrics from a variant's data."""
    row = {
        "tag": data["tag"],
        "label": data["label"],
        "tier": data["tier"],
    }

    # --- Test EF from paper_table (CWRA row) ---
    pt = data.get("paper_table")
    if pt is not None:
        cwra_row = pt[pt["kind"] == "cwra"]
        if not cwra_row.empty:
            cwra_row = cwra_row.iloc[0]
            for c in [0.5, 1, 5, 10, 20]:
                col = f"ef_{c}" if c != 0.5 else "ef_0.5"
                std_col = f"{col}_std"
                if col in cwra_row.index:
                    row[f"test_ef_{c}"] = cwra_row[col]
                if std_col in cwra_row.index:
                    row[f"test_ef_{c}_std"] = cwra_row[std_col]
            for metric in ["auroc", "auprc", "bedroc"]:
                if metric in cwra_row.index:
                    row[f"test_{metric}"] = cwra_row[metric]
                std_m = f"{metric}_std"
                if std_m in cwra_row.index:
                    row[f"test_{metric}_std"] = cwra_row[std_m]

    # --- Train EF (from summary) ---
    summary = data.get("summary")
    if summary is not None:
        for _, srow in summary.iterrows():
            c = srow["cutoff_pct"]
            row[f"train_ef_{c}"] = srow.get("train_ef_mean", np.nan)
            row[f"train_ef_{c}_std"] = srow.get("train_ef_std", np.nan)

    # --- Extra metrics from folds ---
    extra = data.get("extra_metrics")
    if extra is not None and not extra.empty:
        # Find the chosen method (should be first or "fair" variant)
        methods = extra["method"].unique()
        chosen = None
        for candidate in ["fair_bedroc", "fair"]:
            if candidate in methods:
                chosen = candidate
                break
        if chosen is None:
            chosen = methods[0]

        sel = extra[extra["method"] == chosen]
        if not sel.empty:
            for metric in ["test_auroc", "test_auprc", "test_bedroc"]:
                if metric in sel.columns:
                    row[f"{metric}"] = sel[metric].mean()
                    row[f"{metric}_std"] = sel[metric].std(ddof=0)

    # --- Weights summary ---
    mw = data.get("mean_weights")
    if mw is not None and not mw.empty:
        # Effective number of modalities (Herfindahl)
        w = mw["weight"].values
        hhi = (w ** 2).sum()
        row["n_eff"] = 1.0 / hhi if hhi > 0 else 0
        row["n_modalities"] = len(w)
        row["n_significant"] = int((w > 0.05).sum())
        # Shannon entropy (normalized)
        w_pos = w[w > 0]
        h = -np.sum(w_pos * np.log(w_pos))
        h_max = np.log(len(w_pos))
        row["entropy_norm"] = h / h_max if h_max > 0 else 0

    # --- Overfitting gap ---
    if f"train_ef_1" in row and "test_ef_1" in row:
        train = row["train_ef_1"]
        test = row["test_ef_1"]
        row["overfit_gap_ef1"] = train - test
        row["overfit_gap_ef1_pct"] = 100 * (train - test) / train if train > 0 else 0

    # --- Pipeline ---
    pipe_sel = data.get("pipe_selection")
    if pipe_sel is not None:
        row["pipe_n_selected"] = pipe_sel.get("n_selected", np.nan)
        row["pipe_n_universe"] = pipe_sel.get("n_universe", np.nan)
        row["pipe_score_threshold"] = pipe_sel.get("meta_score_threshold", np.nan)

    if "pipe_n_selected" in data:
        row["pipe_n_selected"] = data["pipe_n_selected"]
    if "pipe_source_counts" in data:
        row["pipe_sources"] = str(data.get("pipe_source_counts", {}))

    return row


# ============================================================================
# Table builders
# ============================================================================

def _fmt(val, std=None, precision=2):
    """Format value ± std."""
    if pd.isna(val):
        return "—"
    s = f"{val:.{precision}f}"
    if std is not None and not pd.isna(std):
        s += f" ± {std:.{precision}f}"
    return s


def _bold_best(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Mark the best value in a series for LaTeX bolding."""
    if series.isna().all():
        return series.astype(str)
    if higher_is_better:
        best_idx = series.idxmax()
    else:
        best_idx = series.idxmin()
    result = series.apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "—")
    if not pd.isna(series[best_idx]):
        result[best_idx] = f"\\textbf{{{series[best_idx]:.2f}}}"
    return result


def build_ablation_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build the main ablation comparison table."""
    cols_display = []
    for _, r in metrics_df.iterrows():
        cols_display.append({
            "Variant": r["tag"],
            "Description": r["label"],
            "EF@0.5%": _fmt(r.get("test_ef_0.5"), r.get("test_ef_0.5_std")),
            "EF@1%": _fmt(r.get("test_ef_1"), r.get("test_ef_1_std")),
            "EF@5%": _fmt(r.get("test_ef_5"), r.get("test_ef_5_std")),
            "EF@10%": _fmt(r.get("test_ef_10"), r.get("test_ef_10_std")),
            "AUROC": _fmt(r.get("test_auroc"), r.get("test_auroc_std"), 3),
            "BEDROC": _fmt(r.get("test_bedroc"), r.get("test_bedroc_std"), 3),
            "N_eff": f"{r.get('n_eff', 0):.1f}" if not pd.isna(r.get("n_eff")) else "—",
            "Gap%": f"{r.get('overfit_gap_ef1_pct', 0):.1f}" if not pd.isna(r.get("overfit_gap_ef1_pct")) else "—",
        })
    return pd.DataFrame(cols_display)


def build_ablation_latex(metrics_df: pd.DataFrame) -> str:
    """Generate LaTeX for the ablation table."""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: effect of individual design choices on CWRA cross-validated enrichment.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll rrrr rr}")
    lines.append(r"\toprule")
    lines.append(r"ID & Description & EF@0.5\% & EF@1\% & EF@5\% & EF@10\% & BEDROC$_{20}$ & Gap\% \\")
    lines.append(r"\midrule")

    current_tier = None
    for _, r in metrics_df.iterrows():
        tier = r.get("tier", "")
        if tier != current_tier and current_tier is not None:
            lines.append(r"\addlinespace")
        current_tier = tier

        tag = r["tag"]
        label = r["label"].replace("_", r"\_").replace("&", r"\&")
        # Truncate long labels
        if len(label) > 45:
            label = label[:42] + "..."

        ef05 = _fmt(r.get("test_ef_0.5"))
        ef1 = _fmt(r.get("test_ef_1"))
        ef5 = _fmt(r.get("test_ef_5"))
        ef10 = _fmt(r.get("test_ef_10"))
        bedroc = _fmt(r.get("test_bedroc"), precision=3)
        gap = _fmt(r.get("overfit_gap_ef1_pct"), precision=1)

        # Bold the reference row
        if tag == "V00":
            tag = r"\textbf{V00}"
            label = r"\textbf{" + label + "}"

        lines.append(f"  {tag} & {label} & {ef05} & {ef1} & {ef5} & {ef10} & {bedroc} & {gap} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_weights_table(all_data: list[dict]) -> pd.DataFrame:
    """Compare weight distributions across variants."""
    rows = []
    for data in all_data:
        mw = data.get("mean_weights")
        if mw is None or mw.empty:
            continue
        row = {"tag": data["tag"], "label": data["label"]}
        for _, w in mw.iterrows():
            row[w["modality"]] = w["weight"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_weights_latex(wt_df: pd.DataFrame) -> str:
    """Generate LaTeX for the weight comparison table."""
    # Only show modalities with >5% weight in at least one variant
    mod_cols = [c for c in wt_df.columns if c not in ("tag", "label")]
    significant = [c for c in mod_cols if wt_df[c].max() > 0.05]
    n_other = len(mod_cols) - len(significant)

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Optimized CWRA weights across ablation variants (modalities with $>$5\% weight in any variant shown; remaining " + str(n_other) + r" modalities at floor).}")
    lines.append(r"\label{tab:weights}")
    lines.append(r"\small")

    header = "l l" + " r" * len(significant)
    lines.append(r"\begin{tabular}{" + header + "}")
    lines.append(r"\toprule")

    # Header row with modality names (abbreviated)
    abbrev = {
        "UniMol_sim": "UniMol",
        "DrugBAN": "DrugBAN",
        "Vina": "Vina",
        "Boltz_confidence": "Boltz\\_c",
        "Boltz_affinity": "Boltz\\_a",
        "MLTLE_pKd": "MLTLE",
        "TankBind": "TankB",
        "MolTrans": "MolTr",
        "GraphDTA_Kd": "GDT\\_Kd",
        "GraphDTA_Ki": "GDT\\_Ki",
        "GraphDTA_IC50": "GDT\\_IC50",
    }
    header_names = " & ".join([abbrev.get(c, c) for c in significant])
    lines.append(f"  ID & Description & {header_names} \\\\")
    lines.append(r"\midrule")

    for _, r in wt_df.iterrows():
        tag = r["tag"]
        label = r["label"][:35].replace("_", r"\_")
        vals = []
        for c in significant:
            v = r.get(c, 0)
            if pd.isna(v):
                vals.append("—")
            elif v > 0.15:
                vals.append(f"\\textbf{{{v:.1%}}}")
            elif v > 0.05:
                vals.append(f"{v:.1%}")
            else:
                vals.append(f"\\textcolor{{gray}}{{{v:.1%}}}")
        lines.append(f"  {tag} & {label} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_pipeline_table(metrics_df: pd.DataFrame) -> str:
    """Generate LaTeX for pipeline comparison."""
    pipe_rows = metrics_df.dropna(subset=["pipe_n_selected"])
    if pipe_rows.empty:
        return "% No pipeline results found."

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{PU conformal pipeline selection summary across variants.}")
    lines.append(r"\label{tab:pipeline}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll rrr}")
    lines.append(r"\toprule")
    lines.append(r"ID & Description & Universe & Selected & Score threshold \\")
    lines.append(r"\midrule")

    for _, r in pipe_rows.iterrows():
        tag = r["tag"]
        label = r["label"][:35].replace("_", r"\_")
        n_univ = f"{int(r.get('pipe_n_universe', 0)):,}" if not pd.isna(r.get("pipe_n_universe")) else "—"
        n_sel = f"{int(r.get('pipe_n_selected', 0)):,}" if not pd.isna(r.get("pipe_n_selected")) else "—"
        thresh = f"{r.get('pipe_score_threshold', 0):.3f}" if not pd.isna(r.get("pipe_score_threshold")) else "—"
        lines.append(f"  {tag} & {label} & {n_univ} & {n_sel} & {thresh} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_delta_table(metrics_df: pd.DataFrame) -> str:
    """Generate a compact delta table: change in EF@1% relative to V00."""
    ref = metrics_df[metrics_df["tag"] == "V00"]
    if ref.empty:
        return "% V00 (reference) not found."
    ref_ef1 = ref.iloc[0].get("test_ef_1", np.nan)
    if pd.isna(ref_ef1):
        return "% V00 has no test_ef_1."

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of each ablation on test-set EF@1\% relative to the reference configuration (V00).}")
    lines.append(r"\label{tab:delta}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll rrr}")
    lines.append(r"\toprule")
    lines.append(r"ID & Change from V00 & EF@1\% & $\Delta$ & $\Delta$\% \\")
    lines.append(r"\midrule")

    current_tier = None
    for _, r in metrics_df.iterrows():
        tier = r.get("tier", "")
        if tier != current_tier and current_tier is not None:
            lines.append(r"\addlinespace")
        current_tier = tier

        tag = r["tag"]
        label = r["label"][:40].replace("_", r"\_")
        ef1 = r.get("test_ef_1", np.nan)
        if pd.isna(ef1):
            lines.append(f"  {tag} & {label} & — & — & — \\\\")
            continue

        delta = ef1 - ref_ef1
        delta_pct = 100 * delta / ref_ef1 if ref_ef1 != 0 else 0

        # Color coding: green for improvement, red for degradation
        if tag == "V00":
            delta_str = "—"
            pct_str = "(ref)"
        elif delta > 0.5:
            delta_str = f"\\textcolor{{teal}}{{+{delta:.2f}}}"
            pct_str = f"\\textcolor{{teal}}{{+{delta_pct:.1f}\\%}}"
        elif delta < -0.5:
            delta_str = f"\\textcolor{{red}}{{{delta:.2f}}}"
            pct_str = f"\\textcolor{{red}}{{{delta_pct:.1f}\\%}}"
        else:
            delta_str = f"{delta:+.2f}"
            pct_str = f"{delta_pct:+.1f}\\%"

        ef1_str = f"{ef1:.2f}"
        lines.append(f"  {tag} & {label} & {ef1_str} & {delta_str} & {pct_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze CWRA ablation results.")
    parser.add_argument("--results-root", type=str, default="results",
                        help="Root directory for results")
    parser.add_argument("--format", choices=["latex", "csv", "both"], default="both",
                        help="Output format")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <results-root>/_analysis)")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output) if args.output else results_root / "_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = load_manifest(results_root)
    print(f"Loaded manifest with {len(manifest)} variants")

    # Load all variant data
    all_data = []
    for v in manifest:
        print(f"  Loading {v['tag']}: {v['folder']} ...", end=" ")
        data = load_variant_data(results_root, v)
        has_cv = data.get("paper_table") is not None
        has_pipe = data.get("pipe_selection") is not None
        print(f"CV={'OK' if has_cv else 'MISSING'}  Pipeline={'OK' if has_pipe else 'skip'}")
        all_data.append(data)

    # Extract metrics
    metrics_rows = [extract_cwra_metrics(d) for d in all_data]
    metrics_df = pd.DataFrame(metrics_rows)

    # ---- Table 1: Full ablation comparison ----
    print(f"\n{'='*60}")
    print("  Table 1: Ablation comparison (test-set metrics)")
    print(f"{'='*60}")
    display_df = build_ablation_table(metrics_df)
    print(display_df.to_string(index=False))

    if args.format in ("csv", "both"):
        metrics_df.to_csv(output_dir / "ablation_metrics.csv", index=False)
        display_df.to_csv(output_dir / "ablation_table.csv", index=False)
        print(f"  → {output_dir / 'ablation_metrics.csv'}")

    if args.format in ("latex", "both"):
        tex = build_ablation_latex(metrics_df)
        with open(output_dir / "ablation_table.tex", "w") as f:
            f.write(tex)
        print(f"  → {output_dir / 'ablation_table.tex'}")

    # ---- Table 2: Delta from reference ----
    print(f"\n{'='*60}")
    print("  Table 2: Delta from reference (V00)")
    print(f"{'='*60}")
    if args.format in ("latex", "both"):
        delta_tex = build_delta_table(metrics_df)
        with open(output_dir / "delta_table.tex", "w") as f:
            f.write(delta_tex)
        print(f"  → {output_dir / 'delta_table.tex'}")
    # Also print a quick console summary
    ref_row = metrics_df[metrics_df["tag"] == "V00"]
    if not ref_row.empty:
        ref_ef1 = ref_row.iloc[0].get("test_ef_1", np.nan)
        for _, r in metrics_df.iterrows():
            ef1 = r.get("test_ef_1", np.nan)
            if pd.isna(ef1):
                continue
            delta = ef1 - ref_ef1 if not pd.isna(ref_ef1) else np.nan
            delta_str = f"{delta:+.2f}" if not pd.isna(delta) else "—"
            marker = " ← ref" if r["tag"] == "V00" else ""
            print(f"  {r['tag']:<6} EF@1%={ef1:6.2f}  Δ={delta_str:>7}{marker}")

    # ---- Table 3: Weight comparison ----
    print(f"\n{'='*60}")
    print("  Table 3: Weight distributions")
    print(f"{'='*60}")
    wt_df = build_weights_table(all_data)
    if not wt_df.empty:
        if args.format in ("csv", "both"):
            wt_df.to_csv(output_dir / "weights_comparison.csv", index=False)
            print(f"  → {output_dir / 'weights_comparison.csv'}")
        if args.format in ("latex", "both"):
            wt_tex = build_weights_latex(wt_df)
            with open(output_dir / "weights_table.tex", "w") as f:
                f.write(wt_tex)
            print(f"  → {output_dir / 'weights_table.tex'}")

        # Console: show just the significant modalities
        mod_cols = [c for c in wt_df.columns if c not in ("tag", "label")]
        for _, r in wt_df.iterrows():
            sig = [(c, r[c]) for c in mod_cols if not pd.isna(r.get(c, np.nan)) and r[c] > 0.05]
            sig.sort(key=lambda x: -x[1])
            sig_str = ", ".join([f"{name}={val:.1%}" for name, val in sig])
            print(f"  {r['tag']:<6} {sig_str}")

    # ---- Table 4: Pipeline comparison ----
    pipe_rows = metrics_df.dropna(subset=["pipe_n_selected"])
    if not pipe_rows.empty:
        print(f"\n{'='*60}")
        print("  Table 4: Pipeline selection summary")
        print(f"{'='*60}")
        for _, r in pipe_rows.iterrows():
            n_sel = int(r.get("pipe_n_selected", 0))
            n_univ = int(r.get("pipe_n_universe", 0))
            thresh = r.get("pipe_score_threshold", np.nan)
            print(f"  {r['tag']:<6} selected={n_sel:>5}/{n_univ:>5}  score_thresh={thresh:.3f}")

        if args.format in ("latex", "both"):
            pipe_tex = build_pipeline_table(metrics_df)
            with open(output_dir / "pipeline_table.tex", "w") as f:
                f.write(pipe_tex)
            print(f"  → {output_dir / 'pipeline_table.tex'}")

    # ---- Summary statistics ----
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    ef1_vals = metrics_df["test_ef_1"].dropna()
    if not ef1_vals.empty:
        print(f"  Test EF@1% range: [{ef1_vals.min():.2f}, {ef1_vals.max():.2f}]")
        print(f"  Test EF@1% mean ± std: {ef1_vals.mean():.2f} ± {ef1_vals.std():.2f}")

    gap_vals = metrics_df["overfit_gap_ef1_pct"].dropna()
    if not gap_vals.empty:
        print(f"  Overfit gap range: [{gap_vals.min():.1f}%, {gap_vals.max():.1f}%]")

    print(f"\n  All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
