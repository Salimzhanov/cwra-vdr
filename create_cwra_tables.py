#!/usr/bin/env python3
"""Create LaTeX tables from CWRA cross-validation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

CUTS = [1, 2.5, 5, 10, 20]

INDIV_ORDER = [
    "GraphDTA_Kd",
    "GraphDTA_Ki",
    "GraphDTA_IC50",
    "MLTLE_pKd",
    "TankBind",
    "DrugBAN",
    "MolTrans",
    "Vina",
    "Boltz_affinity",
    "Boltz_confidence",
    "UniMol_sim",
]

FUSION_ORDER = [
    "DE Fair (BEDROC)",
    "DE Unconstrained (BEDROC)",
    "DE Entropy (BEDROC)",
    "DE Fair",
    "DE Unconstrained",
    "DE Entropy",
    "Equal-weight",
    "Random (mean of 100)",
    "CWRA",
]

NAME_MAP = {
    "GraphDTA_Kd": r"GraphDTA $K_{d}$",
    "GraphDTA_Ki": r"GraphDTA $K_{i}$",
    "GraphDTA_IC50": r"GraphDTA $IC_{50}$",
    "MLTLE_pKd": r"MLTLE $pK_d$",
    "TankBind": "TankBind",
    "DrugBAN": "DrugBAN",
    "MolTrans": "MolTrans",
    "Vina": "AutoDock Vina",
    "Boltz_affinity": "Boltz-2 affinity",
    "Boltz_confidence": "Boltz-2 confidence",
    "UniMol_sim": "UniMol",
    "Equal-weight": "Equal-weight",
    "CWRA": "CWRA",
}

METHOD_LABELS = {
    "fair": "DE Fair",
    "unconstrained": "DE Unconstrained",
    "entropy": "DE Entropy",
    "fair_bedroc": "DE Fair (BEDROC)",
    "unconstrained_bedroc": "DE Unconstrained (BEDROC)",
    "entropy_bedroc": "DE Entropy (BEDROC)",
}

BASELINE_LABELS = {
    "Equal Weights": "Equal-weight",
    "Random (mean of 100)": "Random (mean of 100)",
}


def _infer_prefix(results_dir: Path) -> str:
    matches = sorted(results_dir.glob("*_paper_table.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_paper_table.csv found in {results_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple paper tables found in {results_dir}; pass --prefix explicitly. "
            f"Found: {[p.name for p in matches]}"
        )
    return matches[0].name[: -len("_paper_table.csv")]


def _sort_methods(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(order)}
    out = df.copy()
    out["_order"] = out["method"].map(rank).fillna(len(rank)).astype(int)
    out = out.sort_values(["_order", "method"]).drop(columns=["_order"])
    return out.reset_index(drop=True)


def _best_second(df: pd.DataFrame, col: str) -> Tuple[int, int]:
    vals = df[col].to_numpy(dtype=float)
    valid_idx = np.where(np.isfinite(vals))[0]
    if len(valid_idx) == 0:
        return -1, -1
    order = valid_idx[np.argsort(-vals[valid_idx])]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else int(order[0])
    return best, second


def _detect_cuts_from_folds(*dfs: pd.DataFrame) -> List[float]:
    found = set()
    for df in dfs:
        for col in df.columns:
            m = re.match(r"^test_ef_([0-9.]+)$", str(col))
            if m:
                try:
                    found.add(float(m.group(1)))
                except ValueError:
                    continue
    if not found:
        return CUTS
    preferred = [1.0, 2.5, 5.0, 10.0, 20.0, 30.0]
    cuts = [c for c in preferred if c in found]
    if cuts:
        return cuts
    return sorted(found)


def _fmt_pm(mean: float, std: float, digits: int, as_int: bool = False) -> str:
    if as_int:
        if np.isfinite(std):
            return f"{mean:.0f}$\\pm${std:.0f}"
        return f"{mean:.0f}"
    if np.isfinite(std):
        return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"
    return f"{mean:.{digits}f}"


def _fmt_cut(c: float) -> str:
    return f"{int(c)}" if float(c).is_integer() else f"{c:g}"


def _cut_key(c: float) -> str:
    return _fmt_cut(c)


def _decorate(text: str, idx: int, best: int, second: int) -> str:
    if idx == best:
        return f"\\textbf{{{text}}}"
    if idx == second:
        return f"\\underline{{{text}}}"
    return text


def _aggregate_metrics(df: pd.DataFrame, key_col: str, key_to_label: dict | None = None) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(key_col):
        label = key_to_label.get(key, key) if key_to_label is not None else key
        row = {"method": label}
        for c in CUTS:
            ck = _cut_key(c)
            ef_col = f"test_ef_{ck}"
            hits_col = f"test_hits_{ck}"
            if ef_col in grp.columns:
                row[f"ef_{ck}"] = grp[ef_col].mean()
                row[f"ef_{ck}_std"] = grp[ef_col].std(ddof=0)
            if hits_col in grp.columns:
                row[f"hits_{ck}"] = grp[hits_col].mean()
                row[f"hits_{ck}_std"] = grp[hits_col].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_run_metadata(
    results_dir: Path,
    prefix: str,
    chosen_raw: str,
    baseline_df: pd.DataFrame,
) -> dict:
    meta: dict = {}

    folds_info_path = results_dir / f"{prefix}_folds_info.csv"
    if folds_info_path.exists():
        info = pd.read_csv(folds_info_path)
        if not info.empty:
            first = info.iloc[0]
            meta["n_total"] = int(first.get("n_total", np.nan))
            meta["n_actives"] = int(first.get("n_actives_full", np.nan))
            meta["n_inactives"] = int(first.get("n_inactives", np.nan))
            meta["n_folds"] = int(len(info))
            meta["split_seed"] = int(first.get("split_seed", np.nan))

    filt_path = results_dir / f"{prefix}_filter_report.csv"
    if filt_path.exists():
        filt = pd.read_csv(filt_path)
        if not filt.empty:
            f = filt.iloc[0]
            n_before = f.get("n_before", np.nan)
            n_after = f.get("n_after", np.nan)
            n_removed = f.get("n_removed", np.nan)
            max_mw = f.get("max_mw", np.nan)
            max_rotb = f.get("max_rotb", np.nan)
            if pd.notna(n_before):
                meta["filter_before"] = int(n_before)
            if pd.notna(n_after):
                meta["filter_after"] = int(n_after)
            if pd.notna(n_removed):
                meta["filter_removed"] = int(n_removed)
            if pd.notna(max_mw):
                meta["max_mw"] = float(max_mw)
            if pd.notna(max_rotb):
                meta["max_rotb"] = int(max_rotb)

    extra_path = results_dir / f"{prefix}_folds_extra_metrics.csv"
    if extra_path.exists():
        ex = pd.read_csv(extra_path)
        ex = ex[ex["method"] == chosen_raw] if "method" in ex.columns else ex
        if not ex.empty and all(c in ex.columns for c in ["test_auroc", "test_auprc", "test_bedroc"]):
            meta["cwra_auroc_mean"] = float(ex["test_auroc"].mean())
            meta["cwra_auroc_std"] = float(ex["test_auroc"].std(ddof=0))
            meta["cwra_auprc_mean"] = float(ex["test_auprc"].mean())
            meta["cwra_auprc_std"] = float(ex["test_auprc"].std(ddof=0))
            meta["cwra_bedroc_mean"] = float(ex["test_bedroc"].mean())
            meta["cwra_bedroc_std"] = float(ex["test_bedroc"].std(ddof=0))

    random_trials = None
    for b in baseline_df["baseline"].dropna().unique():
        m = re.search(r"Random \\(mean of (\\d+)\\)", str(b))
        if m:
            random_trials = int(m.group(1))
            break
    if random_trials is not None:
        meta["random_trials"] = random_trials

    return meta


def _build_performance_table(df: pd.DataFrame, caption: str, label: str, wide: bool = True) -> str:
    table_cuts = [
        c for c in CUTS
        if (f"ef_{_cut_key(c)}" in df.columns or f"hits_{_cut_key(c)}" in df.columns)
    ]
    if not table_cuts:
        table_cuts = CUTS
    metric_cols = [
        col for col in (
            [f"ef_{_cut_key(c)}" for c in table_cuts] +
            [f"hits_{_cut_key(c)}" for c in table_cuts]
        )
        if col in df.columns
    ]
    best_second = {col: _best_second(df, col) for col in metric_cols}
    n_cuts = len(table_cuts)

    env = "table*" if wide else "table"
    lines: List[str] = []
    lines.append(rf"\begin{{{env}}}[htbp!]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{l{'r' * n_cuts}|{'r' * n_cuts}}}")
    lines.append(r"\toprule")
    lines.append(
        rf"& \multicolumn{{{n_cuts}}}{{c|}}{{Enrichment Factor (EF)}} & "
        rf"\multicolumn{{{n_cuts}}}{{c}}{{Hits}} \\"
    )
    ef_hdr = " & ".join([f"@{_fmt_cut(c)}\\%" for c in table_cuts])
    hits_hdr = " & ".join([f"@{_fmt_cut(c)}\\%" for c in table_cuts])
    lines.append(f"Method & {ef_hdr} & {hits_hdr} \\\\")
    lines.append(r"\midrule")

    for idx, row in df.iterrows():
        name = NAME_MAP.get(row["method"], row["method"])
        ef_cells = []
        for c in table_cuts:
            ck = _cut_key(c)
            ef_col = f"ef_{ck}"
            if ef_col in df.columns:
                txt = _fmt_pm(row[ef_col], row.get(f"{ef_col}_std", np.nan), 2, as_int=False)
                if ef_col in best_second:
                    b, s = best_second[ef_col]
                    txt = _decorate(txt, idx, b, s)
            else:
                txt = "N/A"
            ef_cells.append(txt)
        hit_cells = []
        for c in table_cuts:
            ck = _cut_key(c)
            hits_col = f"hits_{ck}"
            if hits_col in df.columns:
                txt = _fmt_pm(row[hits_col], row.get(f"{hits_col}_std", np.nan), 0, as_int=True)
                if hits_col in best_second:
                    b, s = best_second[hits_col]
                    txt = _decorate(txt, idx, b, s)
            else:
                txt = "N/A"
            hit_cells.append(txt)
        lines.append(f"{name} & {' & '.join(ef_cells + hit_cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


def _load_mean_rank(results_dir: Path, prefix: str) -> pd.DataFrame:
    summary_path = results_dir / f"{prefix}_mean_rank_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        req = {"method", "test_mean_rank_mean", "test_mean_rank_std"}
        if req.issubset(df.columns):
            return df

    # Fallback for runs produced before mean-rank summary export.
    indiv_path = results_dir / f"{prefix}_folds_individual.csv"
    methods_path = results_dir / f"{prefix}_folds_methods.csv"
    if not indiv_path.exists() or not methods_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path} and cannot find fallback fold files in {results_dir}."
        )
    indiv = pd.read_csv(indiv_path)
    methods = pd.read_csv(methods_path)
    if "test_mean_rank" not in indiv.columns or "test_mean_rank" not in methods.columns:
        raise ValueError(
            f"Missing {summary_path} and fallback fold files do not contain test_mean_rank. "
            "Re-run cwra.py with the updated exporter."
        )

    indiv_summary = (
        indiv.groupby("modality", as_index=False)
        .agg(
            test_mean_rank_mean=("test_mean_rank", "mean"),
            test_mean_rank_std=("test_mean_rank", "std"),
        )
        .rename(columns={"modality": "method"})
    )
    chosen = (
        methods.groupby("method", as_index=False)
        .agg(test_ef_1_mean=("test_ef_1", "mean"))
        .sort_values("test_ef_1_mean", ascending=False)
        .iloc[0]["method"]
    )
    cwra = methods[methods["method"] == chosen]
    cwra_summary = pd.DataFrame(
        {
            "method": ["CWRA"],
            "test_mean_rank_mean": [cwra["test_mean_rank"].mean()],
            "test_mean_rank_std": [cwra["test_mean_rank"].std(ddof=0)],
        }
    )
    return pd.concat([indiv_summary, cwra_summary], ignore_index=True)


def _build_weights_meanrank_table(weights_df: pd.DataFrame, mean_rank_df: pd.DataFrame) -> str:
    w = weights_df[["modality", "weight", "weight_std"]].rename(columns={"modality": "method"})
    mr = mean_rank_df[["method", "test_mean_rank_mean", "test_mean_rank_std"]]
    merged = w.merge(mr, on="method", how="outer")

    keep = [m for m in INDIV_ORDER if m in set(merged["method"]) ]
    if "CWRA" in set(merged["method"]):
        keep.append("CWRA")
    merged = merged[merged["method"].isin(keep)]
    merged = _sort_methods(merged, keep)

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Fusion weights learned by CWRA and average rank position of active compounds (5-fold CV, mean $\pm$ std).}"
    )
    lines.append(r"\label{tab:fusion_weights_meanrank}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Weight & Mean Rank \\")
    lines.append(r"\midrule")

    for _, row in merged.iterrows():
        name = NAME_MAP.get(row["method"], row["method"])
        if row["method"] == "CWRA":
            w_txt = "-"
        else:
            w_txt = _fmt_pm(row["weight"], row.get("weight_std", np.nan), 3, as_int=False)
        mr_txt = _fmt_pm(row["test_mean_rank_mean"], row.get("test_mean_rank_std", np.nan), 0, as_int=True)
        lines.append(f"{name} & {w_txt} & {mr_txt} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create LaTeX tables from CWRA cross-validation outputs.")
    parser.add_argument("--results-dir", required=True, help="Directory containing CWRA cross-validation CSV outputs")
    parser.add_argument("--prefix", default=None, help="File prefix (auto-detected from *_paper_table.csv if omitted)")
    parser.add_argument("--out-concise", default="fusion_performance_concise.tex", help="Paper table (fusion-only)")
    parser.add_argument("--out-extended", default="fusion_performance_extended.tex", help="Supplement table (individual + fusion)")
    parser.add_argument("--out-weights", default="fusion_weights_meanrank_table.tex", help="Weights + mean-rank table")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    prefix = args.prefix or _infer_prefix(results_dir)
    indiv_path = results_dir / f"{prefix}_folds_individual.csv"
    baseline_path = results_dir / f"{prefix}_folds_baselines.csv"
    methods_path = results_dir / f"{prefix}_folds_methods.csv"
    weights_path = results_dir / f"{prefix}_mean_weights.csv"

    for p in [indiv_path, baseline_path, methods_path, weights_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    indiv_df = pd.read_csv(indiv_path)
    baseline_df = pd.read_csv(baseline_path)
    methods_df = pd.read_csv(methods_path)
    weights_df = pd.read_csv(weights_path)
    mean_rank_df = _load_mean_rank(results_dir, prefix)

    global CUTS
    CUTS = _detect_cuts_from_folds(indiv_df, baseline_df, methods_df)

    indiv_agg = _aggregate_metrics(indiv_df, "modality")
    methods_agg = _aggregate_metrics(methods_df, "method", METHOD_LABELS)
    baseline_agg = _aggregate_metrics(baseline_df, "baseline", BASELINE_LABELS)

    # CWRA row = best method by mean EF@1 among optimization methods.
    chosen_raw = (
        methods_df.groupby("method", as_index=False)
        .agg(test_ef_1_mean=("test_ef_1", "mean"))
        .sort_values("test_ef_1_mean", ascending=False)
        .iloc[0]["method"]
    )
    chosen_label = METHOD_LABELS.get(chosen_raw, chosen_raw)
    cwra_row = methods_agg[methods_agg["method"] == chosen_label].copy()
    if cwra_row.empty:
        raise ValueError(f"Could not locate chosen method row: {chosen_label}")
    cwra_row.loc[:, "method"] = "CWRA"
    meta = _load_run_metadata(results_dir, prefix, chosen_raw, baseline_df)

    fusion_df = pd.concat([methods_agg, baseline_agg, cwra_row], ignore_index=True)
    concise_keep = [m for m in FUSION_ORDER if m in set(fusion_df["method"])]
    concise_df = fusion_df[fusion_df["method"].isin(concise_keep)]
    concise_df = _sort_methods(concise_df, concise_keep)

    extended_df = pd.concat([indiv_agg, concise_df], ignore_index=True)
    extended_order = [m for m in INDIV_ORDER if m in set(extended_df["method"])] + concise_keep
    extended_df = _sort_methods(extended_df, extended_order)

    summary_bits = []
    if "n_total" in meta and "n_actives" in meta:
        summary_bits.append(
            f"N={meta['n_total']:,} compounds ({meta.get('n_actives', 0):,} actives)"
        )
    if "n_folds" in meta:
        summary_bits.append(f"{meta['n_folds']}-fold CV")
    if "split_seed" in meta:
        summary_bits.append(f"split seed={meta['split_seed']}")
    if "random_trials" in meta:
        summary_bits.append(f"random baseline mean of {meta['random_trials']} trials")
    if "filter_removed" in meta and "filter_before" in meta:
        summary_bits.append(
            f"drug-likeness pre-filter removed {meta['filter_removed']:,}/{meta['filter_before']:,} compounds"
        )
    if "n_total" in meta:
        ks = [max(1, int(meta["n_total"] * c / 100)) for c in CUTS]
        summary_bits.append(
            "top-k cutoffs: " + ", ".join([f"@{_fmt_cut(c)}%={k}" for c, k in zip(CUTS, ks)])
        )
    summary_line = "; ".join(summary_bits) if summary_bits else "5-fold CV"

    cwra_extra_line = ""
    if "cwra_auroc_mean" in meta:
        cwra_extra_line = (
            f" CWRA test AUROC={_fmt_pm(meta['cwra_auroc_mean'], meta['cwra_auroc_std'], 2)},"
            f" AUPRC={_fmt_pm(meta['cwra_auprc_mean'], meta['cwra_auprc_std'], 2)},"
            f" BEDROC={_fmt_pm(meta['cwra_bedroc_mean'], meta['cwra_bedroc_std'], 2)}."
        )

    concise_tex = _build_performance_table(
        concise_df,
        caption=(
            f"Fusion-method performance on the VDR ligand discovery task (mean $\\\\pm$ std). {summary_line}. "
            f"Optimization objective represented by CWRA row: {chosen_label}.{cwra_extra_line} "
            "EF: Enrichment Factor; Hits: number of active compounds retrieved. "
            "\\textbf{Bold}: best; \\underline{underlined}: second best."
        ),
        label="tab:fusion_performance_concise",
        wide=False,
    )

    extended_tex = _build_performance_table(
        extended_df,
        caption=(
            "Extended virtual screening performance of individual scoring methods and fusion baselines "
            f"on the VDR ligand discovery task (mean $\\\\pm$ std). {summary_line}. "
            "EF: Enrichment Factor; Hits: number of active compounds retrieved. "
            "\\textbf{Bold}: best; \\underline{underlined}: second best."
        ),
        label="tab:fusion_performance_extended",
        wide=True,
    )

    weights_tex = _build_weights_meanrank_table(weights_df, mean_rank_df)

    out_concise = results_dir / args.out_concise
    out_extended = results_dir / args.out_extended
    out_weights = results_dir / args.out_weights
    out_concise.write_text(concise_tex, encoding="utf-8")
    out_extended.write_text(extended_tex, encoding="utf-8")
    out_weights.write_text(weights_tex, encoding="utf-8")

    print(f"Wrote concise table to {out_concise}")
    print(f"Wrote extended table to {out_extended}")
    print(f"Wrote weights/mean-rank table to {out_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
