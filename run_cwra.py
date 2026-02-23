#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


RESULTS_ROOT = Path("results/final")


def build_commands(python_bin: str) -> list[list[str]]:
    return [
        [
            python_bin,
            "cwra.py",
            "-i",
            "data/composed_modalities_with_rdkit.csv",
            "-o",
            str(RESULTS_ROOT),
            "--method",
            "fair",
            "--norm",
            "minmax",
            "--objective",
            "default",
            "--bedroc",
            "--bedroc-alpha",
            "100",
            "--max-mw",
            "700",
            "--train-frac",
            "0.85",
            "--cv-folds",
            "5",
            "--seed",
            "42",
            "--de-workers",
            "-1",
            "--include-newref-137-as-active",
            "--fold-honest-unimol",
            "--unimol-embeddings",
            "data/unimol_embeddings.npz",
            "--extra-metrics",
        ],
        [
            python_bin,
            "create_cwra_tables.py",
            "--results-dir",
            str(RESULTS_ROOT),
            "--out-concise",
            "fusion_performance_concise.tex",
            "--out-extended",
            "fusion_performance_extended.tex",
            "--out-weights",
            "fusion_weights_meanrank_table.tex",
        ],
        [
            python_bin,
            "pu_conformal.py",
            "--input",
            "data/composed_modalities_with_rdkit.csv",
            "--outdir",
            str(RESULTS_ROOT / "pipeline"),
            "--score-source",
            "cwra",
            "--cwra-weights-csv",
            str(RESULTS_ROOT / "cwra_cv_mean_weights.csv"),
            "--cwra-norm",
            "minmax",
            "--include-newref-137-as-active",
            "--seed",
            "42",
            "--max-mw",
            "700",
            "--max-rotb",
            "15",
            "--pval-type",
            "unweighted",
        ],
        [
            python_bin,
            "make_mol_panel.py",
            "--csv",
            str(RESULTS_ROOT / "pipeline/E/final_selected.csv"),
            "--out",
            str(RESULTS_ROOT / "panel_10x5.pdf"),
            "--n",
            "50",
            "--per-row",
            "5",
            "--cell",
            "320",
            "--sort",
            "meta",
            "--tau",
            "0.001",
        ],
        [
            python_bin,
            "plot_results/01_data_pipeline.py",
            "--input-selected",
            str(RESULTS_ROOT / "pipeline/E/final_selected.csv"),
            "--output-dir",
            str(RESULTS_ROOT / "analysis"),
        ],
        [
            python_bin,
            "plot_results/plot_cwra_stats.py",
            "--report-json",
            str(RESULTS_ROOT / "analysis/report_data.json"),
            "--pu-labels",
            str(RESULTS_ROOT / "pipeline/A/pu_labels.csv"),
            "--composed-csv",
            "data/composed_modalities_with_rdkit.csv",
            "--out",
            str(RESULTS_ROOT / "FigCWRA_stats_combined.pdf"),
        ],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CWRA paper-results pipeline sequentially."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    python_bin = sys.executable

    (repo_root / RESULTS_ROOT).mkdir(parents=True, exist_ok=True)

    commands = build_commands(python_bin)
    total = len(commands)
    for idx, cmd in enumerate(commands, start=1):
        pretty = shlex.join(cmd)
        print(f"[{idx}/{total}] {pretty}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=repo_root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
