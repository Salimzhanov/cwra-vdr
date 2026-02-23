#!/usr/bin/env python3
"""
Grid search over CWRA configuration using a fixed base command.

Fixed options applied to every run:
- --include-newref-137-as-active
- --fold-honest-unimol
- --unimol-embeddings data/unimol_embeddings.npz
- --seed 42
- --de-workers -1
- --objective default
- --cv-folds 5
- --bedroc

Varied parameters:
1) method       = ['unconstrained', 'fair', 'entropy']
2) bedroc-alpha = [60, 100, 140]
3) max-mw       = [600, 700]
4) train-frac   = [0.75, 0.8, 0.85]
5) norm         = ['minmax']
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Fixed base configuration (applied to all runs)
OBJECTIVE = "default"
SEED = 42
DE_WORKERS = -1
CV_FOLDS = 5
UNIMOL_EMBEDDINGS = "data/unimol_embeddings.npz"

# Search space
METHODS = ["unconstrained", "fair", "entropy"]
BEDROC_ALPHAS = [60, 100, 140]
MAX_MWS = [600, 700]
TRAIN_FRACS = [0.75, 0.8, 0.85]
NORMS = ["minmax"]

COMPARE_HEADER = "--- Test-set comparison (mean across folds) ---"
COMPARE_KEYS = [
    "CWRA vs equal-weight:",
    "CWRA vs best-indiv:",
    "DE Unconstrained vs best-indiv:",
    "DE Entropy vs best-indiv:",
    "CWRA vs best-indiv (all EF):",
]
TARGET_EF_ORDER = ["0.5", "1", "2.5", "5", "10", "20"]


def _run_tag(method: str, norm: str, alpha: int, max_mw: int, train_frac: float) -> str:
    tf = str(train_frac).replace(".", "p")
    return f"{method}_{norm}_a{alpha}_mw{max_mw}_tf{tf}"


def _extract_comparison_block(stdout_text: str) -> Dict[str, str]:
    lines = stdout_text.splitlines()
    out: Dict[str, str] = {COMPARE_HEADER: "<MISSING>"}
    for key in COMPARE_KEYS:
        out[key] = "<MISSING>"

    for line in lines:
        stripped = line.strip()
        if stripped == COMPARE_HEADER:
            out[COMPARE_HEADER] = COMPARE_HEADER
        for key in COMPARE_KEYS:
            if stripped.startswith(key):
                out[key] = stripped
    out["CWRA vs best-indiv (all EF):"] = _normalize_all_ef_line(
        out["CWRA vs best-indiv (all EF):"]
    )
    return out


def _normalize_all_ef_line(line: str) -> str:
    """
    Keep only EF@{0.5,1,2.5,5,10,20}% in fixed order and drop EF@30%.
    """
    key = "CWRA vs best-indiv (all EF):"
    if line == "<MISSING>" or not line.startswith(key):
        return line

    payload = line[len(key):].strip()
    parts = [p.strip() for p in payload.split(",")]
    parsed: Dict[str, str] = {}
    for part in parts:
        m = re.match(r"^EF@([0-9.]+)%\s+([+-]?[0-9.]+%)$", part)
        if not m:
            continue
        cutoff, delta = m.groups()
        parsed[cutoff] = delta

    filtered = [f"EF@{c}% {parsed[c]}" for c in TARGET_EF_ORDER if c in parsed]
    if not filtered:
        return line
    return f"{key} {', '.join(filtered)}"


def _build_cwra_cmd(
    input_csv: str,
    outdir: Path,
    method: str,
    norm: str,
    alpha: int,
    max_mw: int,
    train_frac: float,
) -> List[str]:
    return [
        sys.executable,
        "cwra.py",
        "-i",
        input_csv,
        "-o",
        str(outdir),
        "--method",
        method,
        "--norm",
        norm,
        "--objective",
        OBJECTIVE,
        "--bedroc",
        "--bedroc-alpha",
        str(alpha),
        "--max-mw",
        str(max_mw),
        "--train-frac",
        str(train_frac),
        "--cv-folds",
        str(CV_FOLDS),
        "--seed",
        str(SEED),
        "--de-workers",
        str(DE_WORKERS),
        "--include-newref-137-as-active",
        "--fold-honest-unimol",
        "--unimol-embeddings",
        UNIMOL_EMBEDDINGS,
        "--extra-metrics",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search CWRA grid and collect test-set comparison lines."
    )
    parser.add_argument(
        "--input",
        default="data/composed_modalities_with_rdkit.csv",
        help="Input CSV for cwra.py",
    )
    parser.add_argument(
        "--output-root",
        default="results/config_search",
        help="Root directory for per-run outputs",
    )
    parser.add_argument(
        "--results-file",
        default="search_configuration_results.txt",
        help="Text file to store parameter combinations and comparison lines",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose output directory already exists",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / args.results_file

    combos: List[Tuple[str, str, int, int, float]] = list(
        itertools.product(METHODS, NORMS, BEDROC_ALPHAS, MAX_MWS, TRAIN_FRACS)
    )
    total = len(combos)

    with open(results_path, "w", encoding="utf-8") as out_f:
        out_f.write("# CWRA configuration search results\n")
        out_f.write(f"# total_runs={total}\n")
        out_f.write(f"# cv_folds={CV_FOLDS}\n")
        out_f.write(
            "# fixed: objective=default, bedroc=on, "
            "seed=42, de-workers=-1, include-newref-137-as-active, "
            "fold-honest-unimol, unimol-embeddings=data/unimol_embeddings.npz\n\n"
        )

    print(f"Running {total} configurations (cv-folds={CV_FOLDS})")
    print(f"Results file: {results_path}")

    for i, (method, norm, alpha, max_mw, train_frac) in enumerate(combos, start=1):
        tag = _run_tag(method, norm, alpha, max_mw, train_frac)
        run_outdir = output_root / tag

        if args.resume and run_outdir.exists():
            print(f"[{i:03d}/{total}] {tag}  SKIP (exists)")
            with open(results_path, "a", encoding="utf-8") as out_f:
                out_f.write(f"[{i}/{total}] {tag}\n")
                out_f.write(
                    f"params: method={method}, objective={OBJECTIVE}, bedroc-alpha={alpha}, "
                    f"max-mw={max_mw}, train-frac={train_frac}, norm={norm}\n"
                )
                out_f.write("status: SKIPPED (resume)\n\n")
            continue

        cmd = _build_cwra_cmd(
            input_csv=args.input,
            outdir=run_outdir,
            method=method,
            norm=norm,
            alpha=alpha,
            max_mw=max_mw,
            train_frac=train_frac,
        )

        print(f"[{i:03d}/{total}] {tag}  START")
        result = subprocess.run(cmd, text=True, capture_output=True)
        status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
        print(f"[{i:03d}/{total}] {tag}  {status}")

        comparison = _extract_comparison_block(result.stdout)

        with open(results_path, "a", encoding="utf-8") as out_f:
            out_f.write(f"[{i}/{total}] {tag}\n")
            out_f.write(
                f"params: method={method}, objective={OBJECTIVE}, bedroc-alpha={alpha}, "
                f"max-mw={max_mw}, train-frac={train_frac}, norm={norm}\n"
            )
            out_f.write(f"output_dir: {run_outdir}\n")
            out_f.write(f"status: {status}\n")
            out_f.write(f"{comparison[COMPARE_HEADER]}\n")
            for key in COMPARE_KEYS:
                out_f.write(f"  {comparison[key]}\n")
            if result.returncode != 0:
                stderr_tail = "\n".join(result.stderr.splitlines()[-20:])
                out_f.write("stderr_tail:\n")
                out_f.write(stderr_tail + "\n")
            out_f.write("\n")

    print("Search complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
