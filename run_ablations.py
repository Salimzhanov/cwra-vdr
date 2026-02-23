#!/usr/bin/env python3
"""
Run CWRA CV ablations and (optionally) the PU conformal pipeline for each.

Usage:
    python run_ablations.py                       # run all variants
    python run_ablations.py --only V00 V01 V03    # run specific variants
    python run_ablations.py --dry-run              # print commands without running
    python run_ablations.py --skip-pipeline        # CV only, no pipeline
    python run_ablations.py --only V00 --pipeline-only  # pipeline only (CV must exist)

Results are stored under results/<variant_tag>/ with descriptive folder names.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# Variant definitions
# ============================================================================

@dataclass
class Variant:
    """One ablation variant."""
    tag: str                         # short ID, e.g. "V00"
    label: str                       # human-readable description
    folder: str                      # results subfolder name
    tier: str                        # grouping for the paper table

    # --- CWRA CV parameters (deviations from best) ---
    norm: str = "minmax"
    objective: str = "default"       # only used when use_bedroc=False
    use_bedroc: bool = True
    bedroc_alpha: float = 80.0
    max_mw: float = 600.0
    max_rotb: int = 15
    seed: int = 42
    de_seeds: int = 1
    strict_cv: bool = False
    drop_modalities: list = field(default_factory=list)
    auto_prune: float = 0.0
    extra_cwra_args: list = field(default_factory=list)

    # --- Pipeline control ---
    run_pipeline: bool = True        # whether to run PU conformal after CV


def build_variants() -> list[Variant]:
    """Define all ablation variants.
    
    V00 is the best/reference configuration. Each subsequent variant
    changes exactly ONE thing from V00 to isolate its effect.
    """
    variants = []

    # ================================================================
    # V00: REFERENCE (best configuration)
    # ================================================================
    variants.append(Variant(
        tag="V00", tier="reference",
        label="Best config (reference)",
        folder="V00_best_minmax_bedroc80_mw600",
    ))

    # ================================================================
    # Tier 1: Normalization
    # ================================================================
    variants.append(Variant(
        tag="V01", tier="normalization",
        label="Rank normalization",
        folder="V01_rank_bedroc80_mw600",
        norm="rank",
    ))
    variants.append(Variant(
        tag="V02", tier="normalization",
        label="Robust normalization",
        folder="V02_robust_bedroc80_mw600",
        norm="robust",
    ))

    # ================================================================
    # Tier 2: Objective function
    # ================================================================
    variants.append(Variant(
        tag="V03", tier="objective",
        label="Default EF objective (no BEDROC)",
        folder="V03_minmax_ef_default_mw600",
        use_bedroc=False, objective="default",
    ))
    variants.append(Variant(
        tag="V04", tier="objective",
        label="Sharp EF objective (no BEDROC)",
        folder="V04_minmax_ef_sharp_mw600",
        use_bedroc=False, objective="sharp",
    ))
    variants.append(Variant(
        tag="V05", tier="objective",
        label="BEDROC alpha=160 (top ~0.6%)",
        folder="V05_minmax_bedroc160_mw600",
        bedroc_alpha=160.0,
    ))
    variants.append(Variant(
        tag="V06", tier="objective",
        label="BEDROC alpha=20 (standard, top ~5%)",
        folder="V06_minmax_bedroc20_mw600",
        bedroc_alpha=20.0,
    ))

    # ================================================================
    # Tier 3: Drug-likeness filter
    # ================================================================
    variants.append(Variant(
        tag="V07", tier="filter",
        label="No drug-likeness filter",
        folder="V07_minmax_bedroc80_nofilter",
        max_mw=0, max_rotb=0,
    ))
    variants.append(Variant(
        tag="V08", tier="filter",
        label="Stricter filter (MW<500)",
        folder="V08_minmax_bedroc80_mw500",
        max_mw=500.0,
    ))

    # ================================================================
    # Tier 4: Modality selection
    # ================================================================
    variants.append(Variant(
        tag="V09", tier="modality",
        label="Drop 4 dead modalities (manual)",
        folder="V09_minmax_bedroc80_mw600_drop4",
        drop_modalities=["GraphDTA_Kd", "GraphDTA_Ki", "MLTLE_pKd", "MolTrans"],
    ))
    variants.append(Variant(
        tag="V10", tier="modality",
        label="Auto-prune at 3% threshold",
        folder="V10_minmax_bedroc80_mw600_autoprune",
        auto_prune=0.03,
    ))
    # ================================================================
    # Tier 5: Optimization & CV robustness
    # ================================================================
    variants.append(Variant(
        tag="V12", tier="robustness",
        label="Strict CV (per-fold normalization)",
        folder="V12_minmax_bedroc80_mw600_strictcv",
        strict_cv=True,
    ))
    variants.append(Variant(
        tag="V13", tier="robustness",
        label="Multi-seed DE (3 seeds)",
        folder="V13_minmax_bedroc80_mw600_3seeds",
        de_seeds=3,
    ))
    variants.append(Variant(
        tag="V14", tier="robustness",
        label="Different seed (seed=7)",
        folder="V14_minmax_bedroc80_mw600_seed7",
        seed=7,
    ))

    return variants


# ============================================================================
# Command builders
# ============================================================================

INPUT_CSV = "data/composed_modalities_with_rdkit.csv"
RESULTS_ROOT = Path("results")
DEFAULT_UNIMOL_EMBEDDINGS = "data/unimol_embeddings.npz"


def build_cwra_cmd(
    v: Variant,
    *,
    fold_honest_unimol: bool = True,
    unimol_embeddings: str = DEFAULT_UNIMOL_EMBEDDINGS,
) -> list[str]:
    """Build the cwra.py command for a variant."""
    outdir = RESULTS_ROOT / v.folder / "cwra"
    cmd = [
        sys.executable, "cwra.py",
        "-i", INPUT_CSV,
        "-o", str(outdir),
        "--norm", v.norm,
        "--max-mw", str(v.max_mw),
        "--max-rotb", str(v.max_rotb),
        "--cv-folds", "5",
        "--seed", str(v.seed),
        "--latex",
        "--extra-metrics",
    ]
    if v.use_bedroc:
        cmd += ["--bedroc", "--bedroc-alpha", str(v.bedroc_alpha)]
    else:
        cmd += ["--objective", v.objective]

    if v.strict_cv:
        cmd += ["--strict-cv"]
    if v.de_seeds > 1:
        cmd += ["--de-seeds", str(v.de_seeds)]
    if v.drop_modalities:
        cmd += ["--drop-modalities"] + v.drop_modalities
    if v.auto_prune > 0:
        cmd += ["--auto-prune", str(v.auto_prune)]
    if fold_honest_unimol:
        cmd += [
            "--fold-honest-unimol",
            "--unimol-embeddings", unimol_embeddings,
        ]
    cmd += v.extra_cwra_args

    return cmd


def build_pipeline_cmd(v: Variant) -> list[str]:
    """Build the conformal pipeline command for a variant."""
    cwra_dir = RESULTS_ROOT / v.folder / "cwra"
    pipe_dir = RESULTS_ROOT / v.folder / "pipeline"
    weights_csv = cwra_dir / "cwra_cv_mean_weights.csv"
    conformal_script = "run_conformal.py" if Path("run_conformal.py").exists() else "pu_conformal.py"

    cmd = [
        sys.executable, conformal_script,
        "--input", INPUT_CSV,
        "--outdir", str(pipe_dir),
        "--score-source", "cwra",
        "--cwra-weights-csv", str(weights_csv),
        "--cwra-norm", v.norm,
        "--max-mw", str(v.max_mw) if v.max_mw > 0 else "0",
        "--max-rotb", str(v.max_rotb) if v.max_rotb > 0 else "0",
        "--calib-negatives", "unlabeled_random",
        "--select-mode", "pval_cutoff",
        "--pval-cutoff", "0.001",
        "--pval-type", "unweighted",
        "--top-k", "2000",
        "--seed", str(v.seed),
    ]
    return cmd


def build_panel_cmd(v: Variant, n: int = 50, per_row: int = 5, cell: int = 320) -> list[str]:
    """Build the make_mol_panel.py command for a variant."""
    pipe_dir = RESULTS_ROOT / v.folder / "pipeline"
    final_csv = pipe_dir / "E" / "final_selected.csv"
    panel_pdf = pipe_dir / "E" / "panel.pdf"

    cmd = [
        sys.executable, "make_mol_panel.py",
        "--csv", str(final_csv),
        "--out", str(panel_pdf),
        "--n", str(n),
        "--per-row", str(per_row),
        "--cell", str(cell),
        "--sort", "meta",          # sort by meta_score, not pval (avoids tie-order bug)
        "--tau", "0.001",
    ]
    return cmd


# ============================================================================
# Runner
# ============================================================================

def run_cmd(cmd: list[str], label: str, dry_run: bool = False, log_dir: Optional[Path] = None) -> bool:
    """Run a command with live stdout/stderr streaming and optional full log capture."""
    cmd_str = " \\\n    ".join(cmd)
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  $ {cmd_str}")

    if dry_run:
        print("  [DRY RUN — skipped]")
        return True

    log_path = log_dir / f"{label.replace(' ', '_')}.log" if log_dir else None

    t0 = time.time()
    proc = None
    try:
        exec_cmd = list(cmd)
        if exec_cmd and Path(exec_cmd[0]).name.startswith("python") and "-u" not in exec_cmd:
            exec_cmd.insert(1, "-u")

        proc = subprocess.Popen(
            exec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def _stream(pipe, sink: list[str], prefix: str = ""):
            try:
                for line in iter(pipe.readline, ""):
                    sink.append(line)
                    print(f"  {prefix}{line.rstrip()}")
            finally:
                pipe.close()

        t_out = threading.Thread(
            target=_stream, args=(proc.stdout, stdout_chunks, ""), daemon=True
        )
        t_err = threading.Thread(
            target=_stream, args=(proc.stderr, stderr_chunks, "STDERR: "), daemon=True
        )
        t_out.start()
        t_err.start()

        timed_out = False
        try:
            returncode = proc.wait(timeout=3600)  # 1 hour max per step
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            returncode = proc.wait()

        t_out.join()
        t_err.join()

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        elapsed = time.time() - t0

        if timed_out:
            print("  *** TIMEOUT (>3600s) ***")

        if returncode != 0 and not timed_out:
            print(f"\n  *** FAILED (exit code {returncode}) ***")

        print(f"\n  Elapsed: {elapsed:.1f}s")

        # Save full log
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(f"COMMAND: {' '.join(cmd)}\n")
                f.write(f"EXIT CODE: {returncode}\n")
                f.write(f"ELAPSED: {elapsed:.1f}s\n")
                f.write(f"\n--- STDOUT ---\n{stdout_text}\n")
                f.write(f"\n--- STDERR ---\n{stderr_text}\n")

        return (returncode == 0) and (not timed_out)

    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    except Exception as e:
        print(f"  *** ERROR: {e} ***")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run CWRA ablation experiments.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these variant tags (e.g. V00 V01)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Run only CWRA CV, skip pipeline")
    parser.add_argument("--pipeline-only", action="store_true",
                        help="Skip CV, run pipeline only (CV results must exist)")
    parser.add_argument("--results-root", type=str, default="results",
                        help="Root directory for results (default: results)")
    parser.add_argument(
        "--no-fold-honest-unimol",
        action="store_true",
        help="Disable fold-honest Uni-Mol recomputation (enabled by default).",
    )
    parser.add_argument(
        "--unimol-embeddings",
        type=str,
        default=DEFAULT_UNIMOL_EMBEDDINGS,
        help=f"Path to Uni-Mol embeddings .npz (default: {DEFAULT_UNIMOL_EMBEDDINGS}).",
    )
    args = parser.parse_args()

    global RESULTS_ROOT
    RESULTS_ROOT = Path(args.results_root)

    variants = build_variants()
    fold_honest_unimol = not args.no_fold_honest_unimol

    if fold_honest_unimol and not args.pipeline_only:
        emb_path = Path(args.unimol_embeddings)
        if not emb_path.exists():
            print(
                "ERROR: Fold-honest Uni-Mol is enabled, but embeddings file is missing:\n"
                f"  {emb_path}\n"
                "Generate it first (e.g. with unimol_embeddings.py), or run with "
                "--no-fold-honest-unimol."
            )
            sys.exit(1)

    if args.only:
        tags = set(args.only)
        variants = [v for v in variants if v.tag in tags]
        missing = tags - {v.tag for v in variants}
        if missing:
            print(f"WARNING: Unknown variant tags: {missing}")

    # Summary
    print(f"\n{'#'*72}")
    print(f"  CWRA ABLATION EXPERIMENT")
    print(f"  {len(variants)} variants to run")
    print(f"  Results root: {RESULTS_ROOT}")
    print(f"  Pipeline: {'skip' if args.skip_pipeline else 'pipeline-only' if args.pipeline_only else 'enabled'}")
    print(f"  Fold-honest Uni-Mol: {'enabled' if fold_honest_unimol else 'disabled'}")
    if fold_honest_unimol:
        print(f"  Uni-Mol embeddings: {args.unimol_embeddings}")
    print(f"{'#'*72}")

    for v in variants:
        pipeline_flag = " + pipeline" if not args.skip_pipeline else ""
        print(f"  [{v.tag}] {v.label} → {v.folder}{pipeline_flag}")

    # Create log directory
    log_dir = RESULTS_ROOT / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save variant manifest
    manifest = []
    for v in variants:
        manifest.append({
            "tag": v.tag,
            "label": v.label,
            "folder": v.folder,
            "tier": v.tier,
            "norm": v.norm,
            "objective": v.objective if not v.use_bedroc else f"bedroc_a{v.bedroc_alpha:.0f}",
            "max_mw": v.max_mw,
            "max_rotb": v.max_rotb,
            "seed": v.seed,
            "de_seeds": v.de_seeds,
            "strict_cv": v.strict_cv,
            "drop_modalities": v.drop_modalities,
            "auto_prune": v.auto_prune,
            "run_pipeline": v.run_pipeline,
            "fold_honest_unimol": fold_honest_unimol,
            "unimol_embeddings": args.unimol_embeddings if fold_honest_unimol else None,
        })
    manifest_path = RESULTS_ROOT / "ablation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved to {manifest_path}")

    # Run
    results_summary = []
    total_t0 = time.time()

    interrupted = False
    try:
        for v in variants:
            v_t0 = time.time()
            cv_ok = True
            pipe_ok = True
            panel_ok = True

            # --- CWRA CV ---
            if not args.pipeline_only:
                cwra_cmd = build_cwra_cmd(
                    v,
                    fold_honest_unimol=fold_honest_unimol,
                    unimol_embeddings=args.unimol_embeddings,
                )
                cv_ok = run_cmd(cwra_cmd, f"{v.tag}_cwra", args.dry_run, log_dir)

            # --- Pipeline ---
            if not args.skip_pipeline and (cv_ok or args.pipeline_only):
                weights_csv = RESULTS_ROOT / v.folder / "cwra" / "cwra_cv_mean_weights.csv"
                if args.dry_run or weights_csv.exists():
                    pipe_cmd = build_pipeline_cmd(v)
                    pipe_ok = run_cmd(pipe_cmd, f"{v.tag}_pipeline", args.dry_run, log_dir)
                else:
                    print(f"\n  [{v.tag}] Skipping pipeline: {weights_csv} not found")
                    pipe_ok = False

            # --- Molecule panel ---
            if not args.skip_pipeline and pipe_ok:
                final_csv = RESULTS_ROOT / v.folder / "pipeline" / "E" / "final_selected.csv"
                if args.dry_run or final_csv.exists():
                    panel_cmd = build_panel_cmd(v)
                    panel_ok = run_cmd(panel_cmd, f"{v.tag}_panel", args.dry_run, log_dir)
                else:
                    print(f"\n  [{v.tag}] Skipping panel: {final_csv} not found")
                    panel_ok = False

            v_elapsed = time.time() - v_t0
            results_summary.append({
                "tag": v.tag,
                "label": v.label,
                "cv_ok": cv_ok,
                "pipeline_ok": pipe_ok if not args.skip_pipeline else None,
                "panel_ok": panel_ok if not args.skip_pipeline else None,
                "elapsed_s": round(v_elapsed, 1),
            })
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n  Interrupted by user. Stopping remaining variants.")

    total_elapsed = time.time() - total_t0

    # Final summary
    print(f"\n\n{'#'*72}")
    status = "ABLATION INTERRUPTED" if interrupted else "ABLATION COMPLETE"
    print(f"  {status} — {total_elapsed:.0f}s total")
    print(f"{'#'*72}")
    print(f"  {'Tag':<6} {'CV':>4} {'Pipe':>6} {'Panel':>6} {'Time':>8}  Label")
    print(f"  {'-'*6} {'-'*4} {'-'*6} {'-'*6} {'-'*8}  {'-'*40}")
    for r in results_summary:
        cv_sym = "OK" if r["cv_ok"] else "FAIL"
        pipe_sym = "OK" if r["pipeline_ok"] else ("FAIL" if r["pipeline_ok"] is not None else "skip")
        panel_sym = "OK" if r.get("panel_ok") else ("FAIL" if r.get("panel_ok") is not None else "skip")
        print(f"  {r['tag']:<6} {cv_sym:>4} {pipe_sym:>6} {panel_sym:>6} {r['elapsed_s']:>7.1f}s  {r['label']}")

    # Save summary
    summary_path = RESULTS_ROOT / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
