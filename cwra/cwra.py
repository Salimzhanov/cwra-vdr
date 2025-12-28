#!/usr/bin/env python3
"""
CWRA - Calibrated Weighted Rank Fusion for VDR Virtual Screening

A method for aggregating multiple virtual screening modalities using 
calibrated weighting based on cross-validated performance metrics.

Usage:
  python -m cwra --csv data/labeled_raw_modalities.csv --output_prefix results
  
  # Use fixed optimal weights (for reproduction)
  python -m cwra --csv data/labeled_raw_modalities.csv --fixed_weights --output_prefix results

"""

from __future__ import annotations
import argparse
import math
import itertools
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional, Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import kendalltau
from sklearn.model_selection import GroupKFold, KFold
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Optimal weights
OPTIMAL_FIXED_WEIGHTS = {
    "graphdta_kd": 0.0,
    "graphdta_ki": 0.0001,
    "graphdta_ic50": 0.0,
    "mltle_pKd": 0.0,
    "vina_score": 0.0006,
    "boltz_affinity": 0.0,
    "boltz_confidence": 0.002,
    "unimol_similarity": 0.8206,
    "tankbind_affinity": 0.0004,
    "drugban_affinity": 0.0478,
    "moltrans_affinity": 0.1285,
}


MODALITIES = [
    # (column_name, high_better, latex_label, name)
    ("graphdta_kd", True, r"GraphDTA-$K_\mathrm{d}$", "GraphDTA_Kd"),       # Higher predicted Kd = better
    ("graphdta_ki", True, r"GraphDTA-$K_\mathrm{i}$", "GraphDTA_Ki"),       # Higher predicted Ki = better
    ("graphdta_ic50", True, r"GraphDTA-IC$_{50}$", "GraphDTA_IC50"),        # Higher predicted IC50 = better
    ("mltle_pKd", True, r"MLT-LE p$K_\mathrm{d}$", "MLTLE_pKd"),            # Higher pKd = stronger binding
    ("vina_score", False, r"AutoDock Vina", "Vina"),                        # More negative = better docking
    ("boltz_affinity", False, r"Boltz-2 affinity", "Boltz_affinity"),       # More negative = stronger binding
    ("boltz_confidence", True, r"Boltz-2 confidence", "Boltz_confidence"), # Higher confidence = better
    ("unimol_similarity", True, r"Uni-Mol similarity", "UniMol_sim"),       # Higher similarity = better
    ("tankbind_affinity", False, r"TankBind affinity", "TankBind"),         # More negative = stronger binding
    ("drugban_affinity", False, r"DrugBAN affinity", "DrugBAN"),            # More negative = stronger binding
    ("moltrans_affinity", False, r"MolTrans affinity", "MolTrans"),         # More negative = stronger binding
]

CUTOFF_PCTS = [1, 5, 10, 20, 30]


# =============================================================================
# SCAFFOLD COMPUTATION
# =============================================================================
def murcko_smiles(smiles: str) -> Optional[str]:
    """Compute Murcko scaffold SMILES from input SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf else None
    except Exception:
        return None


# =============================================================================
# METRICS
# =============================================================================
def bedroc(x: np.ndarray, alpha: float, A: int, N: int) -> float:
    """Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).
    
    Args:
        x: Normalized ranks of actives (0 = best, 1 = worst)
        alpha: Exponential decay parameter
        A: Number of actives
        N: Total number of compounds
    
    Returns:
        BEDROC score in [0, 1]
    """
    if A <= 0 or N <= 1:
        return 0.0
    denom = (1.0 - np.exp(-alpha)) / alpha
    rie = np.mean(np.exp(-alpha * x)) / denom
    # Ideal case: actives at ranks 1, 2, ..., A -> normalized positions (0, 1, ..., A-1)/(N-1)
    xideal = np.arange(A) / (N - 1)
    rie_max = np.mean(np.exp(-alpha * xideal)) / denom
    # Handle edge case where rie_max equals 1
    if abs(rie_max - 1.0) < 1e-12:
        return 0.0
    return (rie - 1.0) / (rie_max - 1.0)


def shrink_factors(tau: np.ndarray, tau0: float, lam: float) -> np.ndarray:
    """Compute shrinkage factors based on correlation structure."""
    n_corr = (np.abs(tau) > tau0).sum(axis=1) - 1
    return 1.0 / (1.0 + lam * n_corr)


def eval_at_cutoffs(score: np.ndarray, active_mask: np.ndarray, 
                    cutoffs: Dict[str, int], N: int) -> Dict:
    """Evaluate EF and hits at multiple cutoffs."""
    A = active_mask.sum()
    if A == 0:
        return {f"ef{k}": 0.0 for k in cutoffs} | {f"h{k}": 0 for k in cutoffs}
    
    kmax = max(cutoffs.values())
    top_idx = np.argpartition(score, min(kmax - 1, N - 1))[:kmax]
    top_idx = top_idx[np.argsort(score[top_idx])]
    
    result = {}
    for name, k in cutoffs.items():
        h = int(active_mask[top_idx[:k]].sum())
        result[f"h{name}"] = h
        result[f"ef{name}"] = (h * N) / (k * A)
    return result


def compute_weights(ef_terms: Dict, rank_score: np.ndarray, bedroc_vals: np.ndarray,
                    shrink: np.ndarray, params: Dict, E: int) -> np.ndarray:
    """Compute modality weights from performance metrics."""
    ef_combined = (params['w_ef1'] * ef_terms.get("1", np.zeros(E)) +
                   params['w_ef5'] * ef_terms.get("5", np.zeros(E)) +
                   params['w_ef10'] * ef_terms.get("10", np.zeros(E)))
    ef_sum = params['w_ef1'] + params['w_ef5'] + params['w_ef10']
    if ef_sum > 0:
        ef_combined /= ef_sum
    
    ef_weight = max(0, 1.0 - params['w_bedroc'] - params['w_rank'])
    raw = (params['w_bedroc'] * bedroc_vals + 
           params['w_rank'] * rank_score + 
           ef_weight * ef_combined)
    raw = np.maximum(raw, 1e-12)
    
    w = (raw * shrink) ** params['delta']
    w = w / w.sum()
    
    uniform = np.ones(E) / E
    w_mix = (1.0 - params['gamma']) * w + params['gamma'] * uniform
    return w_mix / w_mix.sum()


def calc_modality_metrics(active_idx: np.ndarray, ranks: np.ndarray, ranks01: np.ndarray,
                          topk_idx: Dict, cutoffs: Dict, N: int, alpha: float) -> Tuple:
    """Calculate EF, rank score, and BEDROC for each modality."""
    A, E = len(active_idx), ranks.shape[1]
    active_bool = np.zeros(N, bool)
    active_bool[active_idx] = True
    
    ef_terms = {}
    for name, idx_arr in topk_idx.items():
        hits = active_bool[idx_arr].sum(axis=0)
        ef_terms[name] = (hits * N) / (cutoffs[name] * A) if A > 0 else np.zeros(E)
    
    mean_rank = ranks[active_idx, :].mean(axis=0)
    rank_score = 1.0 - (mean_rank - 1.0) / (N - 1)
    
    x_act = ranks01[active_idx, :]
    bedroc_vals = np.array([bedroc(x_act[:, j], alpha, A, N) for j in range(E)])
    
    return ef_terms, mean_rank, rank_score, bedroc_vals


# =============================================================================
# CV UTILITIES
# =============================================================================
def balanced_group_kfold(groups: np.ndarray, n_splits: int, 
                         rng: np.random.Generator) -> List[Tuple]:
    """Create balanced group k-fold splits."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.arange(len(uniq))
    rng.shuffle(order)
    
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for g in order:
        j = int(np.argmin(fold_sizes))
        fold_groups[j].append(g)
        fold_sizes[j] += sizes[g]
    
    splits = []
    for j in range(n_splits):
        test_ids = set(fold_groups[j])
        te = np.where([gid in test_ids for gid in inv])[0]
        tr = np.setdiff1d(np.arange(len(groups)), te)
        splits.append((tr, te))
    return splits


# =============================================================================
# INNER CV EVALUATION
# =============================================================================
def eval_inner(params_tuple: Tuple, inner_pairs: List, ranks: np.ndarray, 
               ranks01: np.ndarray, topk_idx: Dict, cutoffs: Dict, N: int, 
               shrink_cache: Dict, objective_fn, E: int) -> Tuple:
    """Evaluate parameters on inner CV folds."""
    params = dict(zip(['w_ef1', 'w_ef5', 'w_ef10', 'w_bedroc', 'w_rank', 
                       'alpha', 'tau0', 'lam', 'delta', 'gamma'], params_tuple))
    shrink = shrink_cache[(params['tau0'], params['lam'])]
    
    objs = []
    for inner_tr, inner_va in inner_pairs:
        if len(inner_va) < 1:
            continue
        
        ef_terms, _, rank_score, bedroc_vals = calc_modality_metrics(
            inner_tr, ranks, ranks01, topk_idx, cutoffs, N, params['alpha'])
        
        w = compute_weights(ef_terms, rank_score, bedroc_vals, shrink, params, E)
        sc = ranks01.dot(w)
        sc[inner_tr] = np.inf
        
        eval_mask = np.zeros(N, bool)
        eval_mask[inner_va] = True
        d = eval_at_cutoffs(sc, eval_mask, cutoffs, N)
        objs.append(objective_fn(d))
    
    if not objs:
        return params_tuple, 0.0, 0.0
    return params_tuple, np.mean(objs), np.std(objs, ddof=1) if len(objs) > 1 else 0.0


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================
def obj_early(d): 
    return 0.5 * d.get("ef1", 0) + 0.3 * d.get("ef5", 0) + 0.2 * d.get("ef10", 0)

def obj_balanced(d): 
    return 0.3 * d.get("ef1", 0) + 0.4 * d.get("ef5", 0) + 0.3 * d.get("ef10", 0)

def obj_standard(d): 
    return 0.2 * d.get("ef5", 0) + 0.5 * d.get("ef10", 0) + 0.3 * d.get("ef20", 0)

def obj_comprehensive(d):
    """Optimize for all EF metrics equally."""
    return (0.25 * d.get("ef1", 0) + 0.25 * d.get("ef5", 0) + 
            0.25 * d.get("ef10", 0) + 0.15 * d.get("ef20", 0) + 0.1 * d.get("ef30", 0))


# =============================================================================
# HYPERPARAMETER GRID
# =============================================================================
def get_grid(mode: str) -> List[Tuple]:
    """Hyperparameter grid based on mode.
    
    Grid format: (ef_weights, bedroc_rank_weights, alpha, tau0, lam, delta, gamma)
    - ef_weights: (w_ef1, w_ef5, w_ef10) - weights for EF at different cutoffs
    - bedroc_rank_weights: (w_bedroc, w_rank) - BEDROC and rank score weights
    - alpha: BEDROC decay parameter
    - tau0: correlation threshold for shrinkage
    - lam: shrinkage strength
    - delta: power transform exponent
    - gamma: uniform mixing weight
    """
    if mode == "narrow":
        # Fast testing grid (~32 configs)
        return list(itertools.product(
            [(0.5, 0.3, 0.2)],
            [(0.3, 0.1)],
            [5.0, 20.0],
            [0.2],
            [0.5, 0.75],
            [2.0],
            [0.001, 0.01]
        ))
    elif mode == "optimal":
        # Optimized grid with emphasis on strong differentiation (~4000 configs)
        return list(itertools.product(
            [(0.6, 0.25, 0.15), (0.5, 0.3, 0.2), (0.4, 0.35, 0.25), (0.35, 0.35, 0.3), (0.3, 0.4, 0.3)],
            [(0.6, 0.05), (0.5, 0.08), (0.45, 0.1), (0.4, 0.1), (0.35, 0.1)],
            [5.0, 10.0, 20.0, 40.0],
            [0.05, 0.1, 0.15, 0.2, 0.3],
            [0.0, 0.1, 0.25, 0.5],
            [1.0, 1.5, 2.0, 2.5],
            [0.0, 0.001, 0.01, 0.05]
        ))
    elif mode == "wide":
        # Comprehensive grid (~6000 configs)
        return list(itertools.product(
            [(0.5, 0.3, 0.2), (0.4, 0.35, 0.25), (0.35, 0.35, 0.3), (0.3, 0.4, 0.3)],
            [(0.5, 0.05), (0.4, 0.1), (0.3, 0.1), (0.25, 0.15)],
            [5.0, 10.0, 20.0, 40.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.5, 0.75, 1.0],
            [1.5, 2.0, 2.5, 3.0],
            [0.001, 0.01, 0.02, 0.05]
        ))
    elif mode == "extended":
        # Large grid for exhaustive search
        return list(itertools.product(
            [(0.5, 0.3, 0.2), (0.4, 0.35, 0.25), (0.35, 0.35, 0.3), 
             (0.3, 0.4, 0.3), (0.3, 0.35, 0.35), (0.25, 0.4, 0.35)],
            [(0.5, 0.1), (0.4, 0.1), (0.35, 0.15), (0.3, 0.1), (0.25, 0.15)],
            [5.0, 10.0, 20.0, 40.0, 80.0],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.25, 0.5, 0.75, 1.0],
            [1.0, 1.5, 2.0, 2.5, 3.0],
            [0.001, 0.01, 0.02, 0.05, 0.1]
        ))
    else:  # default
        # Balanced grid (~2000 configs)
        return list(itertools.product(
            [(0.4, 0.35, 0.25), (0.35, 0.35, 0.3), (0.3, 0.4, 0.3)],
            [(0.4, 0.1), (0.3, 0.1), (0.25, 0.15)],
            [5.0, 10.0, 20.0, 40.0],
            [0.1, 0.2, 0.3],
            [0.5, 0.75, 1.0],
            [1.5, 2.0, 2.5],
            [0.001, 0.01, 0.02, 0.05]
        ))


def flatten_grid(grid: List[Tuple]) -> List[Tuple]:
    """Flatten nested tuples in grid to flat parameter tuples."""
    return [(ef[0], ef[1], ef[2], br[0], br[1], alpha, tau0, lam, delta, gamma)
            for ef, br, alpha, tau0, lam, delta, gamma in grid]


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================
def fmt(m: float, s: float, d: int = 2) -> str:
    """Format mean ± std for display."""
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def generate_latex_tables(table5: pd.DataFrame, s_individual: Dict, s_eq: Dict, 
                          s_cwra: Dict, s_random: Dict, final_params: Dict,
                          mod_labels: List, mod_names: List) -> str:
    """LaTeX tables for manuscript."""
    lines = []
    
    # Table 1: Modality Weights
    lines.append(r"""% Table 1: Modality Weights
\begin{table}[t]
\caption{Modality weights from BEDROC-enhanced scoring.}
\label{tab:modality_weights}
\centering
\small
\begin{tabular}{lccc}
\toprule
\textbf{Modality} & \textbf{Weight} & \textbf{EF@1\%} & \textbf{Mean Rank} \\
\midrule""")
    
    for _, row in table5.sort_values("Weight", ascending=False).iterrows():
        w = row['Weight']
        w_str = f"{w:.2f}" if w >= 0.01 else "<0.01"
        lines.append(f"{row['Modality']} & {w_str} & {row['EF@1%']:.2f} & {int(row['Mean_Rank'])} \\\\")
    
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    
    # Table 2: Performance Comparison
    # Determine caption based on evaluation mode
    is_fixed_weights = final_params.get('mode') == 'fixed_optimal_weights'
    if is_fixed_weights:
        caption = "Hit recovery using fixed optimal weights on full dataset. EF@k\\% measures enrichment factor at top k\\% of the ranked database."
    else:
        caption = "Hit recovery under scaffold-grouped nested CV. Values are mean $\\pm$ std across folds. EF@k\\% measures enrichment factor at top k\\% of the ranked database."
    
    lines.append(f"""% Table 2: Performance Comparison
\\begin{{table*}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:fusion_performance}}
\\small
\\begin{{tabular}}{{@{{}}llccccc@{{}}}}
\\toprule
Category & Method & EF@1\\% & EF@5\\% & EF@10\\% & Hits@10\\% & Hits@20\\% \\\\
\\midrule""")
    
    # Sort modalities by EF@10%
    sorted_mods = sorted(zip(mod_names, mod_labels), 
                         key=lambda x: -s_individual[x[0]]['ef10'][0])
    
    for i, (mod_name, mod_label) in enumerate(sorted_mods):
        s = s_individual[mod_name]
        mod_idx = mod_names.index(mod_name)
        direction = r"\textsuperscript{$\uparrow$}" if MODALITIES[mod_idx][1] else r"\textsuperscript{$\downarrow$}"
        prefix = r"\multirow{11}{*}{\rotatebox[origin=c]{90}{\textit{Single Modality}}}" if i == 0 else ""
        lines.append(f"{prefix} & {mod_label}{direction} & {fmt(*s['ef1'])} & {fmt(*s['ef5'])} & "
                     f"{fmt(*s['ef10'])} & {fmt(*s['h10'], d=1)} & {fmt(*s['h20'], d=1)} \\\\")
    
    lines.append(r"\midrule")
    lines.append(f"\\multirow{{2}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textit{{Fusion}}}}}} "
                 f"& Equal-weight & {fmt(*s_eq['ef1'])} & {fmt(*s_eq['ef5'])} & "
                 f"{fmt(*s_eq['ef10'])} & {fmt(*s_eq['h10'], d=1)} & {fmt(*s_eq['h20'], d=1)} \\\\")
    lines.append(f" & CWRA-early & \\textbf{{{fmt(*s_cwra['ef1'])}}} & \\textbf{{{fmt(*s_cwra['ef5'])}}} & "
                 f"\\textbf{{{fmt(*s_cwra['ef10'])}}} & \\textbf{{{fmt(*s_cwra['h10'], d=1)}}} & "
                 f"\\textbf{{{fmt(*s_cwra['h20'], d=1)}}} \\\\")
    
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{Expected at random}}}} & {fmt(*s_random['ef1'])} & "
                 f"{fmt(*s_random['ef5'])} & {fmt(*s_random['ef10'])} & {fmt(*s_random['h10'], d=1)} & "
                 f"{fmt(*s_random['h20'], d=1)} \\\\")
    
    lines.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")
    
    # Table 3: Hyperparameters
    lines.append(r"""% Table 3: Hyperparameters
\begin{table}[htbp]
\centering
\caption{Hyperparameter search space and selected values.}
\label{tab:hp_search}
\begin{tabular}{lcc}
\hline
\textbf{Parameter} & \textbf{Search space} & \textbf{Selected} \\
\hline""")
    
    if final_params:
        # Use actual grid values from get_grid() - optimal mode values
        lines.append(f"BEDROC decay $\\alpha_{{\\mathrm{{B}}}}$ & $\\{{5.0, 10.0, 20.0, 40.0, 80.0\\}}$ & {final_params.get('alpha', 'N/A')} \\\\")
        lines.append(f"Corr. threshold $\\tau_0$ & $\\{{0.05, 0.1, 0.15, 0.2, 0.3\\}}$ & {final_params.get('tau0', 0):.2f} \\\\")
        lines.append(f"Shrinkage $\\lambda$ & $\\{{0.0, 0.1, 0.25, 0.5\\}}$ & {final_params.get('lam', 0):.2f} \\\\")
        lines.append(f"Power transform $\\delta$ & $\\{{1.0, 1.5, 2.0, 2.5\\}}$ & {final_params.get('delta', 0):.1f} \\\\")
        lines.append(f"Uniform mix $\\gamma$ & $\\{{0.0, 0.001, 0.01, 0.05\\}}$ & {final_params.get('gamma', 0):.3f} \\\\")
        lines.append(f"EF weights $(a,b,c)$ & see grid & ({final_params.get('w_ef1', 0):.2f}, {final_params.get('w_ef5', 0):.2f}, {final_params.get('w_ef10', 0):.2f}) \\\\")
        lines.append(f"BEDROC/rank weights & see grid & ({final_params.get('w_bedroc', 0):.2f}, {final_params.get('w_rank', 0):.2f}) \\\\")
    
    lines.append(r"""\hline
\end{tabular}
\end{table}
""")
    
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="CWRA - Calibrated Weighted Rank Aggregation for VDR Virtual Screening",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--csv", default="data/labeled_raw_modalities.csv",
                    help="Input CSV with modalities + smiles + source")
    ap.add_argument("--outer_splits", type=int, default=5,
                    help="Number of outer CV folds")
    ap.add_argument("--outer_repeats", type=int, default=3,
                    help="Number of outer CV repeats")
    ap.add_argument("--inner_splits", type=int, default=3,
                    help="Number of inner CV folds")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    ap.add_argument("--risk_beta", type=float, default=0.3,
                    help="Risk aversion parameter (mean - beta*std)")
    ap.add_argument("--focus", type=str, default="early",
                    choices=["early", "balanced", "standard", "comprehensive"],
                    help="Optimization focus")
    ap.add_argument("--grid_mode", type=str, default="optimal",
                    choices=["narrow", "optimal", "default", "wide", "extended"],
                    help="Hyperparameter grid size")
    ap.add_argument("--output_prefix", type=str, default="results",
                    help="Prefix for output files")
    ap.add_argument("--top_n", type=int, default=25,
                    help="Number of top/bottom structures to extract")
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="Number of parallel jobs (-1 for all cores)")
    ap.add_argument("--fixed_weights", action="store_true",
                    help="Use optimal fixed weights instead of CV-learned weights")
    ap.add_argument("--exclude_newref", action="store_true", default=True,
                    help="Exclude newRef_137 compounds from analysis")
    args = ap.parse_args()

    print("=" * 70)
    print("CWRA - Calibrated Weighted Rank Aggregation")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(args.csv)
    df["murcko"] = df["smiles"].map(murcko_smiles)
    
    # Filter pool (exclude newRef_137)
    df_pool = df[~df["source"].isin(["newRef_137"])].reset_index(drop=True)
    N = len(df_pool)
    E = len(MODALITIES)
    
    print(f"Pool: {N} compounds, {E} modalities")
    
    # Extract modality info
    mod_cols = [m[0] for m in MODALITIES]
    mod_high_better = [m[1] for m in MODALITIES]
    mod_labels = [m[2] for m in MODALITIES]
    mod_names = [m[3] for m in MODALITIES]
    
    # Validate columns
    missing = [c for c in mod_cols if c not in df_pool.columns]
    if missing:
        raise RuntimeError(f"Missing modality columns: {missing}")
    
    # Define actives
    active_mask = df_pool["source"].isin(["initial_370", "calcitriol"]).to_numpy()
    act_idx_all = np.where(active_mask)[0]
    A_all = int(active_mask.sum())
    print(f"Actives: {A_all} (initial_370 + calcitriol)")
    
    g_mask = df_pool["source"].isin(["G1", "G2", "G3"]).to_numpy()
    print(f"Generated: {g_mask.sum()}")
    
    # Normalize scores: 0 = best, 1 = worst
    scores = np.empty((N, E))
    for j, (col, high_better) in enumerate(zip(mod_cols, mod_high_better)):
        x = df_pool[col].to_numpy(float)
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        s = (x - xmin) / (xmax - xmin) if xmax != xmin else np.zeros_like(x)
        scores[:, j] = 1.0 - s if high_better else s
    
    # Compute ranks
    ranks = np.empty((N, E), dtype=int)
    orders = []
    for j in range(E):
        order = np.argsort(scores[:, j])
        orders.append(order)
        r = np.empty(N, dtype=int)
        r[order] = np.arange(1, N + 1)
        ranks[:, j] = r
    ranks01 = (ranks - 1) / (N - 1)
    
    # Kendall tau correlation
    tau = np.eye(E)
    for i in range(E):
        for jj in range(i + 1, E):
            t, _ = kendalltau(ranks[:, i], ranks[:, jj])
            tau[i, jj] = tau[jj, i] = t if np.isfinite(t) else 0.0
    
    # Cutoffs
    cutoffs = {str(p): max(1, math.ceil(p * N / 100)) for p in CUTOFF_PCTS}
    print(f"Cutoffs: {cutoffs}")
    
    # Precompute top-k indices
    topk_idx = {name: np.stack([orders[j][:k] for j in range(E)], axis=1) 
                for name, k in cutoffs.items()}
    
    # Grid and shrinkage cache
    grid_raw = get_grid(args.grid_mode)
    grid = flatten_grid(grid_raw)
    print(f"Grid mode: {args.grid_mode}, size: {len(grid)}")
    
    tau0_vals = sorted(set(p[6] for p in grid))
    lam_vals = sorted(set(p[7] for p in grid))
    shrink_cache = {(t, l): shrink_factors(tau, t, l) for t in tau0_vals for l in lam_vals}
    
    objective_fn = {"early": obj_early, "balanced": obj_balanced, "standard": obj_standard, "comprehensive": obj_comprehensive}[args.focus]
    
    # CV setup
    act_groups = df_pool.loc[act_idx_all, "murcko"].fillna("UNKNOWN").to_numpy()
    n_scaf = len(np.unique(act_groups))
    outer_splits = min(args.outer_splits, n_scaf, len(act_groups))
    
    uniform_w = np.ones(E) / E
    
    # Fixed weights mode - skip CV
    if args.fixed_weights:
        print(f"\nUsing fixed optimal weights (skipping CV)")
        print("=" * 70)
        
        # Create fixed weight vector
        w_final = np.array([OPTIMAL_FIXED_WEIGHTS[col] for col in mod_cols])
        w_final = w_final / w_final.sum()
        
        print("\nFixed weights:")
        for name, w in sorted(zip(mod_names, w_final), key=lambda x: -x[1]):
            if w > 0.001:
                print(f"  {name}: {w:.4f}")
        
        # hyperparameters used to derive optimal weights
        # OPTIMAL_FIXED_WEIGHTS
        final_params = {
            "mode": "fixed_optimal_weights",
            "alpha": 20.0,
            "tau0": 0.15,
            "lam": 0.25,
            "delta": 2.0,
            "gamma": 0.001,
            "w_ef1": 0.5,
            "w_ef5": 0.3,
            "w_ef10": 0.2,
            "w_bedroc": 0.45,
            "w_rank": 0.1,
        }
        
        # Compute metrics on full dataset
        ef_terms, mean_rank, rank_score, bedroc_vals = calc_modality_metrics(
            act_idx_all, ranks, ranks01, topk_idx, cutoffs, N, 20.0)
        
        # Compute full-dataset EF for CWRA and baselines
        test_mask_full = active_mask  # Evaluate on all actives
        
        # CWRA score
        sc_cwra = ranks01.dot(w_final)
        d_cwra_full = eval_at_cutoffs(sc_cwra, test_mask_full, cutoffs, N)
        
        # Equal weight
        sc_eq = ranks01.mean(axis=1)
        d_eq_full = eval_at_cutoffs(sc_eq, test_mask_full, cutoffs, N)
        
        # Random (expected value)
        d_rand_full = {f"ef{c}": 1.0 for c in cutoffs}
        d_rand_full.update({f"h{c}": cutoffs[c] * A_all / N for c in cutoffs})
        
        # Individual modalities
        d_ind_full = {}
        for j, mod in enumerate(mod_names):
            sc = ranks01[:, j]
            d_ind_full[mod] = eval_at_cutoffs(sc, test_mask_full, cutoffs, N)
        
        # Set results (full dataset, no std dev)
        fold_results = []
        s_cwra = {f"ef{c}": (d_cwra_full[f"ef{c}"], 0.0) for c in cutoffs}
        s_cwra.update({f"h{c}": (d_cwra_full[f"h{c}"], 0.0) for c in cutoffs})
        
        s_eq = {f"ef{c}": (d_eq_full[f"ef{c}"], 0.0) for c in cutoffs}
        s_eq.update({f"h{c}": (d_eq_full[f"h{c}"], 0.0) for c in cutoffs})
        
        s_random = {f"ef{c}": (1.0, 0.0) for c in cutoffs}
        s_random.update({f"h{c}": (cutoffs[c] * A_all / N, 0.0) for c in cutoffs})
        
        s_individual = {}
        for mod in mod_names:
            s_individual[mod] = {f"ef{c}": (d_ind_full[mod][f"ef{c}"], 0.0) for c in cutoffs}
            s_individual[mod].update({f"h{c}": (d_ind_full[mod][f"h{c}"], 0.0) for c in cutoffs})
        
    else:
        print(f"\nCV: {args.outer_repeats} repeats × {outer_splits} outer folds")
        print("=" * 70)
        
        fold_results = []
        chosen_params = []
        
        # Nested CV
        for rep in range(args.outer_repeats):
            rng = np.random.default_rng(args.seed + rep)
            splits = balanced_group_kfold(act_groups, outer_splits, rng)
            
            for fold, (tr_idx, te_idx) in enumerate(splits, 1):
                train_act = act_idx_all[tr_idx]
                test_act = act_idx_all[te_idx]
                if len(test_act) == 0 or len(train_act) < 2:
                    continue
                
                train_groups = act_groups[tr_idx]
                n_inner = min(args.inner_splits, len(np.unique(train_groups)), len(train_act))
                
                # Inner CV pairs
                if len(np.unique(train_groups)) >= n_inner:
                    inner_cv = GroupKFold(n_splits=n_inner)
                    inner_pairs = [(train_act[itr], train_act[iva]) 
                                   for itr, iva in inner_cv.split(train_act, groups=train_groups)]
                else:
                    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=args.seed + rep)
                    inner_pairs = [(train_act[itr], train_act[iva]) 
                                   for itr, iva in inner_cv.split(train_act)]
                
                inner_pairs = [(tr, va) for tr, va in inner_pairs if len(va) >= 1]
                if not inner_pairs:
                    continue
                
                # Parallel grid search
                results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                    delayed(eval_inner)(p, inner_pairs, ranks, ranks01, topk_idx, cutoffs, 
                                        N, shrink_cache, objective_fn, E)
                    for p in grid
                )
                
                best_score, best_param = -1e18, None
                for p, m, s in results:
                    score = m - args.risk_beta * s
                    if score > best_score:
                        best_score, best_param = score, p
                
                chosen_params.append(best_param)
                
                # Evaluate outer fold
                params = dict(zip(['w_ef1', 'w_ef5', 'w_ef10', 'w_bedroc', 'w_rank', 
                                   'alpha', 'tau0', 'lam', 'delta', 'gamma'], best_param))
                shrink = shrink_cache[(params['tau0'], params['lam'])]
                
                ef_terms, _, rank_score, bedroc_vals = calc_modality_metrics(
                    train_act, ranks, ranks01, topk_idx, cutoffs, N, params['alpha'])
                w_mix = compute_weights(ef_terms, rank_score, bedroc_vals, shrink, params, E)
                
                # CWRA score
                sc_cwra = ranks01.dot(w_mix)
                sc_cwra[train_act] = np.inf
                test_mask = np.zeros(N, bool)
                test_mask[test_act] = True
                d_cwra = eval_at_cutoffs(sc_cwra, test_mask, cutoffs, N)
                
                # Baselines
                sc_eq = ranks01.dot(uniform_w)
                sc_eq[train_act] = np.inf
                d_eq = eval_at_cutoffs(sc_eq, test_mask, cutoffs, N)
                
                sc_rand = rng.random(N)
                sc_rand[train_act] = np.inf
                d_rand = eval_at_cutoffs(sc_rand, test_mask, cutoffs, N)
                
                # Individual modalities
                d_ind = {}
                for j in range(E):
                    sc = ranks01[:, j].copy()
                    sc[train_act] = np.inf
                    d_ind[mod_names[j]] = eval_at_cutoffs(sc, test_mask, cutoffs, N)
                
                fold_results.append({
                    "rep": rep + 1, "fold": fold, "params": params,
                    "cwra": d_cwra, "equal": d_eq, "random": d_rand,
                    "individual": d_ind, "weights": w_mix, "A_test": len(test_act)
                })
                
                print(f"[R{rep+1}F{fold}] CWRA EF@1%={d_cwra['ef1']:.2f} EF@5%={d_cwra['ef5']:.2f} "
                      f"EF@10%={d_cwra['ef10']:.2f} | Eq={d_eq['ef10']:.2f}")
        
        if not fold_results:
            raise RuntimeError("No valid CV folds completed")
        
        # Final parameters (majority vote)
        print("\n" + "=" * 70)
        print("Final Results")
        print("=" * 70)
        
        arr = np.array(chosen_params, dtype=object)
        final_param = tuple(Counter(arr[:, j]).most_common(1)[0][0] for j in range(arr.shape[1]))
        final_params = dict(zip(['w_ef1', 'w_ef5', 'w_ef10', 'w_bedroc', 'w_rank', 
                                 'alpha', 'tau0', 'lam', 'delta', 'gamma'], final_param))
        
        print("\nSelected hyperparameters:")
        for k, v in final_params.items():
            print(f"  {k}: {v}")
        
        # Compute final weights
        shrink = shrink_cache[(final_params['tau0'], final_params['lam'])]
        ef_terms, mean_rank, rank_score, bedroc_vals = calc_modality_metrics(
            act_idx_all, ranks, ranks01, topk_idx, cutoffs, N, final_params['alpha'])
        w_final = compute_weights(ef_terms, rank_score, bedroc_vals, shrink, final_params, E)
        
        print("\nFinal weights:")
        for name, w in sorted(zip(mod_names, w_final), key=lambda x: -x[1]):
            print(f"  {name}: {w:.4f}")
        
        # Aggregate CV results
        def agg_results(key):
            vals = {}
            for c in cutoffs.keys():
                h_arr = np.array([fr[key][f"h{c}"] for fr in fold_results])
                ef_arr = np.array([fr[key][f"ef{c}"] for fr in fold_results])
                vals[f"h{c}"] = (h_arr.mean(), h_arr.std(ddof=1) if len(h_arr) > 1 else 0)
                vals[f"ef{c}"] = (ef_arr.mean(), ef_arr.std(ddof=1) if len(ef_arr) > 1 else 0)
            return vals
        
        s_cwra = agg_results("cwra")
        s_eq = agg_results("equal")
        s_random = agg_results("random")
        
        s_individual = {}
        for mod in mod_names:
            vals = {}
            for c in cutoffs.keys():
                h_arr = np.array([fr["individual"][mod][f"h{c}"] for fr in fold_results])
                ef_arr = np.array([fr["individual"][mod][f"ef{c}"] for fr in fold_results])
                vals[f"h{c}"] = (h_arr.mean(), h_arr.std(ddof=1) if len(h_arr) > 1 else 0)
                vals[f"ef{c}"] = (ef_arr.mean(), ef_arr.std(ddof=1) if len(ef_arr) > 1 else 0)
            s_individual[mod] = vals
        
        # Compute mean_rank for table5
        ef_terms, mean_rank, _, _ = calc_modality_metrics(
            act_idx_all, ranks, ranks01, topk_idx, cutoffs, N, final_params.get('alpha', 20.0))
    
    # Table 5: Modality weights
    table5 = pd.DataFrame({
        "Modality": mod_labels,
        "Weight": w_final,
        "EF@1%": ef_terms["1"],
        "EF@5%": ef_terms["5"],
        "EF@10%": ef_terms["10"],
        "Mean_Rank": mean_rank
    })
    
    # Table 6: Performance
    rows = []
    rows.append({"Method": "Random", 
                 **{f"EF@{c}%": f"{s_random[f'ef{c}'][0]:.2f} ± {s_random[f'ef{c}'][1]:.2f}" for c in cutoffs},
                 **{f"Hits@{c}%": f"{s_random[f'h{c}'][0]:.2f} ± {s_random[f'h{c}'][1]:.2f}" for c in cutoffs}})
    
    for mod in mod_names:
        s = s_individual[mod]
        rows.append({"Method": mod, 
                     **{f"EF@{c}%": f"{s[f'ef{c}'][0]:.2f} ± {s[f'ef{c}'][1]:.2f}" for c in cutoffs},
                     **{f"Hits@{c}%": f"{s[f'h{c}'][0]:.2f} ± {s[f'h{c}'][1]:.2f}" for c in cutoffs}})
    
    rows.append({"Method": "Equal-weight", 
                 **{f"EF@{c}%": f"{s_eq[f'ef{c}'][0]:.2f} ± {s_eq[f'ef{c}'][1]:.2f}" for c in cutoffs},
                 **{f"Hits@{c}%": f"{s_eq[f'h{c}'][0]:.2f} ± {s_eq[f'h{c}'][1]:.2f}" for c in cutoffs}})
    rows.append({"Method": "CWRA-early", 
                 **{f"EF@{c}%": f"{s_cwra[f'ef{c}'][0]:.2f} ± {s_cwra[f'ef{c}'][1]:.2f}" for c in cutoffs},
                 **{f"Hits@{c}%": f"{s_cwra[f'h{c}'][0]:.2f} ± {s_cwra[f'h{c}'][1]:.2f}" for c in cutoffs}})
    
    table6 = pd.DataFrame(rows)
    
    # Final consensus ranking
    consensus = ranks01.dot(w_final)
    df_pool["cwra_score"] = consensus
    df_pool["cwra_rank"] = np.argsort(np.argsort(consensus)) + 1
    
    # Add individual ranks
    for j, col in enumerate(mod_cols):
        df_pool[f"rank_{col}"] = ranks[:, j]
    
    # Save outputs
    prefix = args.output_prefix
    table5.to_csv(f"{prefix}_table5_weights.csv", index=False)
    table6.to_csv(f"{prefix}_table6_performance.csv", index=False)
    df_pool.to_csv(f"{prefix}_full_ranking.csv", index=False)
    
    df_gen = df_pool[g_mask].sort_values("cwra_rank")
    df_gen.head(args.top_n).to_csv(f"{prefix}_top{args.top_n}_G.csv", index=False)
    df_gen.tail(args.top_n).to_csv(f"{prefix}_bottom{args.top_n}_G.csv", index=False)
    
    # Save hyperparameters
    hp_df = pd.DataFrame([{
        "Parameter": k, 
        "Value": v
    } for k, v in final_params.items()])
    hp_df.to_csv(f"{prefix}_hyperparameters.csv", index=False)
    
    # LaTeX
    latex = generate_latex_tables(table5, s_individual, s_eq, s_cwra, s_random, 
                                   final_params, mod_labels, mod_names)
    with open(f"{prefix}_latex_tables.tex", "w") as f:
        f.write(latex)
    
    # Print summary table
    print("\n" + "=" * 110)
    print("PERFORMANCE SUMMARY")
    print("=" * 110)
    print(f"\n{'Method':<25} {'EF@1%':>15} {'EF@5%':>15} {'EF@10%':>15} {'EF@20%':>15} {'EF@30%':>15}")
    print("-" * 110)
    print(f"{'Random':<25} {fmt(*s_random['ef1']):>15} {fmt(*s_random['ef5']):>15} "
          f"{fmt(*s_random['ef10']):>15} {fmt(*s_random['ef20']):>15} {fmt(*s_random['ef30']):>15}")
    print("-" * 110)
    
    for mod in sorted(mod_names, key=lambda m: -s_individual[m]['ef10'][0]):
        s = s_individual[mod]
        print(f"{mod:<25} {fmt(*s['ef1']):>15} {fmt(*s['ef5']):>15} "
              f"{fmt(*s['ef10']):>15} {fmt(*s['ef20']):>15} {fmt(*s['ef30']):>15}")
    
    print("-" * 110)
    print(f"{'Equal-weight':<25} {fmt(*s_eq['ef1']):>15} {fmt(*s_eq['ef5']):>15} "
          f"{fmt(*s_eq['ef10']):>15} {fmt(*s_eq['ef20']):>15} {fmt(*s_eq['ef30']):>15}")
    print(f"{'CWRA-early':<25} {fmt(*s_cwra['ef1']):>15} {fmt(*s_cwra['ef5']):>15} "
          f"{fmt(*s_cwra['ef10']):>15} {fmt(*s_cwra['ef20']):>15} {fmt(*s_cwra['ef30']):>15}")
    print("=" * 110)
    
    # Improvement stats
    best_ind = max(s_individual[m]['ef10'][0] for m in mod_names)
    print(f"\nImprovement over Equal-weight: {100*(s_cwra['ef10'][0]/s_eq['ef10'][0] - 1):.1f}%")
    print(f"Improvement over best individual: {100*(s_cwra['ef10'][0]/best_ind - 1):.1f}%")
    print(f"Improvement over random: {100*(s_cwra['ef10'][0]/s_random['ef10'][0] - 1):.1f}%")
    
    # Full dataset EF
    print("\nFull dataset EF (no CV):")
    order = np.argsort(consensus)
    for c, k in cutoffs.items():
        h = active_mask[order[:k]].sum()
        ef = (h * N) / (k * A_all)
        print(f"  EF@{c}%: {ef:.2f} ({h} hits in top {k})")
    
    # Calcitriol rank
    calcitriol = df_pool[df_pool["source"] == "calcitriol"]
    if len(calcitriol) > 0:
        rank = int(calcitriol["cwra_rank"].iloc[0])
        print(f"\nCalcitriol rank: {rank} / {N} (top {100*rank/N:.1f}%)")
    
    print(f"\nOutputs saved: {prefix}_*")


if __name__ == "__main__":
    main()
