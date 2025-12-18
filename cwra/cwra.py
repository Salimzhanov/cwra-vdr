#!/usr/bin/env python3
"""
CWRA - Calibrated Weighted Rank Aggregation for VDR Virtual Screening

Key Features:
1. Comprehensive baselines: Individual modalities, Random, Equal-weight, CWRA
2. Corrected modality directions (empirically validated)
3. Stable objective function using EF@10%/20%/30% (less variance than EF@1%)
4. Focus on best-performing modalities

Changes from original:
- Fixed RRF aggregation to skip meaningless hyperparameter search
- Fixed inconsistent delta usage in power aggregation
- Optimized BEDROC computation (compute only needed alpha)
- Removed redundant operations and memory waste
- Fixed objective function documentation
- Improved data transformation consistency

Usage:
  python cwra_optimized.py --csv labeled_raw_modalities.csv --focus early
"""

from __future__ import annotations
import argparse
import math
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import kendalltau
from sklearn.model_selection import GroupKFold, KFold


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
        if scaf is None:
            return None
        return Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return None


# =============================================================================
# METRICS
# =============================================================================

def bedroc_from_x(x: np.ndarray, alpha: float, A: int, N: int) -> float:
    """Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC)."""
    if A <= 0 or N <= 1:
        return 0.0
    denom = (1.0 - np.exp(-alpha)) / alpha
    rie = np.mean(np.exp(-alpha * x)) / denom
    k = np.arange(A)
    xideal = k / (N - 1)
    rie_max = np.mean(np.exp(-alpha * xideal)) / denom
    return float((rie - 1.0) / (rie_max - 1.0)) if rie_max != 1.0 else 0.0


def shrink_factors(tau: np.ndarray, tau0: float, lam: float) -> np.ndarray:
    """Compute shrinkage factors based on correlation structure."""
    n_corr = (np.abs(tau) > tau0).sum(axis=1) - 1  # Exclude self-correlation
    return 1.0 / (1.0 + lam * n_corr)


def compute_weights_v2(
    ef_terms: dict,  # {"10": array, "20": array, "30": array}
    rank_score: np.ndarray,
    bedroc_alpha: np.ndarray,
    shrink: np.ndarray,
    delta: float,
    gamma: float,
    w_ef10: float, w_ef20: float, w_ef30: float,
    w_bedroc: float, w_rank: float,
    eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhanced weight computation focusing on EF@10% and deeper cutoffs.
    
    Parameters:
        ef_terms: Dict with keys "10", "20", "30" containing EF arrays per modality
        rank_score: Normalized rank score (1 - mean_rank/N) per modality
        bedroc_alpha: BEDROC values at chosen alpha per modality
        shrink: Shrinkage factors to penalize correlated modalities
        delta: Power exponent for weight sharpening (0 = equal, higher = sharper)
        gamma: Uniform mixing coefficient (regularization toward equal weights)
        w_ef10, w_ef20, w_ef30: Relative weights for EF depths
        w_bedroc, w_rank: Weights for BEDROC and rank components
        eps: Small constant for numerical stability
    
    Returns:
        w_mix: Final mixed weights (with uniform regularization)
        w: Raw weights (before mixing)
    """
    E = rank_score.shape[0]
    uniform = np.ones(E) / E
    
    # Weighted combination of EF focusing on EF@10% and deeper cutoffs
    # Note: Keys are "10", "20", "30" without "ef" prefix
    ef_combined = (w_ef10 * ef_terms.get("10", np.zeros(E)) +
                   w_ef20 * ef_terms.get("20", np.zeros(E)) +
                   w_ef30 * ef_terms.get("30", np.zeros(E)))
    ef_sum = w_ef10 + w_ef20 + w_ef30
    if ef_sum > 0:
        ef_combined /= ef_sum
    
    # Combine components: BEDROC + rank + EF
    ef_weight = 1.0 - w_bedroc - w_rank
    raw = w_bedroc * bedroc_alpha + w_rank * rank_score + ef_weight * ef_combined
    raw = np.maximum(raw, 0.0) + eps
    
    # Apply shrinkage and power transformation
    w = (raw * shrink) ** delta
    w = w / w.sum()
    
    # Mix with uniform for regularization
    w_mix = (1.0 - gamma) * w + gamma * uniform
    w_mix = w_mix / w_mix.sum()
    
    return w_mix, w


def reciprocal_rank_fusion(ranks: np.ndarray, k: float = 60.0) -> np.ndarray:
    """
    Reciprocal Rank Fusion (RRF) - robust rank aggregation.
    
    Lower returned score = better (for consistency with other methods).
    """
    N, E = ranks.shape
    scores = np.zeros(N, dtype=float)
    for j in range(E):
        scores += 1.0 / (k + ranks[:, j])
    return -scores  # Negate so lower = better


def power_rank_transform(ranks: np.ndarray, N: int, power: float = 0.5) -> np.ndarray:
    """
    Power-law rank transformation for early enrichment focus.
    
    Lower power emphasizes top ranks more.
    """
    normalized = (ranks - 1) / (N - 1)
    return normalized ** power


def eval_depths_extended(
    score: np.ndarray,
    eval_mask: np.ndarray,
    A_eval: int,
    cutoffs: dict,  # {"10": 161, "20": 321, "30": 481}
    N: int
) -> dict:
    """
    Evaluate hits and enrichment at multiple depth cutoffs.
    
    Parameters:
        score: Compound scores (lower = better)
        eval_mask: Boolean mask for active compounds to evaluate
        A_eval: Number of actives in evaluation set
        cutoffs: Dict mapping cutoff names to k values
        N: Total number of compounds
    
    Returns:
        Dict with h{cutoff} (hits) and ef{cutoff} (enrichment factor) for each cutoff
    """
    if A_eval == 0:
        return {f"h{k}": 0 for k in cutoffs} | {f"ef{k}": 0.0 for k in cutoffs}
    
    kmax = max(cutoffs.values())
    # Get top-k indices efficiently
    top = np.argpartition(score, kmax - 1)[:kmax]
    top = top[np.argsort(score[top])]

    result = {}
    for name, k in cutoffs.items():
        h = int(eval_mask[top[:k]].sum())
        ef = (h * N) / (k * A_eval)
        result[f"h{name}"] = h
        result[f"ef{name}"] = ef
    
    return result


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def objective_early_focus(d: dict) -> float:
    """
    Objective emphasizing EF@10% performance for stable weight assignment.
    Focus on EF@10% which provides good balance between early enrichment and stability.
    """
    return d.get("ef10", 0)


def objective_balanced(d: dict) -> float:
    """Balanced objective across EF@10%, EF@20%, EF@30%."""
    return 0.6 * d.get("ef10", 0) + 0.3 * d.get("ef20", 0) + 0.1 * d.get("ef30", 0)


def objective_standard(d: dict) -> float:
    """Standard objective with equal emphasis on 10% and 20% cutoffs."""
    return 0.4 * d.get("ef10", 0) + 0.4 * d.get("ef20", 0) + 0.2 * d.get("ef30", 0)


# =============================================================================
# VARIANCE-REDUCED CV SPLITTING
# =============================================================================

def balanced_group_kfold_indices(
    groups: np.ndarray, 
    n_splits: int, 
    rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create balanced group k-fold splits with similar sizes per fold.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.arange(len(uniq))
    rng.shuffle(order)

    # Greedy bin packing to balance fold sizes
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for g in order:
        j = int(np.argmin(fold_sizes))
        fold_groups[j].append(g)
        fold_sizes[j] += int(sizes[g])

    splits = []
    for j in range(n_splits):
        test_group_ids = set(fold_groups[j])
        te = np.where(np.array([gid in test_group_ids for gid in inv]))[0]
        tr = np.setdiff1d(np.arange(len(groups)), te, assume_unique=False)
        splits.append((tr, te))
    return splits


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calc_metrics(
    active_idx: np.ndarray,
    ranks: np.ndarray,
    ranks01: np.ndarray,
    topk_idx: dict,
    cutoffs: dict,
    N: int,
    alpha: float  # Single alpha value instead of list
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate metrics for active compounds.
    
    Returns:
        ef_terms: Dict with EF values per modality at each cutoff (keys: "10", "20", "30")
        mean_rank: Mean rank per modality
        rank_score: Normalized rank score per modality
        bedroc: BEDROC values per modality at specified alpha
    """
    A = len(active_idx)
    E = ranks.shape[1]
    
    active_bool = np.zeros(N, bool)
    active_bool[active_idx] = True

    # Compute EF at each cutoff for each modality
    ef_terms = {}
    for name, idx_arr in topk_idx.items():
        hits = active_bool[idx_arr].sum(axis=0)  # Hits per modality
        k = cutoffs[name]
        ef_terms[name] = (hits * N) / (k * A) if A > 0 else np.zeros(E)

    # Rank-based metrics
    r_act = ranks[active_idx, :]
    mean_rank = r_act.mean(axis=0)
    rank_score = 1.0 - (mean_rank - 1.0) / (N - 1)

    # BEDROC at specified alpha
    x_act = ranks01[active_idx, :]
    bedroc = np.empty(E, float)
    for j in range(E):
        bedroc[j] = bedroc_from_x(x_act[:, j], alpha, A, N)

    return ef_terms, mean_rank, rank_score, bedroc


def fmt(m: float, sd: float, digits: int = 2) -> str:
    """Format mean ± std."""
    return f"{m:.{digits}f} +/- {sd:.{digits}f}"


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="CWRA for VDR virtual screening toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--csv", default="labeled_raw_modalities.csv",
                    help="Input CSV with modalities + smiles + source")
    ap.add_argument("--outer_splits", type=int, default=10,
                    help="Number of outer CV folds")
    ap.add_argument("--outer_repeats", type=int, default=5,
                    help="Number of outer CV repeats")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    ap.add_argument("--risk_beta", type=float, default=0.5,
                    help="Risk aversion: mean(obj) - beta*std(obj)")
    ap.add_argument("--focus", type=str, default="early",
                    choices=["early", "balanced", "standard"],
                    help="Optimization focus")
    ap.add_argument("--aggregation", type=str, default="weighted",
                    choices=["weighted", "rrf", "power"],
                    help="Aggregation method: weighted ranks, RRF, or power-transformed")
    ap.add_argument("--output_prefix", type=str, default="cwrad",
                    help="Prefix for output files")
    ap.add_argument("--top_n", type=int, default=25,
                    help="Number of top/bottom structures to extract")
    args = ap.parse_args()

    print("=" * 70)
    print("CWRA - VDR Virtual Screening Toolbox (Optimized)")
    print("=" * 70)
    print(f"Focus: {args.focus}")
    print(f"Aggregation: {args.aggregation}")
    
    # Load data
    df = pd.read_csv(args.csv)
    print(f"\nLoaded {len(df)} compounds from {args.csv}")

    # Validate required columns
    if "smiles" not in df.columns:
        raise RuntimeError("CSV must contain a 'smiles' column")
    if "source" not in df.columns:
        raise RuntimeError("CSV must contain a 'source' column")

    # Compute scaffolds
    df["murcko"] = df["smiles"].map(murcko_smiles)
    n_scaffolds = df["murcko"].nunique()
    print(f"Computed {n_scaffolds} unique Murcko scaffolds")

    # Define candidate pool (exclude reference compounds)
    df_pool = df.loc[~df["source"].isin(["newRef_137", "calcitriol"])].reset_index(drop=True)
    print(f"Candidate pool: {len(df_pool)} compounds")

    # =========================================================================
    # MODALITY DEFINITIONS - CORRECTED ORIENTATIONS
    # =========================================================================
    # high_better=False: Lower raw values → better compound → rank 1
    # high_better=True:  Higher raw values → better compound → rank 1
    modalities = [
        ("graphdta_kd", False),      # Lower Kd prediction = stronger binding = active
        ("graphdta_ki", False),      # Lower Ki prediction = stronger binding = active
        ("graphdta_ic50", False),    # Lower IC50 prediction = more potent = active
        ("mltle_pKd", False),        # Lower pKd prediction = active (empirical)
        ("vina_score", False),       # Lower docking score = better binding
        ("boltz_affinity", False),   # Lower affinity = stronger binding
        ("boltz_confidence", True),  # Higher confidence = better prediction
        ("unimol_similarity", True), # Higher similarity to actives = better
        ("tankbind_affinity", False),# Lower affinity = better binding
        ("drugban_affinity", False), # Lower affinity = better binding
        ("moltrans_affinity", False),# Lower affinity = better binding
    ]
    
    mod_labels = [
        r"GraphDTA $K_d$", r"GraphDTA $K_i$", r"GraphDTA IC$_{50}$",
        r"MLT-LE $pK_d$", r"AutoDock Vina", r"Boltz-2 affinity",
        r"Boltz-2 confidence", r"Uni-Mol similarity", r"TankBind affinity",
        r"DrugBAN affinity", r"MolTrans affinity",
    ]
    
    mod_names_simple = [
        "GraphDTA_Kd", "GraphDTA_Ki", "GraphDTA_IC50",
        "MLTLE_pKd", "Vina", "Boltz_affinity", "Boltz_confidence",
        "UniMol_sim", "TankBind", "DrugBAN", "MolTrans"
    ]
    
    # Validate modality columns exist
    missing_cols = [col for col, _ in modalities if col not in df_pool.columns]
    if missing_cols:
        raise RuntimeError(f"Missing modality columns: {missing_cols}")
    
    E = len(modalities)
    N = len(df_pool)

    # Identify actives
    active_mask = (df_pool["source"] == "initial_370").to_numpy()
    act_idx_all = np.where(active_mask)[0]
    A_all = int(active_mask.sum())
    print(f"Actives (initial_370): {A_all}")

    if A_all == 0:
        raise RuntimeError("No actives found in dataset (source='initial_370')")

    # Generated compounds mask
    g_mask = df_pool["source"].isin(["G1", "G2", "G3"]).to_numpy()
    print(f"Generated compounds (G1/G2/G3): {g_mask.sum()}")

    # =========================================================================
    # NORMALIZE MODALITIES
    # =========================================================================
    # Transform all modalities to [0, 1] where 0 = best, 1 = worst
    scores = np.empty((N, E), float)
    for j, (col, high_better) in enumerate(modalities):
        x = df_pool[col].to_numpy(float)
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        if xmax != xmin:
            s = (x - xmin) / (xmax - xmin)
        else:
            s = np.zeros_like(x)
        # If high_better, invert so that high original → low normalized
        if high_better:
            s = 1.0 - s
        scores[:, j] = s

    # =========================================================================
    # COMPUTE RANKS
    # =========================================================================
    # Rank 1 = best (lowest normalized score)
    ranks = np.empty((N, E), int)
    orders = []  # Store sorted indices for each modality
    for j in range(E):
        order = np.argsort(scores[:, j], kind="mergesort")
        orders.append(order)
        r = np.empty(N, int)
        r[order] = np.arange(1, N + 1)
        ranks[:, j] = r
    
    # Normalized ranks: rank 1 → 0, rank N → 1
    ranks01 = (ranks - 1) / (N - 1)

    # =========================================================================
    # KENDALL TAU CORRELATION MATRIX
    # =========================================================================
    tau = np.eye(E)
    for i in range(E):
        for j in range(i + 1, E):
            t, _ = kendalltau(ranks[:, i], ranks[:, j])
            tau[i, j] = tau[j, i] = t if np.isfinite(t) else 0.0

    print(f"\nKendall tau correlation matrix:")
    print(pd.DataFrame(tau, index=mod_names_simple, columns=mod_names_simple).round(3))

    # =========================================================================
    # CUTOFF DEPTHS
    # =========================================================================
    cutoffs = {
        "10": math.ceil(0.10 * N),
        "20": math.ceil(0.20 * N),
        "30": math.ceil(0.30 * N),
    }
    k100 = min(100, N)
    cutoffs["100"] = k100
    
    k10, k20, k30 = cutoffs["10"], cutoffs["20"], cutoffs["30"]
    print(f"\nCutoffs: {cutoffs}")

    # Precompute top-k indices for each modality
    topk_idx = {}
    for name, k in cutoffs.items():
        # Shape: (k, E) - top k indices for each modality
        topk_idx[name] = np.stack([orders[j][:k] for j in range(E)], axis=1)

    # Select objective function
    objective_fn = {
        "early": objective_early_focus,
        "balanced": objective_balanced,
        "standard": objective_standard,
    }[args.focus]
    print(f"\nUsing {args.focus.upper()} objective function")

    # =========================================================================
    # HYPERPARAMETER GRID
    # =========================================================================
    alphas = [20.0, 40.0, 80.0]
    
    # EF depth weight combinations (w_ef10, w_ef20, w_ef30)
    ef_weight_combos = [
        (0.7, 0.2, 0.1),
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
        (0.3, 0.4, 0.3),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    
    # BEDROC and rank weight combinations (w_bedroc, w_rank)
    bedroc_rank_combos = [
        (0.4, 0.1),
        (0.5, 0.1),
        (0.3, 0.2),
        (0.6, 0.05),
        (0.0, 0.0),  # Pure EF based
    ]
    
    delta_list = [0.0, 0.5, 1.0, 1.5]
    gamma_list = [0.05, 0.1, 0.2]
    tau0_list = [0.3, 0.4, 0.5]
    lam_list = [0.0, 0.25, 0.5]

    # For power aggregation, use separate power parameter
    power_list = [0.3, 0.5, 0.7, 1.0]

    # Build grid based on aggregation method
    if args.aggregation == "rrf":
        # RRF doesn't use learned weights - skip hyperparameter search
        grid = [(None, None, None, None, None, None, None, None, None, None)]
        print(f"\nRRF aggregation: No hyperparameter search needed")
    elif args.aggregation == "power":
        # Power aggregation: power parameter replaces delta for transformation
        grid = [
            (w_ef10, w_ef20, w_ef30, w_bed, w_rank, alpha, tau0, lam, power, gamma)
            for (w_ef10, w_ef20, w_ef30), (w_bed, w_rank), alpha, tau0, lam, power, gamma
            in itertools.product(ef_weight_combos, bedroc_rank_combos, alphas, 
                               tau0_list, lam_list, power_list, gamma_list)
        ]
        print(f"\nPower aggregation grid size: {len(grid)} configurations")
    else:
        # Standard weighted aggregation
        grid = [
            (w_ef10, w_ef20, w_ef30, w_bed, w_rank, alpha, tau0, lam, delta, gamma)
            for (w_ef10, w_ef20, w_ef30), (w_bed, w_rank), alpha, tau0, lam, delta, gamma
            in itertools.product(ef_weight_combos, bedroc_rank_combos, alphas, 
                               tau0_list, lam_list, delta_list, gamma_list)
        ]
        print(f"\nWeighted aggregation grid size: {len(grid)} configurations")

    # Pre-compute shrinkage factors
    shrink_cache = {
        (tau0, lam): shrink_factors(tau, tau0, lam) 
        for tau0 in tau0_list for lam in lam_list
    }

    # Scaffold groups for actives
    act_groups = df_pool.loc[act_idx_all, "murcko"].fillna("UNKNOWN").to_numpy()

    rng = np.random.default_rng(args.seed)
    uniform_w = np.ones(E) / E

    fold_results = []
    chosen_params = []

    print("\n" + "=" * 70)
    print("Running Nested Cross-Validation")
    print("=" * 70)

    # =========================================================================
    # NESTED CROSS-VALIDATION
    # =========================================================================
    for rep in range(args.outer_repeats):
        rep_seed = int(args.seed + rep)
        rep_rng = np.random.default_rng(rep_seed)

        outer_splits = balanced_group_kfold_indices(
            act_groups, n_splits=args.outer_splits, rng=rep_rng
        )

        for fold, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
            train_act = act_idx_all[tr_idx]
            test_act = act_idx_all[te_idx]
            train_groups = act_groups[tr_idx]

            unique_groups = len(np.unique(train_groups))
            n_inner = min(5, max(2, unique_groups))

            if unique_groups >= n_inner:
                inner = GroupKFold(n_splits=n_inner)
                inner_pairs = [
                    (train_act[itr], train_act[iva]) 
                    for itr, iva in inner.split(train_act, groups=train_groups)
                ]
            else:
                inner = KFold(n_splits=n_inner, shuffle=True, random_state=rep_seed)
                inner_pairs = [
                    (train_act[itr], train_act[iva]) 
                    for itr, iva in inner.split(train_act)
                ]

            # =================================================================
            # INNER CV FOR HYPERPARAMETER SELECTION
            # =================================================================
            if args.aggregation == "rrf":
                # RRF: No hyperparameters to tune, skip inner CV
                best_param = (None,) * 10
            else:
                best_param = None
                best_score = -1e18
                beta = float(args.risk_beta)

                for params in grid:
                    (w_ef10, w_ef20, w_ef30, w_bed, w_rank, 
                     alpha, tau0, lam, delta_or_power, gamma) = params
                    
                    shrink = shrink_cache[(tau0, lam)]
                    objs = []
                    
                    for inner_tr, inner_va in inner_pairs:
                        # Compute metrics on inner training set
                        ef_terms_tr, _, rank_score_tr, bedroc_tr = calc_metrics(
                            inner_tr, ranks, ranks01, topk_idx, cutoffs, N, alpha
                        )
                        
                        # Compute weights
                        w_mix, _ = compute_weights_v2(
                            ef_terms_tr, rank_score_tr, bedroc_tr,
                            shrink, delta_or_power if args.aggregation == "weighted" else 1.0,
                            gamma, w_ef10, w_ef20, w_ef30, w_bed, w_rank
                        )
                        
                        # Compute consensus score
                        if args.aggregation == "power":
                            ranks_power = power_rank_transform(ranks, N, power=delta_or_power)
                            sc = ranks_power.dot(w_mix)
                        else:  # weighted
                            sc = ranks01.dot(w_mix)
                        
                        # Mask training actives
                        sc[inner_tr] = np.inf
                        
                        # Evaluate on validation
                        eval_mask = np.zeros(N, bool)
                        eval_mask[inner_va] = True
                        A_va = len(inner_va)
                        
                        d = eval_depths_extended(sc, eval_mask, A_va, cutoffs, N)
                        objs.append(objective_fn(d))
                    
                    # Risk-adjusted score
                    m = float(np.mean(objs))
                    s = float(np.std(objs, ddof=1)) if len(objs) > 1 else 0.0
                    score = m - beta * s
                    
                    if score > best_score:
                        best_score = score
                        best_param = params

            chosen_params.append(best_param)

            # =================================================================
            # OUTER FOLD EVALUATION
            # =================================================================
            test_mask = np.zeros(N, bool)
            test_mask[test_act] = True
            A_test = len(test_act)

            # Compute CWRA score
            if args.aggregation == "rrf":
                sc_cwra = reciprocal_rank_fusion(ranks)
                w_mix = uniform_w  # RRF implicitly uses equal weights
            else:
                (w_ef10, w_ef20, w_ef30, w_bed, w_rank,
                 alpha, tau0, lam, delta_or_power, gamma) = best_param
                
                shrink = shrink_cache[(tau0, lam)]
                ef_terms_tr, _, rank_score_tr, bedroc_tr = calc_metrics(
                    train_act, ranks, ranks01, topk_idx, cutoffs, N, alpha
                )
                w_mix, _ = compute_weights_v2(
                    ef_terms_tr, rank_score_tr, bedroc_tr,
                    shrink, delta_or_power if args.aggregation == "weighted" else 1.0,
                    gamma, w_ef10, w_ef20, w_ef30, w_bed, w_rank
                )
                
                if args.aggregation == "power":
                    ranks_power = power_rank_transform(ranks, N, power=delta_or_power)
                    sc_cwra = ranks_power.dot(w_mix)
                else:
                    sc_cwra = ranks01.dot(w_mix)

            sc_cwra[train_act] = np.inf
            d_cwra = eval_depths_extended(sc_cwra, test_mask, A_test, cutoffs, N)

            # Equal weight baseline
            sc_eq = ranks01.dot(uniform_w)
            sc_eq[train_act] = np.inf
            d_eq = eval_depths_extended(sc_eq, test_mask, A_test, cutoffs, N)

            # Random baseline
            rng_fold = np.random.default_rng(rep_seed + fold)
            sc_random = rng_fold.random(N)
            sc_random[train_act] = np.inf
            d_random = eval_depths_extended(sc_random, test_mask, A_test, cutoffs, N)

            # Individual modality baselines
            individual_results = {}
            for j in range(E):
                sc_ind = ranks01[:, j].copy()
                sc_ind[train_act] = np.inf
                d_ind = eval_depths_extended(sc_ind, test_mask, A_test, cutoffs, N)
                individual_results[mod_names_simple[j]] = d_ind

            fold_results.append({
                "rep": rep + 1, "fold": fold, "best_param": best_param,
                "cwra": d_cwra, "equal": d_eq, "random": d_random,
                "individual": individual_results,
                "weights": w_mix, "A_test": A_test,
            })

            print(f"[Rep {rep+1} Fold {fold:2d}] CWRA EF@10%={d_cwra['ef10']:.2f} "
                  f"EF@20%={d_cwra['ef20']:.2f} EF@30%={d_cwra['ef30']:.2f}  "
                  f"Equal EF@10%={d_eq['ef10']:.2f}  Random EF@10%={d_random['ef10']:.2f}")

    # =========================================================================
    # FINAL MODEL & RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("Computing Final Results")
    print("=" * 70)

    # Determine final hyperparameters
    if args.aggregation == "rrf":
        final_param = (None,) * 10
        alpha_final = 20.0  # Default for metrics computation
        w_mix_full = uniform_w
        print("\nRRF aggregation: Using uniform implicit weights")
    else:
        # Majority vote for hyperparameters
        arr = np.array(chosen_params, dtype=object)
        final_param = tuple(Counter(arr[:, j]).most_common(1)[0][0] for j in range(arr.shape[1]))
        (w_ef10, w_ef20, w_ef30, w_bed, w_rank,
         alpha_final, tau0, lam, delta_or_power, gamma) = final_param
        shrink = shrink_cache[(tau0, lam)]

        print(f"\nFinal hyperparameters (majority vote):")
        print(f"  EF weights: w_ef10={w_ef10}, w_ef20={w_ef20}, w_ef30={w_ef30}")
        print(f"  BEDROC/rank: w_bed={w_bed}, w_rank={w_rank}")
        print(f"  alpha={alpha_final}, tau0={tau0}, lambda={lam}")
        if args.aggregation == "power":
            print(f"  power={delta_or_power}, gamma={gamma}")
        else:
            print(f"  delta={delta_or_power}, gamma={gamma}")

        # Final weights
        ef_terms_full, mean_rank_full, rank_score_full, bedroc_full = calc_metrics(
            act_idx_all, ranks, ranks01, topk_idx, cutoffs, N, alpha_final
        )
        w_mix_full, _ = compute_weights_v2(
            ef_terms_full, rank_score_full, bedroc_full,
            shrink, delta_or_power if args.aggregation == "weighted" else 1.0,
            gamma, w_ef10, w_ef20, w_ef30, w_bed, w_rank
        )

    print(f"\nFinal modality weights:")
    for name, w in zip(mod_names_simple, w_mix_full):
        print(f"  {name}: {w:.4f}")

    # =========================================================================
    # TABLE 5: WEIGHTS
    # =========================================================================
    # Compute metrics on full dataset for table
    ef_terms_full, mean_rank_full, rank_score_full, bedroc_full = calc_metrics(
        act_idx_all, ranks, ranks01, topk_idx, cutoffs, N, alpha_final
    )
    
    table5 = pd.DataFrame({
        "Modality": mod_labels,
        "Weight": w_mix_full,
        "EF@10%": ef_terms_full["10"],
        "EF@20%": ef_terms_full["20"],
        "EF@30%": ef_terms_full["30"],
        "Mean_Rank": mean_rank_full,
    })

    # =========================================================================
    # TABLE 6: PERFORMANCE
    # =========================================================================
    def agg(method_key: str) -> dict:
        """Aggregate metrics across folds."""
        vals = {}
        for name in cutoffs.keys():
            h_arr = np.array([fr[method_key][f"h{name}"] for fr in fold_results], float)
            ef_arr = np.array([fr[method_key][f"ef{name}"] for fr in fold_results], float)
            vals[f"h{name}"] = (float(h_arr.mean()), float(h_arr.std(ddof=1)))
            vals[f"ef{name}"] = (float(ef_arr.mean()), float(ef_arr.std(ddof=1)))
        return vals

    s_eq = agg("equal")
    s_cwra = agg("cwra")
    s_random = agg("random")
    
    # Aggregate individual modality results
    s_individual = {}
    for mod_name in mod_names_simple:
        vals = {}
        for name in cutoffs.keys():
            h_arr = np.array([fr["individual"][mod_name][f"h{name}"] for fr in fold_results], float)
            ef_arr = np.array([fr["individual"][mod_name][f"ef{name}"] for fr in fold_results], float)
            vals[f"h{name}"] = (float(h_arr.mean()), float(h_arr.std(ddof=1)))
            vals[f"ef{name}"] = (float(ef_arr.mean()), float(ef_arr.std(ddof=1)))
        s_individual[mod_name] = vals

    # Build comparison table
    table6_rows = []
    
    table6_rows.append({
        "Method": "Random",
        "EF@10%": fmt(*s_random["ef10"]),
        "EF@20%": fmt(*s_random["ef20"]),
        "EF@30%": fmt(*s_random["ef30"]),
        f"Hits@20% ({k20})": fmt(*s_random["h20"], digits=1),
        f"Hits@30% ({k30})": fmt(*s_random["h30"], digits=1),
    })
    
    for mod_name in mod_names_simple:
        s = s_individual[mod_name]
        table6_rows.append({
            "Method": f"Individual: {mod_name}",
            "EF@10%": fmt(*s["ef10"]),
            "EF@20%": fmt(*s["ef20"]),
            "EF@30%": fmt(*s["ef30"]),
            f"Hits@20% ({k20})": fmt(*s["h20"], digits=1),
            f"Hits@30% ({k30})": fmt(*s["h30"], digits=1),
        })
    
    table6_rows.append({
        "Method": "Average (Equal-weight)",
        "EF@10%": fmt(*s_eq["ef10"]),
        "EF@20%": fmt(*s_eq["ef20"]),
        "EF@30%": fmt(*s_eq["ef30"]),
        f"Hits@20% ({k20})": fmt(*s_eq["h20"], digits=1),
        f"Hits@30% ({k30})": fmt(*s_eq["h30"], digits=1),
    })
    
    method_name = {"weighted": "CWRA", "rrf": "RRF", "power": "CWRA-Power"}[args.aggregation]
    table6_rows.append({
        "Method": f"{method_name}-{args.focus}",
        "EF@10%": fmt(*s_cwra["ef10"]),
        "EF@20%": fmt(*s_cwra["ef20"]),
        "EF@30%": fmt(*s_cwra["ef30"]),
        f"Hits@20% ({k20})": fmt(*s_cwra["h20"], digits=1),
        f"Hits@30% ({k30})": fmt(*s_cwra["h30"], digits=1),
    })
    
    table6 = pd.DataFrame(table6_rows)

    # =========================================================================
    # FULL RANKING
    # =========================================================================
    if args.aggregation == "rrf":
        consensus_score = reciprocal_rank_fusion(ranks)
    elif args.aggregation == "power":
        ranks_power = power_rank_transform(ranks, N, power=delta_or_power)
        consensus_score = ranks_power.dot(w_mix_full)
    else:
        consensus_score = ranks01.dot(w_mix_full)
    
    df_pool["cwra_score"] = consensus_score
    df_pool["cwra_rank"] = np.argsort(np.argsort(consensus_score)) + 1

    # Add modality ranks
    for j, (col, _) in enumerate(modalities):
        df_pool[f"rank_{col}"] = ranks[:, j]

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    prefix = args.output_prefix
    
    table5.to_csv(f"{prefix}_table5_weights.csv", index=False)
    table6.to_csv(f"{prefix}_table6_performance.csv", index=False)
    df_pool.to_csv(f"{prefix}_full_ranking.csv", index=False)
    
    # Top/bottom generated
    df_gen = df_pool[g_mask].copy()
    if len(df_gen) > 0:
        df_gen_sorted = df_gen.sort_values("cwra_rank")
        df_gen_sorted.head(args.top_n).to_csv(f"{prefix}_top{args.top_n}_G.csv", index=False)
        df_gen_sorted.tail(args.top_n).to_csv(f"{prefix}_bottom{args.top_n}_G.csv", index=False)

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    print("\nTable 5 - Modality Weights and Individual Performance:")
    print("-" * 100)
    print(f"{'Modality':<25} {'Weight':>8} {'EF@10%':>8} {'EF@20%':>8} {'EF@30%':>8}")
    print("-" * 100)
    for _, row in table5.iterrows():
        print(f"{row['Modality']:<25} {row['Weight']:>8.4f} {row['EF@10%']:>8.2f} "
              f"{row['EF@20%']:>8.2f} {row['EF@30%']:>8.2f}")
    print("-" * 100)
    
    print("\n" + "=" * 100)
    print("Table 6 - Comprehensive Performance Comparison (CV Results):")
    print("=" * 100)
    print(f"{'Method':<30} {'EF@10%':>14} {'EF@20%':>14} {'EF@30%':>14}")
    print("-" * 100)
    
    print(f"{'Random':<30} {fmt(*s_random['ef10']):>14} {fmt(*s_random['ef20']):>14} "
          f"{fmt(*s_random['ef30']):>14}")
    print("-" * 100)
    
    ind_sorted = sorted(s_individual.items(), key=lambda x: -x[1]["ef10"][0])
    for mod_name, s in ind_sorted:
        print(f"{mod_name:<30} {fmt(*s['ef10']):>14} {fmt(*s['ef20']):>14} "
              f"{fmt(*s['ef30']):>14}")
    print("-" * 100)
    
    print(f"{'Average (Equal-weight)':<30} {fmt(*s_eq['ef10']):>14} {fmt(*s_eq['ef20']):>14} "
          f"{fmt(*s_eq['ef30']):>14}")
    print(f"{method_name + '-' + args.focus:<30} {fmt(*s_cwra['ef10']):>14} "
          f"{fmt(*s_cwra['ef20']):>14} {fmt(*s_cwra['ef30']):>14}")
    print("=" * 100)
    
    # Final EF on full dataset
    print("\nFinal model EF (full dataset, no CV):")
    full_order = np.argsort(consensus_score)
    for name, k in [("10%", k10), ("20%", k20), ("30%", k30)]:
        h = active_mask[full_order[:k]].sum()
        ef = (h * N) / (k * A_all)
        print(f"  EF@{name}: {ef:.2f} ({h} hits in top {k})")
    
    print(f"\nOutputs saved with prefix '{prefix}'")
    if args.aggregation != "rrf":
        print(f"Grid size used: {len(grid)} configurations")


if __name__ == "__main__":
    main()