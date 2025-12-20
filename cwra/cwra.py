#!/usr/bin/env python3
"""
CWRA - Calibrated Weighted Rank Aggregation for VDR Virtual Screening

Usage:
  python -m cwra --csv labeled_raw_modalities.csv --focus early

Sample:
  python -m cwra --csv data/labeled_raw_modalities.csv --outer_repeats 5 --outer_splits 10 --output_prefix results
"""

from __future__ import annotations
import argparse
import math
import itertools
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import kendalltau
from sklearn.model_selection import GroupKFold, KFold
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
    n_corr = (np.abs(tau) > tau0).sum(axis=1) - 1
    return 1.0 / (1.0 + lam * n_corr)


def compute_weights_v2(
    ef_terms: dict,
    rank_score: np.ndarray,
    bedroc_alpha: np.ndarray,
    shrink: np.ndarray,
    delta: float,
    gamma: float,
    w_ef1: float, w_ef5: float, w_ef10: float,
    w_bedroc: float, w_rank: float,
    eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Enhanced weight computation with EF@1% focus."""
    E = rank_score.shape[0]
    uniform = np.ones(E) / E
    
    # Weighted combination of EF with early enrichment focus
    ef_combined = (w_ef1 * ef_terms.get("1", np.zeros(E)) +
                   w_ef5 * ef_terms.get("5", np.zeros(E)) +
                   w_ef10 * ef_terms.get("10", np.zeros(E)))
    ef_sum = w_ef1 + w_ef5 + w_ef10
    if ef_sum > 0:
        ef_combined /= ef_sum
    
    # Combine components
    ef_weight = 1.0 - w_bedroc - w_rank
    raw = w_bedroc * bedroc_alpha + w_rank * rank_score + ef_weight * ef_combined
    raw = np.maximum(raw, 0.0) + eps
    
    # Apply shrinkage and power transformation (higher delta = sharper weights)
    w = (raw * shrink) ** delta
    w = w / w.sum()
    
    # Mix with uniform (lower gamma = less regularization = sharper weights)
    w_mix = (1.0 - gamma) * w + gamma * uniform
    w_mix = w_mix / w_mix.sum()
    
    return w_mix, w


def reciprocal_rank_fusion(ranks: np.ndarray, k: float = 60.0) -> np.ndarray:
    """Reciprocal Rank Fusion (RRF)."""
    N, E = ranks.shape
    scores = np.zeros(N, dtype=float)
    for j in range(E):
        scores += 1.0 / (k + ranks[:, j])
    return -scores


def power_rank_transform(ranks: np.ndarray, N: int, power: float = 0.5) -> np.ndarray:
    """Power-law rank transformation for early enrichment focus."""
    normalized = (ranks - 1) / (N - 1)
    return normalized ** power


def eval_depths_extended(
    score: np.ndarray,
    eval_mask: np.ndarray,
    A_eval: int,
    cutoffs: dict,
    N: int
) -> dict:
    """Evaluate hits and enrichment at multiple depth cutoffs."""
    if A_eval == 0:
        # Python 3.8 compatible dict merge
        out = {f"h{k}": 0 for k in cutoffs}
        out.update({f"ef{k}": 0.0 for k in cutoffs})
        return out
    
    kmax = max(cutoffs.values())
    top = np.argpartition(score, min(kmax - 1, N - 1))[:kmax]
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
    """Objective emphasizing EF@1% for very early enrichment."""
    return 0.6 * d.get("ef1", 0) + 0.3 * d.get("ef5", 0) + 0.1 * d.get("ef10", 0)


def objective_balanced(d: dict) -> float:
    """Balanced objective across early cutoffs."""
    return 0.4 * d.get("ef1", 0) + 0.4 * d.get("ef5", 0) + 0.2 * d.get("ef10", 0)


def objective_standard(d: dict) -> float:
    """Standard objective with EF@5% and EF@10%."""
    return 0.3 * d.get("ef1", 0) + 0.4 * d.get("ef5", 0) + 0.3 * d.get("ef10", 0)


# =============================================================================
# CV SPLITTING
# =============================================================================

def balanced_group_kfold_indices(
    groups: np.ndarray, 
    n_splits: int, 
    rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create balanced group k-fold splits."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    order = np.arange(len(uniq))
    rng.shuffle(order)

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
    alpha: float
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate metrics for active compounds."""
    A = len(active_idx)
    E = ranks.shape[1]
    
    active_bool = np.zeros(N, bool)
    active_bool[active_idx] = True

    ef_terms = {}
    for name, idx_arr in topk_idx.items():
        hits = active_bool[idx_arr].sum(axis=0)
        k = cutoffs[name]
        ef_terms[name] = (hits * N) / (k * A) if A > 0 else np.zeros(E)

    r_act = ranks[active_idx, :]
    mean_rank = r_act.mean(axis=0)
    rank_score = 1.0 - (mean_rank - 1.0) / (N - 1)

    x_act = ranks01[active_idx, :]
    bedroc = np.empty(E, float)
    for j in range(E):
        bedroc[j] = bedroc_from_x(x_act[:, j], alpha, A, N)

    return ef_terms, mean_rank, rank_score, bedroc


def fmt(m: float, sd: float, digits: int = 2) -> str:
    """Format mean ± std."""
    return f"{m:.{digits}f} +/- {sd:.{digits}f}"


# =============================================================================
# PARALLEL INNER CV EVALUATION
# =============================================================================

def evaluate_params_inner(
    params: tuple,
    inner_pairs: list,
    ranks: np.ndarray,
    ranks01: np.ndarray,
    topk_idx: dict,
    cutoffs: dict,
    N: int,
    shrink_cache: dict,
    objective_fn,
    aggregation: str
) -> tuple[tuple, float, float]:
    """Evaluate a single hyperparameter configuration across inner folds."""
    (w_ef1, w_ef5, w_ef10, w_bed, w_rank, 
     alpha, tau0, lam, delta_or_power, gamma) = params
    
    shrink = shrink_cache[(tau0, lam)]
    objs = []
    
    for inner_tr, inner_va in inner_pairs:
        # Skip if validation set is empty or too small
        if len(inner_va) < 1:
            continue
            
        ef_terms_tr, _, rank_score_tr, bedroc_tr = calc_metrics(
            inner_tr, ranks, ranks01, topk_idx, cutoffs, N, alpha
        )
        
        w_mix, _ = compute_weights_v2(
            ef_terms_tr, rank_score_tr, bedroc_tr,
            shrink, delta_or_power if aggregation == "weighted" else 1.0,
            gamma, w_ef1, w_ef5, w_ef10, w_bed, w_rank
        )
        
        if aggregation == "power":
            ranks_power = power_rank_transform(ranks, N, power=delta_or_power)
            sc = ranks_power.dot(w_mix)
        else:
            sc = ranks01.dot(w_mix)
        
        sc[inner_tr] = np.inf
        
        eval_mask = np.zeros(N, bool)
        eval_mask[inner_va] = True
        A_va = len(inner_va)
        
        d = eval_depths_extended(sc, eval_mask, A_va, cutoffs, N)
        objs.append(objective_fn(d))
    
    if len(objs) == 0:
        return params, 0.0, 0.0
    
    m = float(np.mean(objs))
    s = float(np.std(objs, ddof=1)) if len(objs) > 1 else 0.0
    
    return params, m, s


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="CWRA - Calibrated Weighted Rank Aggregation for VDR virtual screening",
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
                    help="Risk aversion parameter (mean - beta*std)")
    ap.add_argument("--focus", type=str, default="early",
                    choices=["early", "balanced", "standard"],
                    help="Optimization focus")
    ap.add_argument("--aggregation", type=str, default="weighted",
                    choices=["weighted", "rrf", "power"],
                    help="Aggregation method")
    ap.add_argument("--output_prefix", type=str, default="cwra",
                    help="Prefix for output files")
    ap.add_argument("--top_n", type=int, default=25,
                    help="Number of top/bottom structures to extract")
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="Number of parallel jobs (-1 for all cores)")
    args = ap.parse_args()

    print("=" * 70)
    print("CWRA - Calibrated Weighted Rank Aggregation for VDR Virtual Screening")
    print("=" * 70)
    print(f"Focus: {args.focus}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Parallel jobs: {args.n_jobs}")
    
    # Load data
    df = pd.read_csv(args.csv)
    print(f"\nLoaded {len(df)} compounds from {args.csv}")

    if "smiles" not in df.columns:
        raise RuntimeError("CSV must contain a 'smiles' column")
    if "source" not in df.columns:
        raise RuntimeError("CSV must contain a 'source' column")

    # Compute scaffolds
    df["murcko"] = df["smiles"].map(murcko_smiles)
    n_scaffolds = df["murcko"].nunique()
    print(f"Computed {n_scaffolds} unique Murcko scaffolds")

    # Include calcitriol in the pool (exclude only newRef_137)
    df_pool = df.loc[~df["source"].isin(["newRef_137"])].reset_index(drop=True)
    print(f"Candidate pool: {len(df_pool)} compounds")

    # =========================================================================
    # MODALITY DEFINITIONS
    # Note on high_better flag:
    #   high_better=True  -> larger raw value = better compound
    #   high_better=False -> smaller raw value = better compound
    # =========================================================================
    modalities = [
        ("graphdta_kd", False),      # Kd: lower = stronger binding
        ("graphdta_ki", False),      # Ki: lower = stronger binding
        ("graphdta_ic50", False),    # IC50: lower = more potent
        ("mltle_pKd", True),         # pKd: higher = stronger binding
        ("vina_score", False),       # Docking score: more negative = better
        ("boltz_affinity", False),   # Affinity: lower = stronger binding
        ("boltz_confidence", True),  # Confidence: higher = better
        ("unimol_similarity", True), # Similarity: higher = more similar to actives
        ("tankbind_affinity", False),# Affinity: lower = stronger binding
        ("drugban_affinity", False), # Affinity: lower = stronger binding
        ("moltrans_affinity", False),# Affinity: lower = stronger binding
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
    
    missing_cols = [col for col, _ in modalities if col not in df_pool.columns]
    if missing_cols:
        raise RuntimeError(f"Missing modality columns: {missing_cols}")
    
    E = len(modalities)
    N = len(df_pool)

    # Define actives: initial_370 + calcitriol
    active_mask = (df_pool["source"].isin(["initial_370", "calcitriol"])).to_numpy()
    act_idx_all = np.where(active_mask)[0]
    A_all = int(active_mask.sum())
    print(f"Actives (initial_370 + calcitriol): {A_all}")

    if A_all == 0:
        raise RuntimeError("No actives found in dataset")

    g_mask = df_pool["source"].isin(["G1", "G2", "G3"]).to_numpy()
    print(f"Generated compounds (G1/G2/G3): {g_mask.sum()}")

    # =========================================================================
    # NORMALIZE MODALITIES
    # Transform to [0, 1] where 0 = best, 1 = worst
    # =========================================================================
    scores = np.empty((N, E), float)
    for j, (col, high_better) in enumerate(modalities):
        x = df_pool[col].to_numpy(float)
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        if np.isnan(xmin) or np.isnan(xmax):
            raise RuntimeError(f"Column '{col}' contains all NaN values")
        if xmax != xmin:
            s = (x - xmin) / (xmax - xmin)
        else:
            s = np.zeros_like(x)
        if high_better:
            s = 1.0 - s
        scores[:, j] = s

    # =========================================================================
    # COMPUTE RANKS
    # =========================================================================
    ranks = np.empty((N, E), int)
    orders = []
    for j in range(E):
        order = np.argsort(scores[:, j], kind="mergesort")
        orders.append(order)
        r = np.empty(N, int)
        r[order] = np.arange(1, N + 1)
        ranks[:, j] = r
    
    ranks01 = (ranks - 1) / (N - 1)

    # =========================================================================
    # KENDALL TAU CORRELATION
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
    # "1", "5", "10" used for weight computation and Table 5
    # "10", "20", "30" used for Table 6 performance reporting
    # =========================================================================
    cutoffs = {
        "1": max(1, math.ceil(0.01 * N)),
        "5": math.ceil(0.05 * N),
        "10": math.ceil(0.10 * N),
        "20": math.ceil(0.20 * N),
        "30": math.ceil(0.30 * N),
    }
    
    k10, k20, k30 = cutoffs["10"], cutoffs["20"], cutoffs["30"]
    print(f"\nCutoffs: {cutoffs}")

    # Precompute top-k indices
    topk_idx = {}
    for name, k in cutoffs.items():
        topk_idx[name] = np.stack([orders[j][:k] for j in range(E)], axis=1)

    objective_fn = {
        "early": objective_early_focus,
        "balanced": objective_balanced,
        "standard": objective_standard,
    }[args.focus]
    print(f"\nUsing {args.focus.upper()} objective function")

    # =========================================================================
    # HYPERPARAMETER GRID
    # =========================================================================
    alphas = [40.0, 80.0, 160.0]
    
    # EF weight combinations (w_ef1, w_ef5, w_ef10)
    ef_weight_combos = [
        (0.8, 0.15, 0.05),
        (0.6, 0.3, 0.1),
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    
    bedroc_rank_combos = [
        (0.5, 0.05),
        (0.4, 0.1),
        (0.6, 0.05),
        (0.3, 0.1),
        (0.0, 0.0),
    ]
    
    delta_list = [1.0, 1.5, 2.0, 2.5]
    gamma_list = [0.01, 0.02, 0.05]
    tau0_list = [0.3, 0.4, 0.5]
    lam_list = [0.0, 0.25, 0.5]
    power_list = [0.3, 0.5, 0.7]

    # Build grid
    if args.aggregation == "rrf":
        grid = [(None,) * 10]
        print(f"\nRRF aggregation: No hyperparameter search needed")
    elif args.aggregation == "power":
        grid = [
            (w_ef1, w_ef5, w_ef10, w_bed, w_rank, alpha, tau0, lam, power, gamma)
            for (w_ef1, w_ef5, w_ef10), (w_bed, w_rank), alpha, tau0, lam, power, gamma
            in itertools.product(ef_weight_combos, bedroc_rank_combos, alphas, 
                               tau0_list, lam_list, power_list, gamma_list)
        ]
        print(f"\nPower aggregation grid size: {len(grid)} configurations")
    else:
        grid = [
            (w_ef1, w_ef5, w_ef10, w_bed, w_rank, alpha, tau0, lam, delta, gamma)
            for (w_ef1, w_ef5, w_ef10), (w_bed, w_rank), alpha, tau0, lam, delta, gamma
            in itertools.product(ef_weight_combos, bedroc_rank_combos, alphas, 
                               tau0_list, lam_list, delta_list, gamma_list)
        ]
        print(f"\nWeighted aggregation grid size: {len(grid)} configurations")

    # Pre-compute shrinkage factors
    shrink_cache = {
        (tau0, lam): shrink_factors(tau, tau0, lam) 
        for tau0 in tau0_list for lam in lam_list
    }

    act_groups = df_pool.loc[act_idx_all, "murcko"].fillna("UNKNOWN").to_numpy()
    uniform_w = np.ones(E) / E

    # =========================================================================
    # VALIDATE CV PARAMETERS
    # =========================================================================
    n_act = len(act_groups)
    n_scaf = len(np.unique(act_groups))
    
    if args.outer_splits > n_scaf:
        print(f"\nWARNING: --outer_splits={args.outer_splits} > unique active scaffolds={n_scaf}")
        print(f"  Reducing outer_splits to {n_scaf}")
        args.outer_splits = n_scaf
    
    if args.outer_splits > n_act:
        print(f"\nWARNING: --outer_splits={args.outer_splits} > #actives={n_act}")
        print(f"  Reducing outer_splits to {n_act}")
        args.outer_splits = n_act

    fold_results = []
    chosen_params = []

    print("\n" + "=" * 70)
    print("Running Nested Cross-Validation (Parallelized)")
    print("=" * 70)

    # =========================================================================
    # NESTED CV WITH PARALLELIZATION
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

            # Skip fold if test set is empty
            if len(test_act) == 0:
                print(f"[Rep {rep+1} Fold {fold:2d}] Skipped: empty test set")
                continue

            unique_groups = len(np.unique(train_groups))
            n_inner = min(5, max(2, unique_groups))

            # Skip if training set is too small for inner CV
            if len(train_act) < 2:
                print(f"[Rep {rep+1} Fold {fold:2d}] Skipped: insufficient training actives")
                continue

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

            # Filter out pairs with empty validation sets
            inner_pairs = [(tr, va) for tr, va in inner_pairs if len(va) >= 1]
            
            if len(inner_pairs) == 0:
                print(f"[Rep {rep+1} Fold {fold:2d}] Skipped: no valid inner CV splits")
                continue

            # Parallelized inner CV
            if args.aggregation == "rrf":
                best_param = (None,) * 10
            else:
                beta = float(args.risk_beta)
                
                results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                    delayed(evaluate_params_inner)(
                        params, inner_pairs, ranks, ranks01, topk_idx,
                        cutoffs, N, shrink_cache, objective_fn, args.aggregation
                    )
                    for params in grid
                )
                
                best_score = -1e18
                best_param = None
                for params, m, s in results:
                    score = m - beta * s
                    if score > best_score:
                        best_score = score
                        best_param = params

            chosen_params.append(best_param)

            # Outer fold evaluation
            test_mask = np.zeros(N, bool)
            test_mask[test_act] = True
            A_test = len(test_act)

            if args.aggregation == "rrf":
                sc_cwra = reciprocal_rank_fusion(ranks)
                w_mix = uniform_w
            else:
                (w_ef1, w_ef5, w_ef10, w_bed, w_rank,
                 alpha, tau0, lam, delta_or_power, gamma) = best_param
                
                shrink = shrink_cache[(tau0, lam)]
                ef_terms_tr, _, rank_score_tr, bedroc_tr = calc_metrics(
                    train_act, ranks, ranks01, topk_idx, cutoffs, N, alpha
                )
                w_mix, _ = compute_weights_v2(
                    ef_terms_tr, rank_score_tr, bedroc_tr,
                    shrink, delta_or_power if args.aggregation == "weighted" else 1.0,
                    gamma, w_ef1, w_ef5, w_ef10, w_bed, w_rank
                )
                
                if args.aggregation == "power":
                    ranks_power = power_rank_transform(ranks, N, power=delta_or_power)
                    sc_cwra = ranks_power.dot(w_mix)
                else:
                    sc_cwra = ranks01.dot(w_mix)

            sc_cwra[train_act] = np.inf
            d_cwra = eval_depths_extended(sc_cwra, test_mask, A_test, cutoffs, N)

            # Baselines
            sc_eq = ranks01.dot(uniform_w)
            sc_eq[train_act] = np.inf
            d_eq = eval_depths_extended(sc_eq, test_mask, A_test, cutoffs, N)

            rng_fold = np.random.default_rng(rep_seed + fold)
            sc_random = rng_fold.random(N)
            sc_random[train_act] = np.inf
            d_random = eval_depths_extended(sc_random, test_mask, A_test, cutoffs, N)

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

            print(f"[Rep {rep+1} Fold {fold:2d}] CWRA EF@1%={d_cwra['ef1']:.2f} "
                  f"EF@5%={d_cwra['ef5']:.2f} EF@10%={d_cwra['ef10']:.2f}  "
                  f"Equal EF@1%={d_eq['ef1']:.2f}")

    # Check if we have any valid folds
    if len(fold_results) == 0:
        raise RuntimeError("No valid CV folds were completed. Check your data and parameters.")

    # =========================================================================
    # FINAL MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("Computing Final Results")
    print("=" * 70)

    if args.aggregation == "rrf":
        final_param = (None,) * 10
        alpha_final = 80.0
        w_mix_full = uniform_w
        delta_or_power = 1.0
    else:
        arr = np.array(chosen_params, dtype=object)
        final_param = tuple(Counter(arr[:, j]).most_common(1)[0][0] for j in range(arr.shape[1]))
        (w_ef1, w_ef5, w_ef10, w_bed, w_rank,
         alpha_final, tau0, lam, delta_or_power, gamma) = final_param
        shrink = shrink_cache[(tau0, lam)]

        print(f"\nFinal hyperparameters (majority vote):")
        print(f"  EF weights: w_ef1={w_ef1}, w_ef5={w_ef5}, w_ef10={w_ef10}")
        print(f"  BEDROC/rank: w_bed={w_bed}, w_rank={w_rank}")
        print(f"  alpha={alpha_final}, tau0={tau0}, lambda={lam}")
        print(f"  delta={delta_or_power}, gamma={gamma}")

        ef_terms_full, mean_rank_full, rank_score_full, bedroc_full = calc_metrics(
            act_idx_all, ranks, ranks01, topk_idx, cutoffs, N, alpha_final
        )
        w_mix_full, _ = compute_weights_v2(
            ef_terms_full, rank_score_full, bedroc_full,
            shrink, delta_or_power if args.aggregation == "weighted" else 1.0,
            gamma, w_ef1, w_ef5, w_ef10, w_bed, w_rank
        )

    print(f"\nFinal modality weights:")
    weight_entropy = -np.sum(w_mix_full * np.log(w_mix_full + 1e-10))
    uniform_entropy = np.log(E)
    print(f"  Weight entropy: {weight_entropy:.3f} (uniform={uniform_entropy:.3f})")
    for name, w in sorted(zip(mod_names_simple, w_mix_full), key=lambda x: -x[1]):
        print(f"  {name}: {w:.4f}")

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

    for j, (col, _) in enumerate(modalities):
        df_pool[f"rank_{col}"] = ranks[:, j]

    # =========================================================================
    # OUTPUT TABLES
    # =========================================================================
    # For RRF, we need to compute metrics here; for others, reuse from weight computation
    if args.aggregation == "rrf":
        ef_terms_full, mean_rank_full, _, _ = calc_metrics(
            act_idx_all, ranks, ranks01, topk_idx, cutoffs, N, alpha_final
        )
    
    table5 = pd.DataFrame({
        "Modality": mod_labels,
        "Weight": w_mix_full,
        "EF@1%": ef_terms_full["1"],
        "EF@5%": ef_terms_full["5"],
        "EF@10%": ef_terms_full["10"],
        "Mean_Rank": mean_rank_full,
    })

    def agg(method_key: str) -> dict:
        vals = {}
        for name in cutoffs.keys():
            h_arr = np.array([fr[method_key][f"h{name}"] for fr in fold_results], float)
            ef_arr = np.array([fr[method_key][f"ef{name}"] for fr in fold_results], float)
            h_std = float(h_arr.std(ddof=1)) if len(h_arr) > 1 else 0.0
            ef_std = float(ef_arr.std(ddof=1)) if len(ef_arr) > 1 else 0.0
            vals[f"h{name}"] = (float(h_arr.mean()), h_std)
            vals[f"ef{name}"] = (float(ef_arr.mean()), ef_std)
        return vals

    s_eq = agg("equal")
    s_cwra = agg("cwra")
    s_random = agg("random")
    
    s_individual = {}
    for mod_name in mod_names_simple:
        vals = {}
        for name in cutoffs.keys():
            h_arr = np.array([fr["individual"][mod_name][f"h{name}"] for fr in fold_results], float)
            ef_arr = np.array([fr["individual"][mod_name][f"ef{name}"] for fr in fold_results], float)
            h_std = float(h_arr.std(ddof=1)) if len(h_arr) > 1 else 0.0
            ef_std = float(ef_arr.std(ddof=1)) if len(ef_arr) > 1 else 0.0
            vals[f"h{name}"] = (float(h_arr.mean()), h_std)
            vals[f"ef{name}"] = (float(ef_arr.mean()), ef_std)
        s_individual[mod_name] = vals

    table6_rows = [{"Method": "Random", "EF@10%": fmt(*s_random["ef10"]),
                    "Hits@10%": fmt(*s_random["h10"]), "Hits@20%": fmt(*s_random["h20"]), "Hits@30%": fmt(*s_random["h30"])}]
    
    for mod_name in mod_names_simple:
        s = s_individual[mod_name]
        table6_rows.append({"Method": f"{mod_name}",
                           "EF@10%": fmt(*s["ef10"]), "Hits@10%": fmt(*s["h10"]),
                           "Hits@20%": fmt(*s["h20"]), "Hits@30%": fmt(*s["h30"])})
    
    table6_rows.append({"Method": "Average (Equal-weight)",
                        "EF@10%": fmt(*s_eq["ef10"]), "Hits@10%": fmt(*s_eq["h10"]),
                        "Hits@20%": fmt(*s_eq["h20"]), "Hits@30%": fmt(*s_eq["h30"])})
    
    method_name = {"weighted": "CWRA", "rrf": "RRF", "power": "CWRA-Power"}[args.aggregation]
    table6_rows.append({"Method": f"{method_name}-{args.focus}",
                        "EF@10%": fmt(*s_cwra["ef10"]), "Hits@10%": fmt(*s_cwra["h10"]),
                        "Hits@20%": fmt(*s_cwra["h20"]), "Hits@30%": fmt(*s_cwra["h30"])})
    
    table6 = pd.DataFrame(table6_rows)

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    prefix = args.output_prefix
    
    table5.to_csv(f"{prefix}_table5_weights.csv", index=False)
    table6.to_csv(f"{prefix}_table6_performance.csv", index=False)
    df_pool.to_csv(f"{prefix}_full_ranking.csv", index=False)
    
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
    print(f"{'Modality':<25} {'Weight':>8} {'EF@1%':>8} {'EF@5%':>8} {'EF@10%':>8}")
    print("-" * 100)
    for _, row in table5.sort_values("Weight", ascending=False).iterrows():
        print(f"{row['Modality']:<25} {row['Weight']:>8.4f} {row['EF@1%']:>8.2f} "
              f"{row['EF@5%']:>8.2f} {row['EF@10%']:>8.2f}")
    print("-" * 100)
    
    print("\n" + "=" * 100)
    print("Table 6 - Performance Comparison (CV Results):")
    print("=" * 100)
    print(f"{'Method':<30} {'EF@10%':>16} {'Hits@10%':>16} {'Hits@20%':>16}")
    print("-" * 100)
    
    print(f"{'Random':<30} {fmt(*s_random['ef10']):>16} {fmt(*s_random['h10']):>16} "
          f"{fmt(*s_random['h20']):>16}")
    print("-" * 100)
    
    ind_sorted = sorted(s_individual.items(), key=lambda x: -x[1]["ef10"][0])
    for mod_name, s in ind_sorted:
        print(f"{mod_name:<30} {fmt(*s['ef10']):>16} {fmt(*s['h10']):>16} "
              f"{fmt(*s['h20']):>16}")
    print("-" * 100)
    
    print(f"{'Average (Equal-weight)':<30} {fmt(*s_eq['ef10']):>16} {fmt(*s_eq['h10']):>16} "
          f"{fmt(*s_eq['h20']):>16}")
    print(f"{method_name + '-' + args.focus:<30} {fmt(*s_cwra['ef10']):>16} "
          f"{fmt(*s_cwra['h10']):>16} {fmt(*s_cwra['h20']):>16}")
    print("=" * 100)
    
    # Improvement summary
    cwra_ef10_mean = s_cwra["ef10"][0]
    eq_ef10_mean = s_eq["ef10"][0]
    best_ind_ef10 = max(s["ef10"][0] for s in s_individual.values())
    
    if eq_ef10_mean > 0:
        print(f"\nImprovement over Equal-weight at EF@10%: {100*(cwra_ef10_mean/eq_ef10_mean - 1):.1f}%")
    if best_ind_ef10 > 0:
        print(f"Improvement over best individual at EF@10%: {100*(cwra_ef10_mean/best_ind_ef10 - 1):.1f}%")
    
    print("\nFinal model EF (full dataset, no CV):")
    full_order = np.argsort(consensus_score)
    for name, k in [("10%", k10), ("20%", k20), ("30%", k30)]:
        h = active_mask[full_order[:k]].sum()
        ef = (h * N) / (k * A_all)
        print(f"  EF@{name}: {ef:.2f} ({h} hits in top {k})")
    
    # Print calcitriol rank
    calcitriol_row = df_pool[df_pool["source"] == "calcitriol"]
    if len(calcitriol_row) > 0:
        calcitriol_rank = calcitriol_row["cwra_rank"].iloc[0]
        percentile = 100 * calcitriol_rank / N
        print(f"\nCalcitriol rank: {int(calcitriol_rank)} / {N} (top {percentile:.1f}%)")
    
    print(f"\nOutputs saved with prefix '{prefix}'")


if __name__ == "__main__":
    main()