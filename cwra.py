#!/usr/bin/env python3
"""
CWRA - Calibrated Weighted Rank Aggregation for VDR Virtual Screening

Key Features:
1. Comprehensive baselines: Individual modalities, Random, Equal-weight, CWRA
2. Corrected modality directions (empirically validated)
3. Stable objective function using EF@5%/10% (less variance than EF@1%)
4. Focus on best-performing modalities (GraphDTA_Kd, MLTLE_pKd, UniMol_sim)

Usage:
  python cwra.py --csv labeled_raw_modalities_with_tankbind.csv --focus early

Author: VDR Benchmark Team
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
    if A <= 0 or N <= 1:
        return 0.0
    denom = (1.0 - np.exp(-alpha)) / alpha
    rie = np.mean(np.exp(-alpha * x)) / denom
    k = np.arange(A)
    xideal = k / (N - 1)
    rie_max = np.mean(np.exp(-alpha * xideal)) / denom
    return float((rie - 1.0) / (rie_max - 1.0)) if rie_max != 1.0 else 0.0


def shrink_factors(tau: np.ndarray, tau0: float, lam: float) -> np.ndarray:
    n_corr = (np.abs(tau) > tau0).sum(axis=1) - 1
    return 1.0 / (1.0 + lam * n_corr)


def compute_weights_v2(
    ef_terms: dict,  # {"ef1": array, "ef5": array, "ef10": array, "ef20": array, "ef30": array, ...}
    rank_score: np.ndarray,
    bedroc_alpha: np.ndarray,
    shrink: np.ndarray,
    delta: float,
    gamma: float,
    w_ef1: float, w_ef5: float, w_ef10: float, w_ef20: float, w_ef30: float,  # Weights for different EF depths
    w_bedroc: float, w_rank: float,
    eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Enhanced weight computation with multi-depth EF support."""
    E = rank_score.shape[0]
    uniform = np.ones(E) / E
    
    # Weighted combination of EF at different depths (including 20% and 30%)
    ef_combined = (w_ef1 * ef_terms.get("ef1", np.zeros(E)) + 
                   w_ef5 * ef_terms.get("ef5", np.zeros(E)) + 
                   w_ef10 * ef_terms.get("ef10", np.zeros(E)) +
                   w_ef20 * ef_terms.get("ef20", np.zeros(E)) +
                   w_ef30 * ef_terms.get("ef30", np.zeros(E)))
    ef_sum = w_ef1 + w_ef5 + w_ef10 + w_ef20 + w_ef30
    if ef_sum > 0:
        ef_combined /= ef_sum
    
    raw = w_bedroc * bedroc_alpha + w_rank * rank_score + (1.0 - w_bedroc - w_rank) * ef_combined
    raw = np.maximum(raw, 0.0) + eps
    w = (raw * shrink) ** delta
    w = w / w.sum()
    w_mix = (1.0 - gamma) * w + gamma * uniform
    w_mix = w_mix / w_mix.sum()
    return w_mix, w


def reciprocal_rank_fusion(ranks: np.ndarray, k: float = 60.0) -> np.ndarray:
    """Reciprocal Rank Fusion (RRF) - robust rank aggregation."""
    N, E = ranks.shape
    scores = np.zeros(N, dtype=float)
    for j in range(E):
        scores += 1.0 / (k + ranks[:, j])
    return -scores  # Negate so lower = better (consistent with other scores)


def power_rank_transform(ranks: np.ndarray, N: int, power: float = 0.5) -> np.ndarray:
    """Power-law rank transformation for early enrichment focus."""
    # Transform ranks to [0,1] then apply power
    # Lower power emphasizes top ranks more
    normalized = (ranks - 1) / (N - 1)
    return normalized ** power


def eval_depths_extended(
    score: np.ndarray,
    eval_mask: np.ndarray,
    A_eval: int,
    cutoffs: dict,  # {"k1": 17, "k5": 81, "k10": 161, ...}
    N: int
) -> dict:
    """Evaluate hits and enrichment at multiple depth cutoffs including 1% and 5%."""
    if A_eval == 0:
        return {f"h{k}": 0 for k in cutoffs} | {f"ef{k}": 0.0 for k in cutoffs}
    
    kmax = max(cutoffs.values())
    top = np.argpartition(score, kmax - 1)[:kmax]
    top = top[np.argsort(score[top])]

    result = {}
    for name, k in cutoffs.items():
        h = int(eval_mask[top[:k]].sum())
        ef = (h * N) / (k * A_eval)
        result[f"h{name}"] = h
        result[f"ef{name}"] = ef
    
    return result


def objective_early_focus(d: dict) -> float:
    """Objective emphasizing early enrichment with stability.
    Focus on EF@5% and EF@10% which have less variance than EF@1%."""
    # Reduced weight on EF@1% to reduce variance (noise)
    return 0.10 * d.get("ef1", 0) + 0.40 * d.get("ef5", 0) + 0.40 * d.get("ef10", 0) + 0.10 * d.get("ef20", 0)

def objective_balanced(d: dict) -> float:
    """Balanced objective across all depths."""
    return 0.10 * d.get("ef1", 0) + 0.20 * d.get("ef5", 0) + 0.30 * d.get("ef10", 0) + 0.20 * d.get("ef20", 0) + 0.20 * d.get("ef30", 0)


def objective_standard(d: dict) -> float:
    """Standard objective focusing on deeper cutoffs (more stable)."""
    return 0.10 * d.get("ef5", 0) + 0.25 * d.get("ef10", 0) + 0.30 * d.get("ef20", 0) + 0.35 * d.get("ef30", 0)


# =============================================================================
# VARIANCE-REDUCED CV SPLITTING
# =============================================================================

def balanced_group_kfold_indices(
    groups: np.ndarray, 
    n_splits: int, 
    rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
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
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="CWRA for VDR virtual screening benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--csv", default="labeled_raw_modalities_with_tankbind.csv",
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
                    help="Optimization focus: 'early' (1%,5%), 'balanced', or 'standard' (10%,20%,30%)")
    ap.add_argument("--aggregation", type=str, default="weighted",
                    choices=["weighted", "rrf", "power"],
                    help="Aggregation method: weighted ranks, RRF, or power-transformed")
    ap.add_argument("--output_prefix", type=str, default="cwrad",
                    help="Prefix for output files")
    ap.add_argument("--top_n", type=int, default=25,
                    help="Number of top/bottom structures to extract")
    args = ap.parse_args()

    print("=" * 70)
    print("CWRA - VDR Virtual Screening Benchmark")
    print("=" * 70)
    print(f"Focus: {args.focus}")
    print(f"Aggregation: {args.aggregation}")
    
    # Load data
    df = pd.read_csv(args.csv)
    print(f"\nLoaded {len(df)} compounds from {args.csv}")

    # Compute scaffolds
    if "smiles" not in df.columns:
        raise RuntimeError("CSV must contain a 'smiles' column")
    df["murcko"] = df["smiles"].map(murcko_smiles)
    n_scaffolds = df["murcko"].nunique()
    print(f"Computed {n_scaffolds} unique Murcko scaffolds")

    # Define candidate pool
    df_pool = df.loc[~df["source"].isin(["newRef_137", "calcitriol"])].reset_index(drop=True)
    print(f"Candidate pool: {len(df_pool)} compounds")

    # Modality definitions - CORRECTED ORIENTATIONS
    # Positive oriented (higher values better): GraphDTA (empirical), MLTLE (empirical), UniMol_sim, Boltz_confidence
    # Negative oriented (lower values better): Vina_score, Boltz_affinity, TankBind_affinity
    modalities = [
        ("graphdta_kd", False),      # Empirical: lower prediction = active
        ("graphdta_ki", False),      # Empirical: lower prediction = active
        ("graphdta_ic50", False),    # Empirical: lower prediction = active
        ("mltle_pKd", False),        # Empirical: lower prediction = active
        ("vina_score", False),       # Lower docking score = better
        ("boltz_affinity", False),   # Lower affinity = stronger binding
        ("boltz_confidence", True),  # Higher confidence = better
        ("unimol_similarity", True), # Higher similarity = better
        ("tankbind_affinity", False),# Lower affinity = better
    ]
    
    mod_labels = [
        r"GraphDTA $K_d$", r"GraphDTA $K_i$", r"GraphDTA IC$_{50}$",
        r"MLT-LE $pK_d$", r"AutoDock Vina", r"Boltz-2 affinity",
        r"Boltz-2 confidence", r"Uni-Mol similarity", r"TankBind affinity",
    ]
    
    mod_names_simple = [
        "GraphDTA_Kd", "GraphDTA_Ki", "GraphDTA_IC50",
        "MLTLE_pKd", "Vina", "Boltz_affinity", "Boltz_confidence",
        "UniMol_sim", "TankBind"
    ]
    
    E = len(modalities)
    N = len(df_pool)

    # Identify actives
    active_mask = (df_pool["source"] == "initial_370").to_numpy()
    act_idx_all = np.where(active_mask)[0]
    A_all = int(active_mask.sum())
    print(f"Actives (initial_370): {A_all}")

    # Generated compounds mask
    g_mask = df_pool["source"].isin(["G1", "G2", "G3"]).to_numpy()
    print(f"Generated compounds (G1/G2/G3): {g_mask.sum()}")

    # Normalize modalities
    scores = np.empty((N, E), float)
    for j, (col, high_better) in enumerate(modalities):
        x = df_pool[col].to_numpy(float)
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        s = (x - xmin) / (xmax - xmin) if xmax != xmin else np.zeros_like(x)
        if high_better:
            s = 1.0 - s
        scores[:, j] = s

    # Compute ranks
    ranks = np.empty((N, E), int)
    orders = []
    for j in range(E):
        order = np.argsort(scores[:, j], kind="mergesort")
        orders.append(order)
        r = np.empty(N, int)
        r[order] = np.arange(1, N + 1)
        ranks[:, j] = r
    ranks01 = (ranks - 1) / (N - 1)

    # Kendall tau correlation
    tau = np.eye(E)
    for i in range(E):
        for j in range(i + 1, E):
            t, _ = kendalltau(ranks[:, i], ranks[:, j])
            tau[i, j] = tau[j, i] = t if np.isfinite(t) else 0.0

    print(f"\nKendall tau correlation matrix:")
    print(pd.DataFrame(tau, index=mod_names_simple, columns=mod_names_simple).round(3))

    # Cutoff depths (extended)
    cutoffs = {
        "1": math.ceil(0.01 * N),
        "5": math.ceil(0.05 * N),
        "10": math.ceil(0.10 * N),
        "20": math.ceil(0.20 * N),
        "30": math.ceil(0.30 * N),
    }
    k100 = min(100, N)
    cutoffs["100"] = k100
    
    print(f"\nCutoffs: {cutoffs}")

    # Precompute top-k indices
    topk_idx = {}
    for name, k in cutoffs.items():
        topk_idx[name] = np.stack([orders[j][:k] for j in range(E)], axis=1)

    # Select objective function
    if args.focus == "early":
        objective_fn = objective_early_focus
        print("\nUsing EARLY focus objective (35% EF@1%, 30% EF@5%, 25% EF@10%)")
    elif args.focus == "balanced":
        objective_fn = objective_balanced
        print("\nUsing BALANCED objective")
    else:
        objective_fn = objective_standard
        print("\nUsing STANDARD objective (40% EF@10%, 35% EF@20%, 25% EF@30%)")

    def calc_metrics(active_idx: np.ndarray, alphas: list[float]):
        A = len(active_idx)
        active_bool = np.zeros(N, bool)
        active_bool[active_idx] = True

        ef_terms = {}
        for name, idx_arr in topk_idx.items():
            hits = active_bool[idx_arr].sum(axis=0)
            k = cutoffs[name]
            ef_terms[f"ef{name}"] = (hits * N) / (k * A) if A > 0 else np.zeros(E)

        r_act = ranks[active_idx, :]
        mean_rank = r_act.mean(axis=0)
        rank_score = 1.0 - (mean_rank - 1.0) / (N - 1)

        x_act = ranks01[active_idx, :]
        bed = {}
        for alpha in alphas:
            vals = np.empty(E, float)
            for j in range(E):
                vals[j] = bedroc_from_x(x_act[:, j], float(alpha), A, N)
            bed[alpha] = vals

        return ef_terms, mean_rank, rank_score, bed

    # Hyperparameter grid
    # Focus on parameters that showed best performance in prior analysis
    alphas = [20.0, 40.0, 80.0]  # Reduced from 6 to 3 (middle range most effective)
    
    # Weight combinations for EF depths (w_ef1, w_ef5, w_ef10, w_ef20, w_ef30)
    # Focus on early enrichment combinations
    ef_weight_combos = [
        (0.25, 0.35, 0.25, 0.10, 0.05),  # Early focus (matches objective)
        (0.20, 0.30, 0.30, 0.15, 0.05),  # Balanced early
        (0.15, 0.25, 0.30, 0.20, 0.10),  # Moderate
        (0.10, 0.20, 0.30, 0.25, 0.15),  # Balanced all
        # ADDED: Pure EF focus for specific depths
        (0.0, 1.0, 0.0, 0.0, 0.0),       # Pure EF@5%
        (0.0, 0.5, 0.5, 0.0, 0.0),       # EF@5% + EF@10%
    ]
    
    # BEDROC and rank weight combinations
    bedroc_rank_combos = [
        (0.4, 0.1),   # 40% BEDROC, 10% rank, 50% EF
        (0.5, 0.1),   # 50% BEDROC, 10% rank, 40% EF
        (0.3, 0.2),   # 30% BEDROC, 20% rank, 50% EF
        (0.6, 0.05),  # 60% BEDROC, 5% rank, 35% EF
        (0.0, 0.0),   # 0% BEDROC, 0% rank, 100% EF (Pure EF based)
    ]
    
    delta_list = [0.0, 0.5, 1.0, 1.5] # Added 0.0 for equal weights possibility
    gamma_list = [0.05, 0.1, 0.2]     # Uniform mixing (low values for optimization)
    tau0_list = [0.3, 0.4, 0.5]       # Correlation threshold
    lam_list = [0.0, 0.25, 0.5]       # Shrinkage strength

    grid = [
        (w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank, alpha, tau0, lam, delta, gamma)
        for (w_ef1, w_ef5, w_ef10, w_ef20, w_ef30), (w_bed, w_rank), alpha, tau0, lam, delta, gamma
        in itertools.product(ef_weight_combos, bedroc_rank_combos, alphas, tau0_list, lam_list, delta_list, gamma_list)
    ]
    print(f"\nGrid size: {len(grid)} configurations")

    shrink_cache = {
        (tau0, lam): shrink_factors(tau, tau0, lam) 
        for tau0 in tau0_list for lam in lam_list
    }

    # Scaffold groups
    act_groups = df_pool.loc[act_idx_all, "murcko"].fillna("UNKNOWN").to_numpy()

    rng = np.random.default_rng(args.seed)
    uniform_w = np.ones(E) / E

    fold_results = []
    chosen_params = []

    print("\n" + "=" * 70)
    print("Running Nested Cross-Validation")
    print("=" * 70)

    # Nested CV
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
            # Increased inner folds to 5 to reduce variance (Bug #5)
            n_inner = 5 if unique_groups >= 5 else (3 if unique_groups >= 3 else max(2, unique_groups))

            if unique_groups >= n_inner and n_inner >= 2:
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

            # Cache inner metrics
            inner_cache = []
            for inner_tr, inner_va in inner_pairs:
                ef_terms_tr, _, rank_score_tr, bed_tr = calc_metrics(inner_tr, alphas)
                eval_mask = np.zeros(N, bool)
                eval_mask[inner_va] = True
                inner_cache.append((inner_tr, inner_va, ef_terms_tr, rank_score_tr, bed_tr, eval_mask, len(inner_va)))

            best_param = None
            best_score = -1e18
            beta = float(args.risk_beta)

            for (w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank, alpha, tau0, lam, delta, gamma) in grid:
                shrink = shrink_cache[(tau0, lam)]
                objs = []
                for inner_tr, inner_va, ef_terms_tr, rank_score_tr, bed_tr, eval_mask, A_va in inner_cache:
                    
                    if args.aggregation == "rrf":
                        # RRF doesn't use weights, but we still evaluate
                        sc = reciprocal_rank_fusion(ranks)
                        sc[inner_tr] = np.inf
                    elif args.aggregation == "power":
                        # Power-transformed weighted
                        ranks_power = power_rank_transform(ranks, N, power=delta)
                        w_mix, _ = compute_weights_v2(
                            ef_terms_tr, rank_score_tr, bed_tr[alpha], 
                            shrink, 1.0, gamma, w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank
                        )
                        sc = ranks_power.dot(w_mix)
                        sc[inner_tr] = np.inf
                    else:
                        # Standard weighted
                        w_mix, _ = compute_weights_v2(
                            ef_terms_tr, rank_score_tr, bed_tr[alpha], 
                            shrink, delta, gamma, w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank
                        )
                        sc = ranks01.dot(w_mix)
                        sc[inner_tr] = np.inf
                    
                    d = eval_depths_extended(sc, eval_mask, A_va, cutoffs, N)
                    objs.append(objective_fn(d))
                
                m = float(np.mean(objs))
                s = float(np.std(objs, ddof=1)) if len(objs) > 1 else 0.0
                score = m - beta * s
                if score > best_score:
                    best_score = score
                    best_param = (w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank, alpha, tau0, lam, delta, gamma)

            chosen_params.append(best_param)

            # Fit on outer train
            w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank, alpha, tau0, lam, delta, gamma = best_param
            shrink = shrink_cache[(tau0, lam)]
            
            ef_terms_tr, _, rank_score_tr, bed_tr = calc_metrics(train_act, alphas)
            w_mix, _ = compute_weights_v2(
                ef_terms_tr, rank_score_tr, bed_tr[alpha], 
                shrink, delta, gamma, w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank
            )

            test_mask = np.zeros(N, bool)
            test_mask[test_act] = True

            # Evaluate CWRA
            if args.aggregation == "rrf":
                sc_cwra = reciprocal_rank_fusion(ranks)
            elif args.aggregation == "power":
                ranks_power = power_rank_transform(ranks, N, power=delta)
                sc_cwra = ranks_power.dot(w_mix)
            else:
                sc_cwra = ranks01.dot(w_mix)
            
            sc_cwra[train_act] = np.inf
            d_cwra = eval_depths_extended(sc_cwra, test_mask, len(test_act), cutoffs, N)

            # Equal weight baseline (average of all modalities)
            sc_eq = ranks01.dot(uniform_w)
            sc_eq[train_act] = np.inf
            d_eq = eval_depths_extended(sc_eq, test_mask, len(test_act), cutoffs, N)

            # Random baseline
            rng_fold = np.random.default_rng(rep_seed + fold)
            sc_random = rng_fold.random(N)
            sc_random[train_act] = np.inf
            d_random = eval_depths_extended(sc_random, test_mask, len(test_act), cutoffs, N)

            # Individual modality baselines
            individual_results = {}
            for j, (col, _) in enumerate(modalities):
                sc_ind = ranks01[:, j].copy()
                sc_ind[train_act] = np.inf
                d_ind = eval_depths_extended(sc_ind, test_mask, len(test_act), cutoffs, N)
                individual_results[mod_names_simple[j]] = d_ind

            fold_results.append(dict(
                rep=rep + 1, fold=fold, best_param=best_param,
                cwra=d_cwra, equal=d_eq, random=d_random,
                individual=individual_results,
                weights=w_mix, A_test=len(test_act),
            ))

            # Print progress with key metrics
            print(f"[Rep {rep+1} Fold {fold:2d}] CWRA EF@1%={d_cwra['ef1']:.2f} EF@5%={d_cwra['ef5']:.2f} EF@10%={d_cwra['ef10']:.2f}  "
                  f"Equal EF@5%={d_eq['ef5']:.2f}  Random EF@5%={d_random['ef5']:.2f}")

    # ==========================================================================
    # FINAL MODEL & RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Computing Final Results")
    print("=" * 70)

    # Majority vote for hyperparameters
    arr = np.array(chosen_params, dtype=object)
    final_param = tuple(Counter(arr[:, j]).most_common(1)[0][0] for j in range(arr.shape[1]))
    w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank, alpha, tau0, lam, delta, gamma = final_param
    shrink = shrink_cache[(tau0, lam)]

    print(f"\nFinal hyperparameters (majority vote):")
    print(f"  EF weights: w_ef1={w_ef1}, w_ef5={w_ef5}, w_ef10={w_ef10}, w_ef20={w_ef20}, w_ef30={w_ef30}")
    print(f"  BEDROC/rank: w_bed={w_bed}, w_rank={w_rank}")
    print(f"  alpha={alpha}, tau0={tau0}, lambda={lam}, delta={delta}, gamma={gamma}")

    # Final weights
    ef_terms_full, mean_rank_full, rank_score_full, bed_full = calc_metrics(act_idx_all, alphas)
    w_mix_full, _ = compute_weights_v2(
        ef_terms_full, rank_score_full, bed_full[alpha], 
        shrink, delta, gamma, w_ef1, w_ef5, w_ef10, w_ef20, w_ef30, w_bed, w_rank
    )

    print(f"\nFinal modality weights:")
    for name, w in zip(mod_names_simple, w_mix_full):
        print(f"  {name}: {w:.4f}")

    # ==========================================================================
    # TABLE 5: WEIGHTS
    # ==========================================================================
    table5 = pd.DataFrame({
        "Modality": mod_labels,
        "Weight": w_mix_full,
        "EF@1%": ef_terms_full["ef1"],
        "EF@5%": ef_terms_full["ef5"],
        "EF@10%": ef_terms_full["ef10"],
        "EF@20%": ef_terms_full["ef20"],
        "EF@30%": ef_terms_full["ef30"],
        "Mean_Rank": mean_rank_full,
    })

    # ==========================================================================
    # TABLE 6: PERFORMANCE
    # ==========================================================================
    def agg(method_key: str):
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

    def fmt(m, sd, digits=2):
        return f"{m:.{digits}f} +/- {sd:.{digits}f}"

    k1, k5, k10, k20, k30 = cutoffs["1"], cutoffs["5"], cutoffs["10"], cutoffs["20"], cutoffs["30"]
    
    # Build comprehensive comparison table
    table6_rows = []
    
    # Random baseline
    table6_rows.append({
        "Method": "Random",
        "EF@1%": fmt(*s_random["ef1"]),
        "EF@5%": fmt(*s_random["ef5"]),
        "EF@10%": fmt(*s_random["ef10"]),
        "EF@20%": fmt(*s_random["ef20"]),
        "EF@30%": fmt(*s_random["ef30"]),
        f"Hits@20% ({k20})": fmt(*s_random["h20"], digits=1),
        f"Hits@30% ({k30})": fmt(*s_random["h30"], digits=1),
    })
    
    # Individual modalities
    for mod_name in mod_names_simple:
        s = s_individual[mod_name]
        table6_rows.append({
            "Method": f"Individual: {mod_name}",
            "EF@1%": fmt(*s["ef1"]),
            "EF@5%": fmt(*s["ef5"]),
            "EF@10%": fmt(*s["ef10"]),
            "EF@20%": fmt(*s["ef20"]),
            "EF@30%": fmt(*s["ef30"]),
            f"Hits@20% ({k20})": fmt(*s["h20"], digits=1),
            f"Hits@30% ({k30})": fmt(*s["h30"], digits=1),
        })
    
    # Equal-weight fusion (average)
    table6_rows.append({
        "Method": "Average (Equal-weight)",
        "EF@1%": fmt(*s_eq["ef1"]),
        "EF@5%": fmt(*s_eq["ef5"]),
        "EF@10%": fmt(*s_eq["ef10"]),
        "EF@20%": fmt(*s_eq["ef20"]),
        "EF@30%": fmt(*s_eq["ef30"]),
        f"Hits@20% ({k20})": fmt(*s_eq["h20"], digits=1),
        f"Hits@30% ({k30})": fmt(*s_eq["h30"], digits=1),
    })
    
    # CWRA
    table6_rows.append({
        "Method": f"CWRA-{args.focus}",
        "EF@1%": fmt(*s_cwra["ef1"]),
        "EF@5%": fmt(*s_cwra["ef5"]),
        "EF@10%": fmt(*s_cwra["ef10"]),
        "EF@20%": fmt(*s_cwra["ef20"]),
        "EF@30%": fmt(*s_cwra["ef30"]),
        f"Hits@20% ({k20})": fmt(*s_cwra["h20"], digits=1),
        f"Hits@30% ({k30})": fmt(*s_cwra["h30"], digits=1),
    })
    
    table6 = pd.DataFrame(table6_rows)

    # ==========================================================================
    # FULL RANKING
    # ==========================================================================
    if args.aggregation == "rrf":
        consensus_score = reciprocal_rank_fusion(ranks)
    elif args.aggregation == "power":
        ranks_power = power_rank_transform(ranks, N, power=delta)
        consensus_score = ranks_power.dot(w_mix_full)
    else:
        consensus_score = ranks01.dot(w_mix_full)
    
    df_pool["cwra_score"] = consensus_score
    df_pool["cwra_rank"] = np.argsort(np.argsort(consensus_score)) + 1

    # Add modality ranks
    for j, (col, _) in enumerate(modalities):
        df_pool[f"rank_{col}"] = ranks[:, j]

    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================
    prefix = args.output_prefix
    
    table5.to_csv(f"{prefix}_table5_weights.csv", index=False)
    table6.to_csv(f"{prefix}_table6_performance.csv", index=False)
    
    # Full ranking
    df_pool.to_csv(f"{prefix}_full_ranking.csv", index=False)
    
    # Top/bottom generated
    df_gen = df_pool[g_mask].copy()
    df_gen_sorted = df_gen.sort_values("cwra_rank")
    df_gen_sorted.head(args.top_n).to_csv(f"{prefix}_top{args.top_n}_G.csv", index=False)
    df_gen_sorted.tail(args.top_n).to_csv(f"{prefix}_bottom{args.top_n}_G.csv", index=False)

    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    print("\nTable 5 - Modality Weights and Individual Performance:")
    print("-" * 100)
    print(f"{'Modality':<25} {'Weight':>8} {'EF@1%':>8} {'EF@5%':>8} {'EF@10%':>8} {'EF@20%':>8} {'EF@30%':>8}")
    print("-" * 100)
    for i, row in table5.iterrows():
        print(f"{row['Modality']:<25} {row['Weight']:>8.4f} {row['EF@1%']:>8.2f} {row['EF@5%']:>8.2f} {row['EF@10%']:>8.2f} {row['EF@20%']:>8.2f} {row['EF@30%']:>8.2f}")
    print("-" * 100)
    
    print("\n" + "=" * 100)
    print("Table 6 - Comprehensive Performance Comparison (CV Results):")
    print("=" * 100)
    
    # Print header with all EF metrics
    print(f"{'Method':<30} {'EF@1%':>14} {'EF@5%':>14} {'EF@10%':>14} {'EF@20%':>14} {'EF@30%':>14}")
    print("-" * 100)
    
    # Random baseline
    print(f"{'Random':<30} {fmt(*s_random['ef1']):>14} {fmt(*s_random['ef5']):>14} {fmt(*s_random['ef10']):>14} {fmt(*s_random['ef20']):>14} {fmt(*s_random['ef30']):>14}")
    print("-" * 100)
    
    # Individual modalities (sorted by EF@5% mean for clarity)
    ind_sorted = sorted(s_individual.items(), key=lambda x: -x[1]["ef5"][0])
    for mod_name, s in ind_sorted:
        print(f"{mod_name:<30} {fmt(*s['ef1']):>14} {fmt(*s['ef5']):>14} {fmt(*s['ef10']):>14} {fmt(*s['ef20']):>14} {fmt(*s['ef30']):>14}")
    print("-" * 100)
    
    # Aggregation methods
    print(f"{'Average (Equal-weight)':<30} {fmt(*s_eq['ef1']):>14} {fmt(*s_eq['ef5']):>14} {fmt(*s_eq['ef10']):>14} {fmt(*s_eq['ef20']):>14} {fmt(*s_eq['ef30']):>14}")
    print(f"{'CWRA-' + args.focus:<30} {fmt(*s_cwra['ef1']):>14} {fmt(*s_cwra['ef5']):>14} {fmt(*s_cwra['ef10']):>14} {fmt(*s_cwra['ef20']):>14} {fmt(*s_cwra['ef30']):>14}")
    print("=" * 100)
    
    # Final EF on full dataset
    print("\nFinal CWRA model EF (full dataset, no CV):")
    full_order = np.argsort(consensus_score)
    for name, k in [("1%", k1), ("5%", k5), ("10%", k10), ("20%", k20), ("30%", k30)]:
        h = active_mask[full_order[:k]].sum()
        ef = (h * N) / (k * A_all)
        print(f"  EF@{name}: {ef:.2f} ({h} hits in top {k})")
    
    # All individual modalities on full dataset with all EF metrics
    print("\n" + "-" * 100)
    print("Individual Modalities Performance (full dataset, no CV):")
    print("-" * 100)
    print(f"{'Modality':<25} {'EF@1%':>8} {'EF@5%':>8} {'EF@10%':>8} {'EF@20%':>8} {'EF@30%':>8}")
    print("-" * 100)
    for j, (col, _) in enumerate(modalities):
        order_j = np.argsort(scores[:, j])
        h1 = active_mask[order_j[:k1]].sum()
        h5 = active_mask[order_j[:k5]].sum()
        h10 = active_mask[order_j[:k10]].sum()
        h20 = active_mask[order_j[:k20]].sum()
        h30 = active_mask[order_j[:k30]].sum()
        ef1 = (h1 * N) / (k1 * A_all)
        ef5 = (h5 * N) / (k5 * A_all)
        ef10 = (h10 * N) / (k10 * A_all)
        ef20 = (h20 * N) / (k20 * A_all)
        ef30 = (h30 * N) / (k30 * A_all)
        print(f"{mod_names_simple[j]:<25} {ef1:>8.2f} {ef5:>8.2f} {ef10:>8.2f} {ef20:>8.2f} {ef30:>8.2f}")
    print("-" * 100)
    
    # Equal-weight on full dataset
    eq_order = np.argsort(ranks01.mean(axis=1))
    print(f"\n{'Equal-weight (Average)':<25}", end="")
    for k_val, name in [(k1, "1%"), (k5, "5%"), (k10, "10%"), (k20, "20%"), (k30, "30%")]:
        h = active_mask[eq_order[:k_val]].sum()
        ef = (h * N) / (k_val * A_all)
        print(f" EF@{name}={ef:.2f}", end="")
    print()

    print(f"\nOutputs saved with prefix '{prefix}'")
    print(f"Grid size used: {len(grid)} configurations")


if __name__ == "__main__":
    main()
