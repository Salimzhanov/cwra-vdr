#!/usr/bin/env python3
"""
CWRA with Train/Test Split or K-Fold CV for Honest Evaluation
==============================================================
Extends CWRA with a single train/test split (or k-fold CV) over known
actives to produce unbiased enrichment estimates. Weights are optimized
on the train actives and evaluated on held-out test actives.

Key differences from original:
- Actives are split into train/test (or k folds)
- Weights are optimized using ONLY train actives as positives
- Enrichment is reported on BOTH train (in-sample) and test (out-of-sample)
- All non-active (generated) compounds remain in both pools

Usage:
    python cwra_cv.py --input data.csv --output results/
    python cwra_cv.py --input data.csv --train-frac 0.5 --seed 42
    python cwra_cv.py --input data.csv --cv-folds 5 --seed 42
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
import warnings
import time

warnings.filterwarnings('ignore')

def _print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    if not rows:
        return
    print(f"\n--- {title} ---")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    header_line = "  ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(f"{str(row[i]):<{widths[i]}}" for i in range(len(headers))))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CWRAConfig:
    """Configuration for CWRA optimization."""

    modalities: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        'graphdta_kd': ('high', 'GraphDTA_Kd'),
        'graphdta_ki': ('high', 'GraphDTA_Ki'),
        'graphdta_ic50': ('high', 'GraphDTA_IC50'),
        'mltle_pKd': ('high', 'MLTLE_pKd'),
        'vina_score': ('low', 'Vina'),
        'boltz_affinity': ('low', 'Boltz_affinity'),
        'tankbind_affinity': ('low', 'TankBind'),
        'drugban_affinity': ('low', 'DrugBAN'),
        'moltrans_affinity': ('low', 'MolTrans'),
    })

    active_sources: List[str] = field(default_factory=lambda: ['initial_370', 'calcitriol'])
    exclude_sources: List[str] = field(default_factory=lambda: ['newRef_137'])

    method: str = 'fair'

    min_weight: float = 0.03
    max_weight: float = 0.4

    entropy_weight: float = 0.5

    top_k: int = 7

    cutoff_weights: Dict[int, float] = field(default_factory=lambda: {1: 5, 5: 2, 10: 1})

    de_maxiter: int = 500
    de_seed: int = 42

    n_random_trials: int = 100

    # --- Train/test split settings (single split only) ---
    train_frac: float = 0.7       # fraction of actives used for training
    split_seed: int = 42          # seed for the active split

    output_prefix: str = 'cwra_cv'


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def normalize_modalities(df: pd.DataFrame, modalities: Dict[str, Tuple[str, str]]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Normalize modality columns to [0, 1] range with consistent direction."""
    available_cols = []
    mod_names = []
    norm_data = []

    for col, (direction, name) in modalities.items():
        if col not in df.columns:
            continue

        x = df[col].values.copy().astype(float)
        x = np.nan_to_num(x, nan=np.nanmean(x[~np.isnan(x)]))

        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            normalized = (x - xmin) / (xmax - xmin)
        else:
            normalized = np.zeros_like(x)

        if direction == 'low':
            normalized = 1 - normalized

        norm_data.append(normalized)
        available_cols.append(col)
        mod_names.append(name)

    return np.column_stack(norm_data), available_cols, mod_names


def compute_ef(scores: np.ndarray, active_mask: np.ndarray, cutoff_pct: float) -> Tuple[float, int, int]:
    """Compute enrichment factor at given cutoff percentage."""
    N = len(scores)
    A = active_mask.sum()
    k = max(1, int(N * cutoff_pct / 100))

    top_k_idx = np.argsort(scores)[-k:]
    hits = active_mask[top_k_idx].sum()
    ef = (hits * N) / (k * A) if A > 0 else 0

    return ef, int(hits), k


def evaluate_weights(weights: np.ndarray, X: np.ndarray, active_mask: np.ndarray,
                     cutoffs: List[int] = None) -> Dict:
    """Evaluate weights and return performance dict."""
    if cutoffs is None:
        cutoffs = [1, 5, 10, 20, 30]
    scores = X @ weights
    results = {}
    for cutoff in cutoffs:
        ef, hits, k = compute_ef(scores, active_mask, cutoff)
        results[cutoff] = {'ef': ef, 'hits': hits, 'k': k}
    return results


# =============================================================================
# OBJECTIVE FUNCTIONS (unchanged from original)
# =============================================================================

def objective_unconstrained(weights, X, active_mask, cutoff_weights):
    scores = X @ weights
    total = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())
    return -total


def objective_fair(weights, X, active_mask, cutoff_weights):
    w_norm = weights / weights.sum()
    scores = X @ w_norm
    total = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())
    return -total


def objective_entropy(weights, X, active_mask, cutoff_weights, entropy_weight, n_mod):
    w_norm = weights / weights.sum()
    scores = X @ w_norm
    total_ef = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())
    w_safe = np.clip(w_norm, 1e-10, 1)
    entropy = -np.sum(w_safe * np.log(w_safe)) / np.log(n_mod)
    return -(total_ef + entropy_weight * total_ef * entropy)


# =============================================================================
# OPTIMIZATION METHODS (unchanged from original)
# =============================================================================

def optimize_unconstrained(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(0.01, 1.0)] * n_mod
    result = differential_evolution(
        objective_unconstrained, bounds,
        args=(X, active_mask, config.cutoff_weights),
        maxiter=config.de_maxiter, seed=config.de_seed,
        workers=1, polish=True
    )
    return result.x / result.x.sum()


def optimize_fair(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(config.min_weight, config.max_weight)] * n_mod
    result = differential_evolution(
        objective_fair, bounds,
        args=(X, active_mask, config.cutoff_weights),
        maxiter=config.de_maxiter, seed=config.de_seed,
        workers=1, polish=True
    )
    return result.x / result.x.sum()


def optimize_entropy(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(0.01, 1.0)] * n_mod
    result = differential_evolution(
        objective_entropy, bounds,
        args=(X, active_mask, config.cutoff_weights, config.entropy_weight, n_mod),
        maxiter=config.de_maxiter, seed=config.de_seed,
        workers=1, polish=True
    )
    return result.x / result.x.sum()


OPTIMIZERS = {
    'unconstrained': optimize_unconstrained,
    'fair': optimize_fair,
    'entropy': optimize_entropy,
}


# =============================================================================
# TRAIN/TEST SPLIT LOGIC
# =============================================================================

def split_actives(active_mask: np.ndarray, train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the active compounds into train and test sets.

    Returns two boolean masks (same length as active_mask):
      - train_active_mask: True for train actives, False otherwise
      - test_active_mask:  True for test actives, False otherwise

    All non-active compounds have False in both masks.
    The FULL pool (actives + non-actives) is used for ranking in both cases;
    only the label visibility changes.
    """
    rng = np.random.RandomState(seed)

    active_indices = np.where(active_mask)[0]
    n_actives = len(active_indices)
    n_train = int(n_actives * train_frac)

    shuffled = rng.permutation(active_indices)
    train_indices = shuffled[:n_train]
    test_indices = shuffled[n_train:]

    train_active_mask = np.zeros(len(active_mask), dtype=bool)
    test_active_mask = np.zeros(len(active_mask), dtype=bool)

    train_active_mask[train_indices] = True
    test_active_mask[test_indices] = True

    return train_active_mask, test_active_mask


def make_cv_folds(active_mask: np.ndarray, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k-fold splits over the active compounds.

    Returns a list of (train_active_mask, test_active_mask) pairs.
    All non-active compounds have False in both masks.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 for cross-validation.")

    active_indices = np.where(active_mask)[0]
    n_actives = len(active_indices)
    if n_folds > n_actives:
        raise ValueError("n_folds cannot exceed number of actives.")

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(active_indices)
    fold_indices = np.array_split(shuffled, n_folds)

    folds = []
    for test_idx in fold_indices:
        train_mask = active_mask.copy()
        test_mask = np.zeros_like(active_mask, dtype=bool)
        test_mask[test_idx] = True
        train_mask[test_idx] = False
        folds.append((train_mask, test_mask))

    return folds


# =============================================================================
# BASELINE COMPUTATION
# =============================================================================

def compute_baselines(X: np.ndarray, active_mask: np.ndarray, mod_names: List[str],
                      config: CWRAConfig) -> Dict:
    """Compute all baseline performances against a given active mask."""
    n_mod = X.shape[1]
    baselines = {}

    # Equal weights
    weights_equal = np.ones(n_mod) / n_mod
    baselines['equal'] = {
        'weights': weights_equal,
        'performance': evaluate_weights(weights_equal, X, active_mask),
        'name': 'Equal Weights'
    }

    # Random weights (averaged)
    np.random.seed(config.de_seed)
    random_perfs = {c: [] for c in [1, 5, 10]}
    for _ in range(config.n_random_trials):
        w_rand = np.random.dirichlet(np.ones(n_mod))
        perf = evaluate_weights(w_rand, X, active_mask)
        for c in [1, 5, 10]:
            random_perfs[c].append(perf[c]['ef'])

    N = len(active_mask)
    A = active_mask.sum()
    perf_random = {}
    for c in [1, 5, 10, 20, 30]:
        if c in [1, 5, 10]:
            mean_ef = np.mean(random_perfs[c])
            k = max(1, int(N * c / 100))
            est_hits = int(mean_ef * k * A / N)
            perf_random[c] = {'ef': mean_ef, 'hits': est_hits, 'k': k}
        else:
            perf_random[c] = baselines['equal']['performance'][c]

    baselines['random'] = {
        'weights': None,
        'performance': perf_random,
        'name': f'Random (mean of {config.n_random_trials})'
    }

    # Individual modalities
    best_ef1 = 0
    best_name = None
    for i, name in enumerate(mod_names):
        weights_i = np.zeros(n_mod)
        weights_i[i] = 1.0
        perf = evaluate_weights(weights_i, X, active_mask)
        baselines[f'individual_{name}'] = {
            'weights': weights_i,
            'performance': perf,
            'name': f'Individual: {name}'
        }
        if perf[1]['ef'] > best_ef1:
            best_ef1 = perf[1]['ef']
            best_name = name

    baselines['best_individual'] = baselines[f'individual_{best_name}']
    baselines['best_individual']['name'] = f'Best Individual ({best_name})'

    return baselines


# =============================================================================
# MAIN CWRA CLASS WITH TRAIN/TEST SPLIT
# =============================================================================

class CWRATrainTest:
    """
    CWRA with train/test split for honest evaluation.

    Workflow:
      1. Normalize the full compound pool (actives + generated).
      2. Split the actives into train/test per train_frac.
      3. Optimize modality weights using ONLY train actives as positives.
         (All non-actives remain in the pool — they're just unlabeled.)
      4. Evaluate enrichment on BOTH train actives (in-sample) and
         test actives (out-of-sample) using the same optimized weights.
      5. Also report original (all-actives) performance for reference.
    """

    def __init__(self, config: Optional[CWRAConfig] = None):
        self.config = config or CWRAConfig()
        self.weights_ = None
        self.available_cols_ = None
        self.mod_names_ = None
        # Store results for train, test, and full (reference)
        self.train_performance_ = None
        self.test_performance_ = None
        self.full_performance_ = None
        self.baselines_train_ = None
        self.baselines_test_ = None
        self.baselines_full_ = None
        self.all_methods_ = None
        self.split_info_ = None

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> 'CWRATrainTest':
        """Fit CWRA with train/test split."""
        t0 = time.time()

        if verbose:
            print("=" * 80)
            print("CWRA WITH TRAIN/TEST SPLIT")
            print("=" * 80)

        # ----- Prepare full pool -----
        df_pool = df[~df['source'].isin(self.config.exclude_sources)].copy()
        full_active_mask = df_pool['source'].isin(self.config.active_sources).values

        X, self.available_cols_, self.mod_names_ = normalize_modalities(
            df_pool, self.config.modalities
        )

        N, n_mod = X.shape
        A_full = full_active_mask.sum()

        if verbose:
            print(f"\nDataset: {N:,} compounds, {A_full} actives ({100*A_full/N:.2f}%)")
            print(f"Modalities: {n_mod}")
            print(f"Method: {self.config.method}")
            print(f"Train fraction: {self.config.train_frac}")
            print(f"Split seed: {self.config.split_seed}")

        # ----- Split actives -----
        train_active_mask, test_active_mask = split_actives(
            full_active_mask, self.config.train_frac, self.config.split_seed
        )

        A_train = train_active_mask.sum()
        A_test = test_active_mask.sum()

        self.split_info_ = {
            'n_total': N,
            'n_actives_full': int(A_full),
            'n_actives_train': int(A_train),
            'n_actives_test': int(A_test),
            'n_inactives': int(N - A_full),
            'train_frac': self.config.train_frac,
            'split_seed': self.config.split_seed,
        }

        if verbose:
            print(f"\nSplit: {A_train} train actives / {A_test} test actives "
                  f"(out of {A_full} total)")
            print(f"Pool size for ranking: {N:,} (all compounds, same for train & test eval)")

        # ----- Optimize weights on TRAIN actives only -----
        if verbose:
            print(f"\n--- Optimizing weights on TRAIN actives ({A_train}) ---")

        self.all_methods_ = {}

        for method_name in ['unconstrained', 'fair', 'entropy']:
            if verbose:
                print(f"  Running {method_name} optimization...")

            optimizer = OPTIMIZERS[method_name]
            # KEY: optimize using train_active_mask, not full_active_mask
            weights = optimizer(X, train_active_mask, self.config)

            # Evaluate on train, test, and full
            perf_train = evaluate_weights(weights, X, train_active_mask)
            perf_test = evaluate_weights(weights, X, test_active_mask)
            perf_full = evaluate_weights(weights, X, full_active_mask)

            n_sig = int(np.sum(weights > 0.05))

            self.all_methods_[method_name] = {
                'weights': weights,
                'perf_train': perf_train,
                'perf_test': perf_test,
                'perf_full': perf_full,
                'n_significant': n_sig,
            }

        # ----- Select final weights based on config method -----
        chosen = self.config.method
        if chosen not in self.all_methods_:
            chosen = 'fair'

        self.weights_ = self.all_methods_[chosen]['weights']
        self.train_performance_ = self.all_methods_[chosen]['perf_train']
        self.test_performance_ = self.all_methods_[chosen]['perf_test']
        self.full_performance_ = self.all_methods_[chosen]['perf_full']

        # ----- Compute baselines (evaluated on train, test, and full) -----
        if verbose:
            print("\nComputing baselines...")

        self.baselines_train_ = compute_baselines(X, train_active_mask, self.mod_names_, self.config)
        self.baselines_test_ = compute_baselines(X, test_active_mask, self.mod_names_, self.config)
        self.baselines_full_ = compute_baselines(X, full_active_mask, self.mod_names_, self.config)

        # ----- Print summary -----
        elapsed = time.time() - t0

        if verbose:
            print(f"\n{'='*80}")
            print(f"RESULTS (method: {chosen}) — elapsed: {elapsed:.1f}s")
            print(f"{'='*80}")

            header = f"{'Cutoff':>8}  {'TRAIN EF':>10} {'(hits)':>7}  {'TEST EF':>10} {'(hits)':>7}  {'FULL EF':>10} {'(hits)':>7}"
            print(header)
            print("-" * len(header))

            for c in [1, 5, 10, 20, 30]:
                tr = self.train_performance_[c]
                te = self.test_performance_[c]
                fu = self.full_performance_[c]
                print(f"  @{c:>2}%   "
                      f"  {tr['ef']:>8.2f} ({tr['hits']:>4})   "
                      f"  {te['ef']:>8.2f} ({te['hits']:>4})   "
                      f"  {fu['ef']:>8.2f} ({fu['hits']:>4})")

            # Overfit gap
            print(f"\n--- Overfitting diagnostics ---")
            for c in [1, 5, 10]:
                tr_ef = self.train_performance_[c]['ef']
                te_ef = self.test_performance_[c]['ef']
                gap = tr_ef - te_ef
                gap_pct = 100 * gap / tr_ef if tr_ef > 0 else 0
                print(f"  EF@{c}%  train={tr_ef:.2f}  test={te_ef:.2f}  "
                      f"gap={gap:.2f} ({gap_pct:.1f}% drop)")

            # Compare to baselines on test set
            print(f"\n--- Test-set comparison (honest estimates) ---")
            eq_ef1 = self.baselines_test_['equal']['performance'][1]['ef']
            bi_ef1 = self.baselines_test_['best_individual']['performance'][1]['ef']
            te_ef1 = self.test_performance_[1]['ef']
            print(f"  Equal-weight  EF@1%: {eq_ef1:.2f}")
            print(f"  Best-indiv    EF@1%: {bi_ef1:.2f} "
                  f"({self.baselines_test_['best_individual']['name']})")
            print(f"  CWRA (test)   EF@1%: {te_ef1:.2f}")
            if eq_ef1 > 0:
                print(f"  CWRA vs equal-weight: {100*(te_ef1/eq_ef1 - 1):+.1f}%")
            if bi_ef1 > 0:
                print(f"  CWRA vs best-indiv:   {100*(te_ef1/bi_ef1 - 1):+.1f}%")

            # Weights
            print(f"\nOptimized weights:")
            order = np.argsort(-self.weights_)
            for idx in order:
                print(f"  {self.mod_names_[idx]:>20s}: {self.weights_[idx]:.4f} "
                      f"({100*self.weights_[idx]:.1f}%)")

            h_norm = self._normalized_entropy(self.weights_)
            print(f"\n  Normalized entropy: {h_norm:.3f}")
            print(f"  Significant (>5%): {np.sum(self.weights_ > 0.05)}/{n_mod}")

            # Individual modalities (test set)
            indiv_df = self.get_individual_modality_table()
            indiv_rows = []
            for _, row in indiv_df.iterrows():
                indiv_rows.append([
                    row['modality'],
                    f"{row['test_ef_1']:.2f}",
                    f"{row['test_ef_5']:.2f}",
                    f"{row['test_ef_10']:.2f}",
                ])
            _print_table(
                "Individual modality performance (test set)",
                ["Modality", "EF@1%", "EF@5%", "EF@10%"],
                indiv_rows,
            )

            # Combination methods (test set)
            combo_rows = []
            method_labels = {
                'fair': 'DE Fair',
                'unconstrained': 'DE Unconstrained',
                'entropy': 'DE Entropy',
            }
            for key in ['fair', 'unconstrained', 'entropy']:
                perf = self.all_methods_[key]['perf_test']
                combo_rows.append([
                    method_labels[key],
                    f"{perf[1]['ef']:.2f}",
                    f"{perf[5]['ef']:.2f}",
                    f"{perf[10]['ef']:.2f}",
                ])

            eq = self.baselines_test_['equal']['performance']
            combo_rows.append([
                "Equal Weights",
                f"{eq[1]['ef']:.2f}",
                f"{eq[5]['ef']:.2f}",
                f"{eq[10]['ef']:.2f}",
            ])
            rnd = self.baselines_test_['random']['performance']
            combo_rows.append([
                self.baselines_test_['random']['name'],
                f"{rnd[1]['ef']:.2f}",
                f"{rnd[5]['ef']:.2f}",
                f"{rnd[10]['ef']:.2f}",
            ])

            _print_table(
                "Combination methods (test set)",
                ["Method", "EF@1%", "EF@5%", "EF@10%"],
                combo_rows,
            )

        # Store internals for transform
        self._X = X
        self._df_pool = df_pool
        self._full_active_mask = full_active_mask
        self._train_active_mask = train_active_mask
        self._test_active_mask = test_active_mask

        return self

    def _normalized_entropy(self, w: np.ndarray) -> float:
        """Compute normalized Shannon entropy of weight vector."""
        w_safe = np.clip(w, 1e-10, 1)
        return float(-np.sum(w_safe * np.log(w_safe)) / np.log(len(w)))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted weights to compute rankings."""
        if self.weights_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        df_result = df.copy()
        mask = ~df_result['source'].isin(self.config.exclude_sources)
        df_pool = df_result[mask].copy()

        X, _, _ = normalize_modalities(df_pool, self.config.modalities)
        scores = X @ self.weights_

        df_pool['cwra_score'] = scores
        df_pool['cwra_rank'] = stats.rankdata(-scores, method='ordinal')

        df_result.loc[mask, 'cwra_score'] = df_pool['cwra_score'].values
        df_result.loc[mask, 'cwra_rank'] = df_pool['cwra_rank'].values

        return df_result

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Comprehensive comparison table with train, test, and full columns.
        """
        rows = []

        # --- Baselines (use test-set evaluation for honest comparison) ---
        for key in ['equal', 'random', 'best_individual']:
            bt = self.baselines_train_[key]
            be = self.baselines_test_[key]
            bf = self.baselines_full_[key]
            row = {'method': bt['name']}
            for c in [1, 5, 10]:
                row[f'train_ef_{c}'] = bt['performance'][c]['ef']
                row[f'train_hits_{c}'] = bt['performance'][c]['hits']
                row[f'test_ef_{c}'] = be['performance'][c]['ef']
                row[f'test_hits_{c}'] = be['performance'][c]['hits']
                row[f'full_ef_{c}'] = bf['performance'][c]['ef']
                row[f'full_hits_{c}'] = bf['performance'][c]['hits']
            rows.append(row)

        # --- Optimized methods ---
        method_labels = {
            'unconstrained': 'DE Unconstrained',
            'fair': 'DE Fair (3-25%)',
            'entropy': 'DE Entropy',
        }

        for key, label in method_labels.items():
            m = self.all_methods_[key]
            row = {'method': label}
            for c in [1, 5, 10]:
                row[f'train_ef_{c}'] = m['perf_train'][c]['ef']
                row[f'train_hits_{c}'] = m['perf_train'][c]['hits']
                row[f'test_ef_{c}'] = m['perf_test'][c]['ef']
                row[f'test_hits_{c}'] = m['perf_test'][c]['hits']
                row[f'full_ef_{c}'] = m['perf_full'][c]['ef']
                row[f'full_hits_{c}'] = m['perf_full'][c]['hits']
            rows.append(row)

        return pd.DataFrame(rows)

    def get_individual_modality_table(self) -> pd.DataFrame:
        """Individual modality performance on train, test, full."""
        rows = []
        for name in self.mod_names_:
            key = f'individual_{name}'
            if key in self.baselines_train_:
                bt = self.baselines_train_[key]
                be = self.baselines_test_[key]
                bf = self.baselines_full_[key]
                row = {'modality': name}
                for c in [1, 5, 10]:
                    row[f'train_ef_{c}'] = bt['performance'][c]['ef']
                    row[f'test_ef_{c}'] = be['performance'][c]['ef']
                    row[f'full_ef_{c}'] = bf['performance'][c]['ef']
                rows.append(row)
        return pd.DataFrame(rows).sort_values('test_ef_1', ascending=False)

    def save_results(self, output_dir: str):
        """Save all results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.config.output_prefix

        # Weights
        weights_df = pd.DataFrame({
            'column': self.available_cols_,
            'modality': self.mod_names_,
            'weight': self.weights_,
            'weight_pct': self.weights_ * 100
        }).sort_values('weight', ascending=False)
        weights_df.to_csv(output_dir / f'{prefix}_weights.csv', index=False)

        # Comparison table
        comp_df = self.get_comparison_table()
        comp_df.to_csv(output_dir / f'{prefix}_comparison.csv', index=False)

        # Individual modality table
        indiv_df = self.get_individual_modality_table()
        indiv_df.to_csv(output_dir / f'{prefix}_individual_modalities.csv', index=False)

        # Train/test/full performance for the selected method
        perf_rows = []
        for c in [1, 5, 10, 20, 30]:
            perf_rows.append({
                'cutoff_pct': c,
                'train_ef': self.train_performance_[c]['ef'],
                'train_hits': self.train_performance_[c]['hits'],
                'test_ef': self.test_performance_[c]['ef'],
                'test_hits': self.test_performance_[c]['hits'],
                'full_ef': self.full_performance_[c]['ef'],
                'full_hits': self.full_performance_[c]['hits'],
                'top_k': self.train_performance_[c]['k'],
            })
        perf_df = pd.DataFrame(perf_rows)
        perf_df.to_csv(output_dir / f'{prefix}_performance.csv', index=False)

        # Split info
        split_df = pd.DataFrame([self.split_info_])
        split_df.to_csv(output_dir / f'{prefix}_split_info.csv', index=False)

        print(f"\nResults saved to {output_dir}/")


# =============================================================================
# K-FOLD CROSS-VALIDATION
# =============================================================================

def run_cross_validation(df: pd.DataFrame, config: CWRAConfig, n_folds: int,
                         output_dir: Optional[Path] = None, verbose: bool = True):
    """Run k-fold CV over actives and save per-fold + summary outputs."""
    t0 = time.time()
    def _fmt_pm(mean: float, std: float, digits: int = 2) -> str:
        return f"{mean:.{digits}f} +/- {std:.{digits}f}"

    def _normalized_entropy(w: np.ndarray) -> float:
        w_safe = np.clip(w, 1e-10, 1)
        return float(-np.sum(w_safe * np.log(w_safe)) / np.log(len(w)))

    if verbose:
        print("=" * 80)
        print(f"CWRA K-FOLD CROSS-VALIDATION (k={n_folds})")
        print("=" * 80)

    df_pool = df[~df['source'].isin(config.exclude_sources)].copy()
    full_active_mask = df_pool['source'].isin(config.active_sources).values

    X, available_cols, mod_names = normalize_modalities(df_pool, config.modalities)
    N, n_mod = X.shape
    A_full = int(full_active_mask.sum())

    if verbose:
        print(f"\nDataset: {N:,} compounds, {A_full} actives ({100*A_full/N:.2f}%)")
        print(f"Modalities: {n_mod}")
        print(f"Method: {config.method}")
        print(f"CV folds: {n_folds}")
        print(f"CV seed: {config.split_seed}")

    folds = make_cv_folds(full_active_mask, n_folds, config.split_seed)

    perf_rows = []
    weight_rows = []
    baseline_rows = []
    fold_info_rows = []
    indiv_rows = []
    method_rows = []
    entropy_rows = []
    significant_rows = []

    chosen = config.method if config.method in OPTIMIZERS else 'fair'

    for fold_idx, (train_active_mask, test_active_mask) in enumerate(folds, start=1):
        A_train = int(train_active_mask.sum())
        A_test = int(test_active_mask.sum())

        fold_info_rows.append({
            'fold': fold_idx,
            'n_total': N,
            'n_actives_full': A_full,
            'n_actives_train': A_train,
            'n_actives_test': A_test,
            'n_inactives': int(N - A_full),
            'split_seed': config.split_seed,
        })

        if verbose:
            print(f"\nFold {fold_idx}/{n_folds}: {A_train} train actives / {A_test} test actives")
            print(f"--- Optimizing weights on TRAIN actives ({A_train}) ---")

        all_methods = {}
        for method_name in ['unconstrained', 'fair', 'entropy']:
            if verbose:
                print(f"  Running {method_name} optimization...")
            optimizer = OPTIMIZERS[method_name]
            weights = optimizer(X, train_active_mask, config)

            perf_train = evaluate_weights(weights, X, train_active_mask)
            perf_test = evaluate_weights(weights, X, test_active_mask)
            perf_full = evaluate_weights(weights, X, full_active_mask)

            all_methods[method_name] = {
                'weights': weights,
                'perf_train': perf_train,
                'perf_test': perf_test,
                'perf_full': perf_full,
            }

            method_rows.append({
                'fold': fold_idx,
                'method': method_name,
                'test_ef_1': perf_test[1]['ef'],
                'test_ef_5': perf_test[5]['ef'],
                'test_ef_10': perf_test[10]['ef'],
            })

        chosen_data = all_methods[chosen]

        for c in [1, 5, 10, 20, 30]:
            perf_rows.append({
                'fold': fold_idx,
                'method': chosen,
                'cutoff_pct': c,
                'train_ef': chosen_data['perf_train'][c]['ef'],
                'train_hits': chosen_data['perf_train'][c]['hits'],
                'test_ef': chosen_data['perf_test'][c]['ef'],
                'test_hits': chosen_data['perf_test'][c]['hits'],
                'full_ef': chosen_data['perf_full'][c]['ef'],
                'full_hits': chosen_data['perf_full'][c]['hits'],
                'top_k': chosen_data['perf_train'][c]['k'],
            })

        weights = chosen_data['weights']
        entropy_rows.append(_normalized_entropy(weights))
        significant_rows.append(int(np.sum(weights > 0.05)))
        for i, name in enumerate(mod_names):
            weight_rows.append({
                'fold': fold_idx,
                'method': chosen,
                'column': available_cols[i],
                'modality': name,
                'weight': weights[i],
                'weight_pct': weights[i] * 100,
            })

        if verbose:
            print("Computing baselines...")

        baselines_train = compute_baselines(X, train_active_mask, mod_names, config)
        baselines_test = compute_baselines(X, test_active_mask, mod_names, config)
        baselines_full = compute_baselines(X, full_active_mask, mod_names, config)

        for key in ['equal', 'random', 'best_individual']:
            bt = baselines_train[key]
            be = baselines_test[key]
            bf = baselines_full[key]
            row = {
                'fold': fold_idx,
                'baseline': be['name'],
            }
            for c in [1, 5, 10]:
                row[f'train_ef_{c}'] = bt['performance'][c]['ef']
                row[f'train_hits_{c}'] = bt['performance'][c]['hits']
                row[f'test_ef_{c}'] = be['performance'][c]['ef']
                row[f'test_hits_{c}'] = be['performance'][c]['hits']
                row[f'full_ef_{c}'] = bf['performance'][c]['ef']
                row[f'full_hits_{c}'] = bf['performance'][c]['hits']
            baseline_rows.append(row)

        for name in mod_names:
            key = f'individual_{name}'
            if key not in baselines_test:
                continue
            be = baselines_test[key]
            indiv_rows.append({
                'fold': fold_idx,
                'modality': name,
                'test_ef_1': be['performance'][1]['ef'],
                'test_ef_5': be['performance'][5]['ef'],
                'test_ef_10': be['performance'][10]['ef'],
            })

    perf_df = pd.DataFrame(perf_rows)
    weights_df = pd.DataFrame(weight_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    fold_info_df = pd.DataFrame(fold_info_rows)
    indiv_df = pd.DataFrame(indiv_rows)
    method_df = pd.DataFrame(method_rows)

    summary_df = (
        perf_df.groupby('cutoff_pct', as_index=False)
        .agg(
            train_ef_mean=('train_ef', 'mean'),
            train_ef_std=('train_ef', 'std'),
            test_ef_mean=('test_ef', 'mean'),
            test_ef_std=('test_ef', 'std'),
            full_ef_mean=('full_ef', 'mean'),
            full_ef_std=('full_ef', 'std'),
            train_hits_mean=('train_hits', 'mean'),
            test_hits_mean=('test_hits', 'mean'),
            full_hits_mean=('full_hits', 'mean'),
        )
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = config.output_prefix

        perf_df.to_csv(output_dir / f'{prefix}_cv_folds_performance.csv', index=False)
        weights_df.to_csv(output_dir / f'{prefix}_cv_folds_weights.csv', index=False)
        baseline_df.to_csv(output_dir / f'{prefix}_cv_folds_baselines.csv', index=False)
        fold_info_df.to_csv(output_dir / f'{prefix}_cv_folds_info.csv', index=False)
        summary_df.to_csv(output_dir / f'{prefix}_cv_summary.csv', index=False)

        if verbose:
            print(f"\nResults saved to {output_dir}/")

    if verbose:
        elapsed = time.time() - t0
        print(f"\n{'='*80}")
        print(f"RESULTS (method: {chosen}) — elapsed: {elapsed:.1f}s")
        print(f"{'='*80}")

        header = (
            f"{'Cutoff':>8}  {'TRAIN EF':>18} {'(hits)':>7}  "
            f"{'TEST EF':>18} {'(hits)':>7}  {'FULL EF':>18} {'(hits)':>7}"
        )
        print(header)
        print("-" * len(header))

        for _, row in summary_df.iterrows():
            c = int(row['cutoff_pct'])
            tr = _fmt_pm(row['train_ef_mean'], row['train_ef_std'])
            te = _fmt_pm(row['test_ef_mean'], row['test_ef_std'])
            fu = _fmt_pm(row['full_ef_mean'], row['full_ef_std'])
            tr_hits = f"{row['train_hits_mean']:.1f}"
            te_hits = f"{row['test_hits_mean']:.1f}"
            fu_hits = f"{row['full_hits_mean']:.1f}"
            print(
                f"  @{c:>2}%   "
                f"{tr:>18} ({tr_hits:>5})   "
                f"{te:>18} ({te_hits:>5})   "
                f"{fu:>18} ({fu_hits:>5})"
            )

        print(f"\n--- Overfitting diagnostics (means) ---")
        for c in [1, 5, 10]:
            row = summary_df.loc[summary_df['cutoff_pct'] == c].iloc[0]
            tr_ef = row['train_ef_mean']
            te_ef = row['test_ef_mean']
            gap = tr_ef - te_ef
            gap_pct = 100 * gap / tr_ef if tr_ef > 0 else 0
            print(
                f"  EF@{c}%  train={tr_ef:.2f}  test={te_ef:.2f}  "
                f"gap={gap:.2f} ({gap_pct:.1f}% drop)"
            )

        print(f"\n--- Test-set comparison (mean across folds) ---")
        baseline_summary = (
            baseline_df.groupby('baseline', as_index=False)
            .agg(test_ef_1_mean=('test_ef_1', 'mean'))
        )
        eq_row = baseline_summary[baseline_summary['baseline'] == 'Equal Weights']
        bi_row = baseline_summary[baseline_summary['baseline'].str.startswith('Best Individual')]
        eq_ef1 = float(eq_row['test_ef_1_mean'].iloc[0]) if not eq_row.empty else 0.0
        bi_ef1 = float(bi_row['test_ef_1_mean'].iloc[0]) if not bi_row.empty else 0.0
        bi_name = bi_row['baseline'].iloc[0] if not bi_row.empty else 'Best Individual'
        te_ef1 = float(summary_df.loc[summary_df['cutoff_pct'] == 1, 'test_ef_mean'].iloc[0])
        print(f"  Equal-weight  EF@1%: {eq_ef1:.2f}")
        print(f"  Best-indiv    EF@1%: {bi_ef1:.2f} ({bi_name})")
        print(f"  CWRA (test)   EF@1%: {te_ef1:.2f}")
        if eq_ef1 > 0:
            print(f"  CWRA vs equal-weight: {100*(te_ef1/eq_ef1 - 1):+.1f}%")
        if bi_ef1 > 0:
            print(f"  CWRA vs best-indiv:   {100*(te_ef1/bi_ef1 - 1):+.1f}%")

        print(f"\nOptimized weights (mean across folds):")
        weights_summary = (
            weights_df.groupby(['modality', 'column'], as_index=False)
            .agg(weight_pct_mean=('weight_pct', 'mean'))
            .sort_values('weight_pct_mean', ascending=False)
        )
        for _, row in weights_summary.iterrows():
            weight = row['weight_pct_mean'] / 100.0
            print(
                f"  {row['modality']:>20s}: {weight:.4f} "
                f"({row['weight_pct_mean']:.1f}%)"
            )

        h_mean = float(np.mean(entropy_rows)) if entropy_rows else 0.0
        h_std = float(np.std(entropy_rows, ddof=0)) if len(entropy_rows) > 1 else 0.0
        sig_mean = float(np.mean(significant_rows)) if significant_rows else 0.0
        sig_std = float(np.std(significant_rows, ddof=0)) if len(significant_rows) > 1 else 0.0
        print(f"\n  Normalized entropy: {_fmt_pm(h_mean, h_std, digits=3)}")
        print(f"  Significant (>5%): {_fmt_pm(sig_mean, sig_std, digits=2)}/{n_mod}")

        # Individual modalities (test set)
        if not indiv_df.empty:
            rows = []
            for name, grp in indiv_df.groupby('modality'):
                rows.append([
                    name,
                    _fmt_pm(grp['test_ef_1'].mean(), grp['test_ef_1'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_5'].mean(), grp['test_ef_5'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_10'].mean(), grp['test_ef_10'].std(ddof=0)),
                ])
            rows.sort(key=lambda r: float(r[1].split()[0]), reverse=True)
            _print_table(
                "Individual modality performance (test set, mean +/- std)",
                ["Modality", "EF@1%", "EF@5%", "EF@10%"],
                rows,
            )

        # Combination methods (test set)
        combo_rows = []
        method_labels = {
            'fair': 'DE Fair',
            'unconstrained': 'DE Unconstrained',
            'entropy': 'DE Entropy',
        }
        for key in ['fair', 'unconstrained', 'entropy']:
            grp = method_df[method_df['method'] == key]
            if grp.empty:
                continue
            combo_rows.append([
                method_labels[key],
                _fmt_pm(grp['test_ef_1'].mean(), grp['test_ef_1'].std(ddof=0)),
                _fmt_pm(grp['test_ef_5'].mean(), grp['test_ef_5'].std(ddof=0)),
                _fmt_pm(grp['test_ef_10'].mean(), grp['test_ef_10'].std(ddof=0)),
            ])

        for baseline_name in baseline_df['baseline'].unique():
            if baseline_name == 'Equal Weights' or baseline_name.startswith('Random'):
                grp = baseline_df[baseline_df['baseline'] == baseline_name]
                combo_rows.append([
                    baseline_name,
                    _fmt_pm(grp['test_ef_1'].mean(), grp['test_ef_1'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_5'].mean(), grp['test_ef_5'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_10'].mean(), grp['test_ef_10'].std(ddof=0)),
                ])

        if combo_rows:
            _print_table(
                "Combination methods (test set, mean +/- std)",
                ["Method", "EF@1%", "EF@5%", "EF@10%"],
                combo_rows,
            )

        print(f"\nDone! Elapsed: {elapsed:.1f}s")

    return perf_df, summary_df


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CWRA with Train/Test Split for Honest Evaluation'
    )

    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', default='cwra_cv_results', help='Output directory')
    parser.add_argument('--method', '-m', default='fair',
                        choices=['unconstrained', 'fair', 'entropy'],
                        help='Optimization method')
    parser.add_argument('--min-weight', type=float, default=0.03)
    parser.add_argument('--max-weight', type=float, default=0.25)
    parser.add_argument('--train-frac', type=float, default=0.7,
                        help='Fraction of actives for training (default: 0.7)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of CV folds over actives (>=2 enables CV)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for both DE and active split')

    args = parser.parse_args()

    config = CWRAConfig(
        method=args.method,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        de_seed=args.seed,
        split_seed=args.seed,
        train_frac=args.train_frac,
    )

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} compounds")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cv_folds and args.cv_folds > 1:
        run_cross_validation(df, config, args.cv_folds, output_dir=output_dir, verbose=True)
    else:
        cwra = CWRATrainTest(config)
        cwra.fit(df)

        df_ranked = cwra.transform(df)
        df_ranked.to_csv(output_dir / f'{config.output_prefix}_rankings.csv', index=False)
        cwra.save_results(output_dir)

        print("\nDone!")


if __name__ == '__main__':
    main()
