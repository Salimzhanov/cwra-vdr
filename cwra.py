#!/usr/bin/env python3
"""
CWRA Final Version -  Multi-Modality Ensemble
==================================================
Confidence-Weighted Rank Aggregation with balanced modality contributions.

Key Features:
- Weight constraints (min 3%, max 25% per modality)
- Multiple optimization strategies
- Comprehensive baseline comparisons
- Publication-ready outputs

Usage:
    python cwra_final.py --input data.csv --output results/
    python cwra_final.py --input data.csv --method fair --min-weight 0.03 --max-weight 0.25

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

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CWRAConfig:
    """Configuration for CWRA optimization."""

    # Modality definitions: column_name -> (direction, display_name)
    modalities: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        'graphdta_kd': ('high', 'GraphDTA_Kd'),
        'graphdta_ki': ('high', 'GraphDTA_Ki'),
        'graphdta_ic50': ('high', 'GraphDTA_IC50'),
        'mltle_pKd': ('high', 'MLTLE_pKd'),
        'vina_score': ('low', 'Vina'),
        'boltz_affinity': ('low', 'Boltz_affinity'),
        'boltz_confidence': ('high', 'Boltz_confidence'),
        'unimol_similarity': ('high', 'UniMol_sim'),
        'tankbind_affinity': ('low', 'TankBind'),
        'drugban_affinity': ('low', 'DrugBAN'),
        'moltrans_affinity': ('low', 'MolTrans'),
    })

    # Source definitions
    active_sources: List[str] = field(default_factory=lambda: ['initial_370', 'calcitriol'])
    exclude_sources: List[str] = field(default_factory=lambda: ['newRef_137'])

    # Optimization method: 'unconstrained', 'fair', 'entropy', 'topk'
    method: str = 'fair'

    # Weight constraints (for 'fair' method)
    min_weight: float = 0.03  # Minimum weight per modality
    max_weight: float = 0.25  # Maximum weight per modality

    # Entropy weight (for 'entropy' method)
    entropy_weight: float = 0.5

    # Top-K selection (for 'topk' method)
    top_k: int = 7

    # Cutoff weights for objective function
    cutoff_weights: Dict[int, float] = field(default_factory=lambda: {1: 5, 5: 2, 10: 1})

    # DE parameters
    de_maxiter: int = 500
    de_seed: int = 42

    # Random trials for baseline
    n_random_trials: int = 100

    # Output settings
    output_prefix: str = 'cwra'


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


def evaluate_weights(weights: np.ndarray, X: np.ndarray, active_mask: np.ndarray) -> Dict:
    """Evaluate weights and return performance dict."""
    scores = X @ weights
    results = {}
    for cutoff in [1, 5, 10, 20, 30]:
        ef, hits, k = compute_ef(scores, active_mask, cutoff)
        results[cutoff] = {'ef': ef, 'hits': hits, 'k': k}
    return results


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def objective_unconstrained(weights, X, active_mask, cutoff_weights):
    """Standard weighted cutoff objective."""
    scores = X @ weights
    total = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())
    return -total


def objective_fair(weights, X, active_mask, cutoff_weights):
    """ objective with normalized weights."""
    w_norm = weights / weights.sum()
    scores = X @ w_norm
    total = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())
    return -total


def objective_entropy(weights, X, active_mask, cutoff_weights, entropy_weight, n_mod):
    """Entropy-regularized objective for weight diversity."""
    w_norm = weights / weights.sum()
    scores = X @ w_norm

    # Performance component
    total_ef = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())

    # Entropy component (higher = more uniform)
    w_safe = np.clip(w_norm, 1e-10, 1)
    entropy = -np.sum(w_safe * np.log(w_safe)) / np.log(n_mod)

    return -(total_ef + entropy_weight * total_ef * entropy)


# =============================================================================
# OPTIMIZATION METHODS
# =============================================================================

def optimize_unconstrained(X: np.ndarray, active_mask: np.ndarray, config: CWRAConfig) -> np.ndarray:
    """Unconstrained differential evolution."""
    n_mod = X.shape[1]
    bounds = [(0.01, 1.0)] * n_mod

    result = differential_evolution(
        objective_unconstrained,
        bounds,
        args=(X, active_mask, config.cutoff_weights),
        maxiter=config.de_maxiter,
        seed=config.de_seed,
        workers=1,
        polish=True
    )
    return result.x / result.x.sum()


def optimize_fair(X: np.ndarray, active_mask: np.ndarray, config: CWRAConfig) -> np.ndarray:
    """ constrained optimization with min/max bounds."""
    n_mod = X.shape[1]
    bounds = [(config.min_weight, config.max_weight)] * n_mod

    result = differential_evolution(
        objective_fair,
        bounds,
        args=(X, active_mask, config.cutoff_weights),
        maxiter=config.de_maxiter,
        seed=config.de_seed,
        workers=1,
        polish=True
    )
    return result.x / result.x.sum()


def optimize_entropy(X: np.ndarray, active_mask: np.ndarray, config: CWRAConfig) -> np.ndarray:
    """Entropy-constrained optimization."""
    n_mod = X.shape[1]
    bounds = [(0.01, 1.0)] * n_mod

    result = differential_evolution(
        objective_entropy,
        bounds,
        args=(X, active_mask, config.cutoff_weights, config.entropy_weight, n_mod),
        maxiter=config.de_maxiter,
        seed=config.de_seed,
        workers=1,
        polish=True
    )
    return result.x / result.x.sum()


def optimize_topk(X: np.ndarray, active_mask: np.ndarray, config: CWRAConfig, mod_names: List[str]) -> np.ndarray:
    """Top-K modality selection optimization."""
    n_mod = X.shape[1]

    # Evaluate individual modalities
    individual_ef10 = []
    for i in range(n_mod):
        ef, _, _ = compute_ef(X[:, i], active_mask, 10)
        individual_ef10.append((i, ef))

    # Select top K
    individual_ef10.sort(key=lambda x: -x[1])
    top_k_idx = [x[0] for x in individual_ef10[:config.top_k]]

    X_topk = X[:, top_k_idx]
    bounds = [(0.05, 0.35)] * config.top_k

    result = differential_evolution(
        objective_fair,
        bounds,
        args=(X_topk, active_mask, config.cutoff_weights),
        maxiter=config.de_maxiter,
        seed=config.de_seed,
        workers=1,
        polish=True
    )

    # Expand to full weight vector
    weights_full = np.zeros(n_mod)
    weights_subset = result.x / result.x.sum()
    for i, idx in enumerate(top_k_idx):
        weights_full[idx] = weights_subset[i]

    return weights_full


OPTIMIZERS = {
    'unconstrained': optimize_unconstrained,
    'fair': optimize_fair,
    'entropy': optimize_entropy,
    'topk': optimize_topk,
}


# =============================================================================
# BASELINE METHODS
# =============================================================================

def compute_baselines(X: np.ndarray, active_mask: np.ndarray, mod_names: List[str],
                      config: CWRAConfig) -> Dict:
    """Compute all baseline performances."""
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
# MAIN CWRA CLASS
# =============================================================================

class CWRAFinal:
    """Final CWRA implementation with fair multi-modality ensemble."""

    def __init__(self, config: Optional[CWRAConfig] = None):
        self.config = config or CWRAConfig()
        self.weights_ = None
        self.available_cols_ = None
        self.mod_names_ = None
        self.performance_ = None
        self.baselines_ = None
        self.all_methods_ = None

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> 'CWRAFinal':
        """Fit CWRA weights with comprehensive baseline comparisons."""
        if verbose:
            print("=" * 80)
            print("CWRA FINAL -  Multi-Modality Ensemble")
            print("=" * 80)

        # Prepare data
        df_pool = df[~df['source'].isin(self.config.exclude_sources)].copy()
        active_mask = df_pool['source'].isin(self.config.active_sources).values

        X, self.available_cols_, self.mod_names_ = normalize_modalities(
            df_pool, self.config.modalities
        )

        N, n_mod = X.shape
        A = active_mask.sum()

        if verbose:
            print(f"\nDataset: {N:,} compounds, {A} actives ({100*A/N:.2f}%)")
            print(f"Modalities: {n_mod}")
            print(f"Method: {self.config.method}")

        # Compute baselines
        if verbose:
            print("\nComputing baselines...")
        self.baselines_ = compute_baselines(X, active_mask, self.mod_names_, self.config)

        # Run all optimization methods
        self.all_methods_ = {}

        for method_name in ['unconstrained', 'fair', 'entropy']:
            if verbose:
                print(f"Running {method_name} optimization...")

            if method_name == 'topk':
                weights = optimize_topk(X, active_mask, self.config, self.mod_names_)
            else:
                optimizer = OPTIMIZERS[method_name]
                weights = optimizer(X, active_mask, self.config)

            perf = evaluate_weights(weights, X, active_mask)
            n_sig = np.sum(weights > 0.05)

            self.all_methods_[method_name] = {
                'weights': weights,
                'performance': perf,
                'n_significant': n_sig
            }

        # Select final weights based on config method
        if self.config.method in self.all_methods_:
            self.weights_ = self.all_methods_[self.config.method]['weights']
            self.performance_ = self.all_methods_[self.config.method]['performance']
        else:
            self.weights_ = self.all_methods_['fair']['weights']
            self.performance_ = self.all_methods_['fair']['performance']

        if verbose:
            print("\nFinal Performance:")
            for c in [1, 5, 10]:
                ef = self.performance_[c]['ef']
                hits = self.performance_[c]['hits']
                print(f"  EF@{c}%: {ef:.2f} ({hits} hits)")

            print(f"\nSignificant modalities (>5%): {np.sum(self.weights_ > 0.05)}/11")

        # Store for transform
        self._X = X
        self._df_pool = df_pool
        self._active_mask = active_mask

        return self

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
        """Generate comprehensive comparison table."""
        rows = []

        # Add baselines
        for key in ['equal', 'random', 'best_individual']:
            b = self.baselines_[key]
            row = {
                'method': b['name'],
                'ef_1': b['performance'][1]['ef'],
                'hits_1': b['performance'][1]['hits'],
                'ef_5': b['performance'][5]['ef'],
                'hits_5': b['performance'][5]['hits'],
                'ef_10': b['performance'][10]['ef'],
                'hits_10': b['performance'][10]['hits'],
                'n_significant': 11 if key == 'equal' else (1 if key == 'best_individual' else None)
            }
            rows.append(row)

        # Add optimized methods
        method_names = {
            'unconstrained': 'DE Unconstrained',
            'fair': 'DE  (3-25%)',
            'entropy': 'DE Entropy-Constrained'
        }

        for key, name in method_names.items():
            m = self.all_methods_[key]
            row = {
                'method': name,
                'ef_1': m['performance'][1]['ef'],
                'hits_1': m['performance'][1]['hits'],
                'ef_5': m['performance'][5]['ef'],
                'hits_5': m['performance'][5]['hits'],
                'ef_10': m['performance'][10]['ef'],
                'hits_10': m['performance'][10]['hits'],
                'n_significant': m['n_significant']
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_individual_modality_table(self) -> pd.DataFrame:
        """Generate individual modality performance table."""
        rows = []
        for name in self.mod_names_:
            key = f'individual_{name}'
            if key in self.baselines_:
                b = self.baselines_[key]
                rows.append({
                    'modality': name,
                    'ef_1': b['performance'][1]['ef'],
                    'hits_1': b['performance'][1]['hits'],
                    'ef_5': b['performance'][5]['ef'],
                    'hits_5': b['performance'][5]['hits'],
                    'ef_10': b['performance'][10]['ef'],
                    'hits_10': b['performance'][10]['hits'],
                })
        return pd.DataFrame(rows).sort_values('ef_10', ascending=False)

    def save_results(self, output_dir: str):
        """Save all results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_df = pd.DataFrame({
            'column': self.available_cols_,
            'modality': self.mod_names_,
            'weight': self.weights_,
            'weight_pct': self.weights_ * 100
        }).sort_values('weight', ascending=False)
        weights_df.to_csv(output_dir / f'{self.config.output_prefix}_weights.csv', index=False)

        # Save comparison
        comparison_df = self.get_comparison_table()
        comparison_df.to_csv(output_dir / f'{self.config.output_prefix}_comparison.csv', index=False)

        # Save individual modality performance
        indiv_df = self.get_individual_modality_table()
        indiv_df.to_csv(output_dir / f'{self.config.output_prefix}_individual_modalities.csv', index=False)

        # Save performance
        perf_df = pd.DataFrame([
            {'cutoff_pct': k, 'ef': v['ef'], 'hits': v['hits'], 'top_k': v['k']}
            for k, v in self.performance_.items()
        ])
        perf_df.to_csv(output_dir / f'{self.config.output_prefix}_performance.csv', index=False)

        print(f"Results saved to {output_dir}/")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CWRA Final -  Multi-Modality Ensemble')

    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', default='cwra_results', help='Output directory')
    parser.add_argument('--method', '-m', default='fair',
                        choices=['unconstrained', 'fair', 'entropy', 'topk'],
                        help='Optimization method')
    parser.add_argument('--min-weight', type=float, default=0.03,
                        help='Minimum weight per modality (for fair method)')
    parser.add_argument('--max-weight', type=float, default=0.25,
                        help='Maximum weight per modality (for fair method)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    config = CWRAConfig(
        method=args.method,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        de_seed=args.seed
    )

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} compounds")

    cwra = CWRAFinal(config)
    cwra.fit(df)

    df_ranked = cwra.transform(df)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_ranked.to_csv(output_dir / f'{config.output_prefix}_rankings.csv', index=False)
    cwra.save_results(output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
