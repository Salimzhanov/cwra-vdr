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
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
import warnings
import sys
import time

warnings.filterwarnings('ignore')
OBJECTIVE_PRESETS = {
    "default":    {1: 5, 5: 2, 10: 1},
    "sharp":      {0.5: 10, 1: 5, 5: 1},
    "top_heavy":  {0.25: 10, 0.5: 5, 1: 2},
    "balanced":   {1: 1, 5: 1, 10: 1, 20: 1},
}
CUTOFFS = [0.5, 1, 2.5, 5, 10, 20]

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


class TeeStream:
    """Write stream output to multiple destinations (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def project_to_capped_simplex(
    v: np.ndarray,
    lo: float | np.ndarray,
    hi: float | np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> np.ndarray:
    """Project v onto {w | sum(w)=1, lo<=w<=hi} via bisection on lambda."""
    v = np.asarray(v, dtype=float)
    n = v.size
    lo_arr = np.full(n, lo, dtype=float) if np.isscalar(lo) else np.asarray(lo, dtype=float)
    hi_arr = np.full(n, hi, dtype=float) if np.isscalar(hi) else np.asarray(hi, dtype=float)
    if lo_arr.shape != (n,) or hi_arr.shape != (n,):
        raise ValueError("lo/hi must be scalar or length-n arrays.")
    if np.any(lo_arr > hi_arr):
        raise ValueError("lo must be <= hi for all dimensions.")

    sum_lo = float(lo_arr.sum())
    sum_hi = float(hi_arr.sum())
    if sum_lo - 1.0 > tol or 1.0 - sum_hi > tol:
        raise ValueError(
            f"Infeasible bounds: n={n} sum(lo)={sum_lo:.6f} sum(hi)={sum_hi:.6f}."
        )

    if abs(sum_hi - 1.0) <= tol:
        return hi_arr.copy()
    if abs(sum_lo - 1.0) <= tol:
        return lo_arr.copy()

    lam_low = np.min(v - hi_arr)
    lam_high = np.max(v - lo_arr)
    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_low + lam_high)
        w = np.clip(v - lam_mid, lo_arr, hi_arr)
        s = float(w.sum())
        if abs(s - 1.0) <= tol:
            return w
        if s > 1.0:
            lam_low = lam_mid
        else:
            lam_high = lam_mid
    return np.clip(v - lam_mid, lo_arr, hi_arr)


def normalize_weights(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    """Normalize weights to satisfy bounds and sum-to-1."""
    w = np.asarray(weights, dtype=float)
    if min_weight <= 0.0 and max_weight >= 1.0:
        s = w.sum()
        if s == 0:
            raise ValueError("Weights sum to zero; cannot normalize.")
        return w / s
    return project_to_capped_simplex(w, min_weight, max_weight)


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
        'boltz_confidence': ('high', 'Boltz_confidence'),
        'tankbind_affinity': ('low', 'TankBind'),
        'drugban_affinity': ('low', 'DrugBAN'),
        'moltrans_affinity': ('low', 'MolTrans'),
        'unimol_similarity': ('high', 'UniMol_sim'),
    })

    active_sources: List[str] = field(default_factory=lambda: ['initial_370', 'calcitriol'])
    exclude_sources: List[str] = field(default_factory=lambda: ['newRef_137'])

    method: str = 'fair'
    norm_method: str = "minmax"
    objective_preset: str = "default"
    use_bedroc: bool = False
    bedroc_alpha: float = 80.0

    min_weight: float = 0.03
    max_weight: float = 0.4

    entropy_weight: float = 0.5

    top_k: int = 7

    cutoff_weights: Dict[int, float] = field(default_factory=lambda: {1: 5, 5: 2, 10: 1})

    de_maxiter: int = 1000
    de_seed: int = 42
    de_n_seeds: int = 1
    de_workers: int = 1  # DE parallel workers; use -1 for all CPUs

    n_random_trials: int = 100

    max_mw: float = 0.0
    max_rotb: int = 0
    smiles_col: str = "smiles"
    drop_modalities: List[str] = field(default_factory=list)
    auto_prune_threshold: float = 0.0
    strict_cv: bool = False
    fold_honest_unimol: bool = False  # Fold-honest Uni-Mol in CV only
    unimol_embeddings_path: Optional[str] = None  # .npz with smiles + embeddings
    report_extra_metrics: bool = True
    report_bedroc_alpha: float = 20.0

    # --- Train/test split settings (single split only) ---
    train_frac: float = 0.7       # fraction of actives used for training
    split_seed: int = 42          # seed for the active split

    output_prefix: str = 'cwra_cv'

    def __post_init__(self):
        if self.objective_preset in OBJECTIVE_PRESETS:
            self.cutoff_weights = OBJECTIVE_PRESETS[self.objective_preset]
        if self.de_workers == 0 or self.de_workers < -1:
            raise ValueError("de_workers must be -1 (all CPUs) or a positive integer.")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def normalize_modalities(
    df: pd.DataFrame,
    modalities: Dict[str, Tuple[str, str]],
    norm_method: str = "minmax",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Normalize modality columns to [0, 1] range with consistent direction."""
    available_cols = []
    mod_names = []
    norm_data = []

    for col, (direction, name) in modalities.items():
        if col not in df.columns:
            continue

        x = df[col].values.copy().astype(float)
        x = np.nan_to_num(x, nan=np.nanmean(x[~np.isnan(x)]))

        if norm_method == "minmax":
            xmin, xmax = x.min(), x.max()
            if xmax > xmin:
                normalized = (x - xmin) / (xmax - xmin)
            else:
                normalized = np.zeros_like(x)
        elif norm_method == "rank":
            ranked = stats.rankdata(x, method='average') / len(x)
            rmin, rmax = ranked.min(), ranked.max()
            normalized = (ranked - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(x)
        elif norm_method == "robust":
            p1, p99 = np.percentile(x, [1, 99])
            x_clipped = np.clip(x, p1, p99)
            xmin, xmax = x_clipped.min(), x_clipped.max()
            if xmax > xmin:
                normalized = (x_clipped - xmin) / (xmax - xmin)
            else:
                normalized = np.zeros_like(x)
        else:
            raise ValueError(f"Unknown norm_method: {norm_method}")

        if direction == 'low':
            normalized = 1 - normalized

        norm_data.append(normalized)
        available_cols.append(col)
        mod_names.append(name)

    return np.column_stack(norm_data), available_cols, mod_names


def normalize_modalities_cv(
    df: pd.DataFrame,
    modalities: Dict[str, Tuple[str, str]],
    norm_method: str = "minmax",
    exclude_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Normalize modalities with optional exclusion of rows from statistics computation
    (for fold-honest CV).

    Parameters
    ----------
    df : DataFrame with modality columns
    modalities : dict of {col: (direction, name)}
    norm_method : "minmax", "rank", or "robust"
    exclude_mask : boolean array, same length as df. If provided, rows where
        exclude_mask==True are excluded from normalization statistics. All rows
        still receive normalized values. For rank normalization, the empirical
        CDF is fit on the non-excluded rows.

    Returns: (X, available_cols, mod_names)
    """
    available_cols = []
    mod_names = []
    norm_data = []

    for col, (direction, name) in modalities.items():
        if col not in df.columns:
            continue

        x = df[col].values.copy().astype(float)
        x = np.nan_to_num(x, nan=np.nanmean(x[~np.isnan(x)]))

        if norm_method == "rank":
            x_fit = x[~exclude_mask] if exclude_mask is not None else x
            if x_fit.size == 0:
                x_fit = x
            sorted_fit = np.sort(x_fit)
            ranks = np.searchsorted(sorted_fit, x, side="right") / len(sorted_fit)
            normalized = ranks.astype(float)
        elif norm_method == "robust":
            x_fit = x[~exclude_mask] if exclude_mask is not None else x
            p1, p99 = np.percentile(x_fit, [1, 99])
            x_clipped = np.clip(x, p1, p99)
            xmin, xmax = x_clipped.min(), x_clipped.max()
            if xmax > xmin:
                normalized = (x_clipped - xmin) / (xmax - xmin)
            else:
                normalized = np.zeros_like(x)
        else:
            x_fit = x[~exclude_mask] if exclude_mask is not None else x
            xmin, xmax = x_fit.min(), x_fit.max()
            if xmax > xmin:
                normalized = (x - xmin) / (xmax - xmin)
                normalized = np.clip(normalized, 0.0, 1.0)
            else:
                normalized = np.zeros_like(x)

        if direction == 'low':
            normalized = 1 - normalized

        norm_data.append(normalized)
        available_cols.append(col)
        mod_names.append(name)

    return np.column_stack(norm_data), available_cols, mod_names


def _normalize_smiles_key(value) -> str:
    """Normalize SMILES key representation for robust dict lookups."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_unimol_embeddings_npz(path: str, df: pd.DataFrame, smiles_col: str) -> np.ndarray:
    """
    Load Uni-Mol embeddings from .npz and align to dataframe rows by SMILES.

    Expected .npz keys:
      - smiles: array-like of shape (N,)
      - emb:    array-like of shape (N, D)
    """
    emb_path = Path(path)
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Uni-Mol embeddings file not found: {emb_path}. "
            "Provide --unimol-embeddings path to a valid .npz file."
        )
    if smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found in dataframe. "
            f"Available columns include: {list(df.columns)[:12]}..."
        )

    try:
        npz = np.load(emb_path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"Failed to load Uni-Mol embeddings npz: {emb_path}") from exc

    if "smiles" not in npz or "emb" not in npz:
        raise ValueError(
            "Uni-Mol embeddings npz must contain keys 'smiles' and 'emb'. "
            f"Found keys: {list(npz.keys())}"
        )

    emb_smiles = np.asarray(npz["smiles"])
    emb = np.asarray(npz["emb"], dtype=float)
    if emb.ndim != 2:
        raise ValueError(f"Expected 'emb' to have shape (N, D), got {emb.shape}.")
    if emb_smiles.shape[0] != emb.shape[0]:
        raise ValueError(
            f"Mismatched npz sizes: len(smiles)={emb_smiles.shape[0]} vs emb rows={emb.shape[0]}."
        )

    emb_keys = [_normalize_smiles_key(s) for s in emb_smiles.tolist()]
    if len(set(emb_keys)) != len(emb_keys):
        raise ValueError(
            "Duplicate SMILES found in Uni-Mol embeddings npz. "
            "Expected unique SMILES keys for alignment."
        )
    emb_lookup = {s: i for i, s in enumerate(emb_keys)}

    aligned_idx = np.empty(len(df), dtype=int)
    missing = []
    for i, smi in enumerate(df[smiles_col].tolist()):
        if pd.isna(smi):
            missing.append("<NaN>")
            continue
        key = _normalize_smiles_key(smi)
        idx = emb_lookup.get(key)
        if idx is None:
            missing.append(key)
            continue
        aligned_idx[i] = idx

    if missing:
        uniq_missing = list(dict.fromkeys(missing))
        preview = ", ".join(uniq_missing[:5])
        more = "" if len(uniq_missing) <= 5 else f", ... (+{len(uniq_missing)-5} more)"
        raise ValueError(
            f"Uni-Mol embedding alignment failed: {len(missing)} dataframe rows are missing "
            f"from embeddings map ({len(uniq_missing)} unique SMILES). "
            f"Examples: {preview}{more}"
        )

    return emb[aligned_idx]


def compute_fold_honest_unimol_similarity(
    emb: np.ndarray,
    train_active_mask: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Recompute Uni-Mol cosine similarity to centroid built from TRAIN-FOLD actives only.
    """
    train_active_mask = np.asarray(train_active_mask, dtype=bool)
    if emb.shape[0] != train_active_mask.shape[0]:
        raise ValueError(
            "Embedding rows and train_active_mask length must match: "
            f"{emb.shape[0]} vs {train_active_mask.shape[0]}"
        )
    if not np.any(train_active_mask):
        raise ValueError("Train active mask is empty for this fold; cannot build Uni-Mol centroid.")

    centroid = np.mean(emb[train_active_mask], axis=0)
    emb_norm = np.linalg.norm(emb, axis=1)
    cent_norm = float(np.linalg.norm(centroid))
    denom = np.maximum(emb_norm * cent_norm, eps)
    sim = (emb @ centroid) / denom
    return np.clip(sim, -1.0, 1.0)


def apply_druglike_filter(
    df_pool: pd.DataFrame,
    active_mask: np.ndarray,
    config: CWRAConfig,
    verbose: bool = True,
):
    """
    Remove non-active compounds that violate MW/RotB thresholds.
    Actives are NEVER removed.

    Returns: (df_filtered, active_mask_filtered, report_dict)
    """
    if config.max_mw <= 0 and config.max_rotb <= 0:
        return df_pool, active_mask, None

    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rdkit is required for drug-likeness filtering. "
            "Install with: pip install rdkit-pypi"
        ) from exc

    smiles_col = config.smiles_col
    if smiles_col not in df_pool.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found. Available: {list(df_pool.columns)}"
        )

    n_before = len(df_pool)
    keep = np.ones(n_before, dtype=bool)
    reasons = {'mw': 0, 'rotb': 0, 'parse_fail': 0}

    for i in range(n_before):
        if active_mask[i]:
            continue  # never filter actives
        smi = df_pool.iloc[i][smiles_col]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            keep[i] = False
            reasons['parse_fail'] += 1
            continue
        if config.max_mw > 0 and Descriptors.MolWt(mol) > config.max_mw:
            keep[i] = False
            reasons['mw'] += 1
            continue
        if config.max_rotb > 0:
            rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rotb > config.max_rotb:
                keep[i] = False
                reasons['rotb'] += 1

    df_filtered = df_pool[keep].reset_index(drop=True)
    active_mask_filtered = active_mask[keep]
    n_after = len(df_filtered)

    report = {
        'n_before': n_before,
        'n_after': n_after,
        'n_removed': n_before - n_after,
        'removed_mw': reasons['mw'],
        'removed_rotb': reasons['rotb'],
        'removed_parse_fail': reasons['parse_fail'],
        'n_actives': int(active_mask_filtered.sum()),
    }

    if verbose:
        print(
            f"\nDrug-likeness filter: {n_before} -> {n_after} "
            f"(removed {n_before - n_after} non-actives: "
            f"MW>{config.max_mw}: {reasons['mw']}, "
            f"RotB>{config.max_rotb}: {reasons['rotb']}, "
            f"parse fail: {reasons['parse_fail']})"
        )

    return df_filtered, active_mask_filtered, report


def filter_modalities(modalities: Dict[str, Tuple[str, str]], drop_names: List[str]) -> Dict[str, Tuple[str, str]]:
    """Remove modalities by key or display name (case-insensitive)."""
    if not drop_names:
        return modalities
    drop_exact = {d.strip() for d in drop_names if isinstance(d, str) and d.strip()}
    drop_lower = {d.lower() for d in drop_exact}

    filtered = {}
    for key, (direction, display_name) in modalities.items():
        key_l = key.lower()
        name_l = display_name.lower()
        if (
            key in drop_exact
            or display_name in drop_exact
            or key_l in drop_lower
            or name_l in drop_lower
        ):
            continue
        filtered[key] = (direction, display_name)
    return filtered


def compute_ef(scores: np.ndarray, active_mask: np.ndarray, cutoff_pct: float) -> Tuple[float, int, int]:
    """Compute enrichment factor at given cutoff percentage."""
    N = len(scores)
    A = active_mask.sum()
    k = max(1, int(N * cutoff_pct / 100))

    top_k_idx = np.argsort(scores)[-k:]
    hits = active_mask[top_k_idx].sum()
    ef = (hits * N) / (k * A) if A > 0 else 0

    return ef, int(hits), k


def evaluate_weights(
    weights: np.ndarray,
    X: np.ndarray,
    active_mask: np.ndarray,
    cutoffs: List[float] = None,
    extra_metrics: bool = False,
    bedroc_alpha: float = 20.0,
) -> Dict:
    """Evaluate weights and return performance dict."""
    if cutoffs is None:
        cutoffs = CUTOFFS
    scores = X @ weights
    results = {}
    for cutoff in cutoffs:
        ef, hits, k = compute_ef(scores, active_mask, cutoff)
        results[cutoff] = {'ef': ef, 'hits': hits, 'k': k}
    if extra_metrics:
        results['extra'] = compute_extra_metrics(
            scores, active_mask, bedroc_alpha=bedroc_alpha
        )
    return results


def compute_bedroc(scores: np.ndarray, active_mask: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly 2007). Requires RDKit."""
    try:
        from rdkit.ML.Scoring.Scoring import CalcBEDROC  # type: ignore
    except ImportError:
        warnings.warn("RDKit ML module not available; BEDROC = NaN")
        return float('nan')
    n_actives = int(active_mask.sum())
    if n_actives == 0 or n_actives == len(scores):
        return 0.0
    order = np.argsort(-scores)
    scored = [(float(scores[i]), int(active_mask[i])) for i in order]
    try:
        return float(CalcBEDROC(scored, 1, alpha))
    except Exception:
        warnings.warn("RDKit BEDROC computation failed; BEDROC = NaN")
        return float('nan')


def compute_extra_metrics(scores: np.ndarray, active_mask: np.ndarray, bedroc_alpha: float = 20.0) -> Dict[str, float]:
    """Compute AUROC, AUPRC, and BEDROC for a score vector."""
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore
    except ImportError:
        warnings.warn("sklearn not available; AUROC/AUPRC = NaN")
        return {'auroc': float('nan'), 'auprc': float('nan'), 'bedroc': float('nan')}

    n_actives = int(active_mask.sum())
    n = len(scores)
    if n_actives == 0 or n_actives == n:
        return {'auroc': 0.0, 'auprc': 0.0, 'bedroc': 0.0}

    auroc = roc_auc_score(active_mask, scores)
    auprc = average_precision_score(active_mask, scores)
    bedroc = compute_bedroc(scores, active_mask, bedroc_alpha)

    return {'auroc': float(auroc), 'auprc': float(auprc), 'bedroc': float(bedroc)}


# =============================================================================
# OBJECTIVE FUNCTIONS (unchanged from original)
# =============================================================================

def objective_unconstrained(weights, X, active_mask, cutoff_weights, min_weight, max_weight):
    w_norm = normalize_weights(weights, min_weight, max_weight)
    scores = X @ w_norm
    total = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())
    return -total


def objective_fair(weights, X, active_mask, cutoff_weights, min_weight, max_weight):
    w_norm = normalize_weights(weights, min_weight, max_weight)
    scores = X @ w_norm
    total = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())
    return -total


def objective_entropy(weights, X, active_mask, cutoff_weights, entropy_weight, n_mod, min_weight, max_weight):
    w_norm = normalize_weights(weights, min_weight, max_weight)
    scores = X @ w_norm
    total_ef = sum(w * compute_ef(scores, active_mask, c)[0] for c, w in cutoff_weights.items())
    w_safe = np.clip(w_norm, 1e-10, 1)
    entropy = -np.sum(w_safe * np.log(w_safe)) / np.log(n_mod)
    return -(total_ef + entropy_weight * total_ef * entropy)


def objective_unconstrained_bedroc(weights, X, active_mask, alpha, min_weight, max_weight):
    w_norm = normalize_weights(weights, min_weight, max_weight)
    scores = X @ w_norm
    return -compute_bedroc(scores, active_mask, alpha)


def objective_fair_bedroc(weights, X, active_mask, alpha, min_weight, max_weight):
    w_norm = normalize_weights(weights, min_weight, max_weight)
    scores = X @ w_norm
    return -compute_bedroc(scores, active_mask, alpha)


def objective_entropy_bedroc(weights, X, active_mask, alpha, entropy_weight, n_mod, min_weight, max_weight):
    w_norm = normalize_weights(weights, min_weight, max_weight)
    scores = X @ w_norm
    bed = compute_bedroc(scores, active_mask, alpha)
    w_safe = np.clip(w_norm, 1e-10, 1)
    entropy = -np.sum(w_safe * np.log(w_safe)) / np.log(n_mod)
    return -(bed + entropy_weight * bed * entropy)


def _de_call_kwargs(config, seed: int) -> Dict[str, object]:
    """Shared SciPy DE kwargs with optional parallel workers."""
    kwargs: Dict[str, object] = {
        "maxiter": config.de_maxiter,
        "seed": seed,
        "workers": config.de_workers,
        "polish": True,
    }
    if config.de_workers != 1:
        # Required for parallel evaluation in SciPy differential_evolution.
        kwargs["updating"] = "deferred"
    return kwargs


# =============================================================================
# OPTIMIZATION METHODS (unchanged from original)
# =============================================================================

def optimize_unconstrained(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(0.01, 1.0)] * n_mod
    best_weights = None
    best_obj = float('inf')
    for s in range(config.de_n_seeds):
        seed = config.de_seed + s
        result = differential_evolution(
            objective_unconstrained, bounds,
            args=(X, active_mask, config.cutoff_weights, config.min_weight, config.max_weight),
            **_de_call_kwargs(config, seed),
        )
        w = normalize_weights(result.x, config.min_weight, config.max_weight)
        obj = result.fun
        if obj < best_obj:
            best_obj = obj
            best_weights = w
    return best_weights


def optimize_fair(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(config.min_weight, config.max_weight)] * n_mod
    best_weights = None
    best_obj = float('inf')
    for s in range(config.de_n_seeds):
        seed = config.de_seed + s
        result = differential_evolution(
            objective_fair, bounds,
            args=(X, active_mask, config.cutoff_weights, config.min_weight, config.max_weight),
            **_de_call_kwargs(config, seed),
        )
        w = normalize_weights(result.x, config.min_weight, config.max_weight)
        obj = result.fun
        if obj < best_obj:
            best_obj = obj
            best_weights = w
    return best_weights


def optimize_entropy(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(0.01, 1.0)] * n_mod
    best_weights = None
    best_obj = float('inf')
    for s in range(config.de_n_seeds):
        seed = config.de_seed + s
        result = differential_evolution(
            objective_entropy, bounds,
            args=(X, active_mask, config.cutoff_weights, config.entropy_weight, n_mod, config.min_weight, config.max_weight),
            **_de_call_kwargs(config, seed),
        )
        w = normalize_weights(result.x, config.min_weight, config.max_weight)
        obj = result.fun
        if obj < best_obj:
            best_obj = obj
            best_weights = w
    return best_weights


def optimize_unconstrained_bedroc(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(0.01, 1.0)] * n_mod
    best_weights = None
    best_obj = float('inf')
    for s in range(config.de_n_seeds):
        seed = config.de_seed + s
        result = differential_evolution(
            objective_unconstrained_bedroc, bounds,
            args=(X, active_mask, config.bedroc_alpha, config.min_weight, config.max_weight),
            **_de_call_kwargs(config, seed),
        )
        w = normalize_weights(result.x, config.min_weight, config.max_weight)
        obj = result.fun
        if obj < best_obj:
            best_obj = obj
            best_weights = w
    return best_weights


def optimize_fair_bedroc(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(config.min_weight, config.max_weight)] * n_mod
    best_weights = None
    best_obj = float('inf')
    for s in range(config.de_n_seeds):
        seed = config.de_seed + s
        result = differential_evolution(
            objective_fair_bedroc, bounds,
            args=(X, active_mask, config.bedroc_alpha, config.min_weight, config.max_weight),
            **_de_call_kwargs(config, seed),
        )
        w = normalize_weights(result.x, config.min_weight, config.max_weight)
        obj = result.fun
        if obj < best_obj:
            best_obj = obj
            best_weights = w
    return best_weights


def optimize_entropy_bedroc(X, active_mask, config):
    n_mod = X.shape[1]
    bounds = [(0.01, 1.0)] * n_mod
    best_weights = None
    best_obj = float('inf')
    for s in range(config.de_n_seeds):
        seed = config.de_seed + s
        result = differential_evolution(
            objective_entropy_bedroc, bounds,
            args=(X, active_mask, config.bedroc_alpha, config.entropy_weight, n_mod, config.min_weight, config.max_weight),
            **_de_call_kwargs(config, seed),
        )
        w = normalize_weights(result.x, config.min_weight, config.max_weight)
        obj = result.fun
        if obj < best_obj:
            best_obj = obj
            best_weights = w
    return best_weights


OPTIMIZERS = {
    'unconstrained': optimize_unconstrained,
    'fair': optimize_fair,
    'entropy': optimize_entropy,
    'unconstrained_bedroc': optimize_unconstrained_bedroc,
    'fair_bedroc': optimize_fair_bedroc,
    'entropy_bedroc': optimize_entropy_bedroc,
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

    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")

    active_indices = np.where(active_mask)[0]
    n_actives = len(active_indices)
    if n_actives < 2:
        raise ValueError("Need at least 2 actives to create train/test split.")
    n_train = int(n_actives * train_frac)
    if n_train <= 0 or n_train >= n_actives:
        raise ValueError(
            f"train_frac={train_frac} yields invalid split with {n_actives} actives "
            f"(n_train={n_train}, n_test={n_actives - n_train})."
        )

    shuffled = rng.permutation(active_indices)
    train_indices = shuffled[:n_train]
    test_indices = shuffled[n_train:]

    train_active_mask = np.zeros(len(active_mask), dtype=bool)
    test_active_mask = np.zeros(len(active_mask), dtype=bool)

    train_active_mask[train_indices] = True
    test_active_mask[test_indices] = True

    return train_active_mask, test_active_mask


def make_cv_folds(
    active_mask: np.ndarray,
    n_folds: int,
    seed: int,
    train_frac: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create repeated random train/test splits over the active compounds.

    Returns a list of (train_active_mask, test_active_mask) pairs.
    All non-active compounds have False in both masks.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")

    folds = []
    for i in range(n_folds):
        train_mask, test_mask = split_actives(
            active_mask=active_mask,
            train_frac=train_frac,
            seed=seed + i,
        )
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
    weights_equal = normalize_weights(np.ones(n_mod), config.min_weight, config.max_weight)
    baselines['equal'] = {
        'weights': weights_equal,
        'performance': evaluate_weights(weights_equal, X, active_mask),
        'name': 'Equal Weights'
    }

    # Random weights (averaged)
    np.random.seed(config.de_seed)
    all_cutoffs = CUTOFFS
    random_perfs = {c: [] for c in all_cutoffs}
    for _ in range(config.n_random_trials):
        w_rand = np.random.dirichlet(np.ones(n_mod))
        perf = evaluate_weights(w_rand, X, active_mask)
        for c in all_cutoffs:
            if c in perf:
                random_perfs[c].append(perf[c]['ef'])

    N = len(active_mask)
    A = active_mask.sum()
    perf_random = {}
    for c in all_cutoffs:
        if random_perfs[c]:
            mean_ef = np.mean(random_perfs[c])
            k = max(1, int(N * c / 100))
            est_hits = int(mean_ef * k * A / N)
            perf_random[c] = {'ef': mean_ef, 'hits': est_hits, 'k': k}
        else:
            perf_random[c] = baselines['equal']['performance'].get(
                c, {'ef': 1.0, 'hits': int(A * c / 100), 'k': max(1, int(N * c / 100))}
            )

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
        self.effective_modalities_ = None
        # Store results for train, test, and full (reference)
        self.train_performance_ = None
        self.test_performance_ = None
        self.full_performance_ = None
        self.baselines_train_ = None
        self.baselines_test_ = None
        self.baselines_full_ = None
        self.all_methods_ = None
        self.split_info_ = None
        self.filter_report_ = None
        self.prune_report_ = None

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
        df_pool, full_active_mask, filter_report = apply_druglike_filter(
            df_pool, full_active_mask, self.config, verbose=verbose
        )
        self.filter_report_ = filter_report
        effective_modalities = filter_modalities(
            self.config.modalities, self.config.drop_modalities
        )
        if self.config.auto_prune_threshold > 0:
            X_pre, avail_pre, names_pre = normalize_modalities(
                df_pool, effective_modalities, norm_method=self.config.norm_method
            )
            base_method = self.config.method if self.config.method in OPTIMIZERS else 'fair'
            opt_name = f"{base_method}_bedroc" if self.config.use_bedroc else base_method
            if opt_name not in OPTIMIZERS:
                opt_name = "fair_bedroc" if self.config.use_bedroc else "fair"
            w_pre = OPTIMIZERS[opt_name](X_pre, full_active_mask, self.config)
            prune_report = []
            to_drop = []
            for name, w in zip(names_pre, w_pre):
                pruned = w < self.config.auto_prune_threshold
                prune_report.append({
                    'modality': name,
                    'preliminary_weight': float(w),
                    'pruned': bool(pruned),
                })
                if pruned:
                    to_drop.append(name)
            if to_drop:
                print(f"Auto-prune: dropping {to_drop} (weights: {[float(w_pre[list(names_pre).index(n)]) for n in to_drop]})")
                self.config.drop_modalities = list(set(self.config.drop_modalities + to_drop))
                effective_modalities = filter_modalities(
                    self.config.modalities, self.config.drop_modalities
                )
            else:
                print("Auto-prune: no modalities dropped.")
            self.prune_report_ = prune_report

        X, self.available_cols_, self.mod_names_ = normalize_modalities(
            df_pool, effective_modalities, norm_method=self.config.norm_method
        )
        self.effective_modalities_ = effective_modalities

        N, n_mod = X.shape
        A_full = full_active_mask.sum()

        if verbose:
            print(f"\nDataset: {N:,} compounds, {A_full} actives ({100*A_full/N:.2f}%)")
            print(f"Modalities: {n_mod}")
            print(f"Method: {self.config.method}")
            print(f"Train fraction: {self.config.train_frac}")
            print(f"Split seed: {self.config.split_seed}")
            print(f"DE workers: {self.config.de_workers}")

        # ----- Split actives -----
        train_active_mask, test_active_mask = split_actives(
            full_active_mask, self.config.train_frac, self.config.split_seed
        )

        if self.config.strict_cv:
            X, _, _ = normalize_modalities_cv(
                df_pool,
                effective_modalities,
                self.config.norm_method,
                exclude_mask=test_active_mask,
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

        method_names = ['unconstrained', 'fair', 'entropy']
        if self.config.use_bedroc:
            method_names = ['unconstrained_bedroc', 'fair_bedroc', 'entropy_bedroc']

        for method_name in method_names:
            if verbose:
                print(f"  Running {method_name} optimization...")

            optimizer = OPTIMIZERS[method_name]
            # KEY: optimize using train_active_mask, not full_active_mask
            weights = optimizer(X, train_active_mask, self.config)

            # Evaluate on train, test, and full
            perf_train = evaluate_weights(
                weights,
                X,
                train_active_mask,
                extra_metrics=self.config.report_extra_metrics,
                bedroc_alpha=self.config.report_bedroc_alpha,
            )
            perf_test = evaluate_weights(
                weights,
                X,
                test_active_mask,
                extra_metrics=self.config.report_extra_metrics,
                bedroc_alpha=self.config.report_bedroc_alpha,
            )
            perf_full = evaluate_weights(
                weights,
                X,
                full_active_mask,
                extra_metrics=self.config.report_extra_metrics,
                bedroc_alpha=self.config.report_bedroc_alpha,
            )

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
        if self.config.use_bedroc:
            chosen = f"{chosen}_bedroc"
        if chosen not in self.all_methods_:
            chosen = "fair_bedroc" if self.config.use_bedroc else "fair"

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

            for c in CUTOFFS:
                tr = self.train_performance_[c]
                te = self.test_performance_[c]
                fu = self.full_performance_[c]
                c_str = f"{c:g}"
                print(f"  @{c_str:>4}%   "
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
            method_keys = ['fair', 'unconstrained', 'entropy']
            if self.config.use_bedroc:
                method_labels = {
                    'fair_bedroc': 'DE Fair (BEDROC)',
                    'unconstrained_bedroc': 'DE Unconstrained (BEDROC)',
                    'entropy_bedroc': 'DE Entropy (BEDROC)',
                }
                method_keys = ['fair_bedroc', 'unconstrained_bedroc', 'entropy_bedroc']
            for key in method_keys:
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

        modalities = self.effective_modalities_ or self.config.modalities
        X, _, _ = normalize_modalities(
            df_pool, modalities, norm_method=self.config.norm_method
        )
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
            for c in CUTOFFS:
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
        if self.config.use_bedroc:
            method_labels = {
                'unconstrained_bedroc': 'DE Unconstrained (BEDROC)',
                'fair_bedroc': 'DE Fair (BEDROC)',
                'entropy_bedroc': 'DE Entropy (BEDROC)',
            }

        for key, label in method_labels.items():
            m = self.all_methods_[key]
            row = {'method': label}
            for c in CUTOFFS:
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
                for c in CUTOFFS:
                    row[f'train_ef_{c}'] = bt['performance'][c]['ef']
                    row[f'train_hits_{c}'] = bt['performance'][c]['hits']
                    row[f'test_ef_{c}'] = be['performance'][c]['ef']
                    row[f'test_hits_{c}'] = be['performance'][c]['hits']
                    row[f'full_ef_{c}'] = bf['performance'][c]['ef']
                    row[f'full_hits_{c}'] = bf['performance'][c]['hits']
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
        for c in CUTOFFS:
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
        if self.filter_report_ is not None:
            pd.DataFrame([self.filter_report_]).to_csv(
                output_dir / f'{prefix}_filter_report.csv', index=False
            )
        if self.prune_report_ is not None:
            pd.DataFrame(self.prune_report_).to_csv(
                output_dir / f'{prefix}_pruning_report.csv', index=False
            )

        print(f"\nResults saved to {output_dir}/")


# =============================================================================
# K-FOLD CROSS-VALIDATION
# =============================================================================

def run_cross_validation(df: pd.DataFrame, config: CWRAConfig, n_folds: int,
                         output_dir: Optional[Path] = None, verbose: bool = True):
    """Run repeated active train/test splits and save per-split + summary outputs."""
    t0 = time.time()
    def _fmt_pm(mean: float, std: float, digits: int = 2) -> str:
        return f"{mean:.{digits}f} +/- {std:.{digits}f}"

    def _normalized_entropy(w: np.ndarray) -> float:
        w_safe = np.clip(w, 1e-10, 1)
        return float(-np.sum(w_safe * np.log(w_safe)) / np.log(len(w)))

    def _mean_rank(scores: np.ndarray, active_mask: np.ndarray) -> float:
        """Mean rank position of actives (1 = best rank)."""
        n_act = int(np.asarray(active_mask, dtype=bool).sum())
        if n_act == 0:
            return float("nan")
        ranks = stats.rankdata(-np.asarray(scores, dtype=float), method='average')
        return float(np.mean(ranks[np.asarray(active_mask, dtype=bool)]))

    if verbose:
        print("=" * 80)
        print(f"CWRA REPEATED ACTIVE-SPLIT CV (n={n_folds})")
        print("=" * 80)

    df_pool = df[~df['source'].isin(config.exclude_sources)].copy()
    full_active_mask = df_pool['source'].isin(config.active_sources).values
    df_pool, full_active_mask, filter_report = apply_druglike_filter(
        df_pool, full_active_mask, config, verbose=verbose
    )

    effective_modalities = filter_modalities(
        config.modalities, config.drop_modalities
    )
    prune_report = None
    if config.auto_prune_threshold > 0:
        X_pre, avail_pre, names_pre = normalize_modalities(
            df_pool, effective_modalities, norm_method=config.norm_method
        )
        base_method = config.method if config.method in OPTIMIZERS else 'fair'
        opt_name = f"{base_method}_bedroc" if config.use_bedroc else base_method
        if opt_name not in OPTIMIZERS:
            opt_name = "fair_bedroc" if config.use_bedroc else "fair"
        w_pre = OPTIMIZERS[opt_name](X_pre, full_active_mask, config)
        prune_report = []
        to_drop = []
        for name, w in zip(names_pre, w_pre):
            pruned = w < config.auto_prune_threshold
            prune_report.append({
                'modality': name,
                'preliminary_weight': float(w),
                'pruned': bool(pruned),
            })
            if pruned:
                to_drop.append(name)
        if to_drop:
            print(f"Auto-prune: dropping {to_drop} (weights: {[float(w_pre[list(names_pre).index(n)]) for n in to_drop]})")
            config.drop_modalities = list(set(config.drop_modalities + to_drop))
            effective_modalities = filter_modalities(
                config.modalities, config.drop_modalities
            )
        else:
            print("Auto-prune: no modalities dropped.")

    if not config.strict_cv:
        X_global, available_cols, mod_names = normalize_modalities(
            df_pool, effective_modalities, norm_method=config.norm_method
        )
        N, n_mod = X_global.shape
    else:
        X_tmp, available_cols, mod_names = normalize_modalities(
            df_pool, effective_modalities, norm_method=config.norm_method
        )
        X_global = None
        N, n_mod = X_tmp.shape
    A_full = int(full_active_mask.sum())

    if verbose:
        print(f"\nDataset: {N:,} compounds, {A_full} actives ({100*A_full/N:.2f}%)")
        print(f"Modalities: {n_mod}")
        print(f"Method: {config.method}")
        print(f"CV splits: {n_folds}")
        print(f"Train fraction (actives): {config.train_frac:.3f}")
        print(f"CV seed: {config.split_seed}")
        print(f"DE workers: {config.de_workers}")

    folds = make_cv_folds(
        full_active_mask,
        n_folds,
        config.split_seed,
        config.train_frac,
    )

    perf_rows = []
    weight_rows = []
    baseline_rows = []
    fold_info_rows = []
    indiv_rows = []
    method_rows = []
    extra_rows = []
    entropy_rows = []
    significant_rows = []
    unimol_fold_rows = []

    # Fold-honest Uni-Mol setup (CV-only).
    use_fold_honest_unimol = False
    unimol_emb_aligned = None
    unimol_feature_key = 'unimol_similarity'
    unimol_display_name = config.modalities.get(unimol_feature_key, ('high', 'UniMol_sim'))[1]
    if config.fold_honest_unimol:
        if unimol_feature_key not in effective_modalities:
            if verbose:
                print(
                    "Fold-honest Uni-Mol requested but modality 'unimol_similarity' "
                    "is not enabled (possibly dropped); skipping fold-honest recomputation."
                )
        else:
            if verbose:
                print(
                    "Fold-honest Uni-Mol similarity enabled: centroid computed from "
                    "training actives only per fold."
                )
            if not config.unimol_embeddings_path:
                raise ValueError(
                    "Fold-honest Uni-Mol is enabled but unimol_embeddings_path is not set. "
                    "Provide --unimol-embeddings <path.npz>."
                )
            unimol_emb_aligned = load_unimol_embeddings_npz(
                config.unimol_embeddings_path, df_pool, config.smiles_col
            )
            use_fold_honest_unimol = True

    base_method = config.method if config.method in ['unconstrained', 'fair', 'entropy'] else 'fair'
    chosen = f"{base_method}_bedroc" if config.use_bedroc else base_method
    if chosen not in OPTIMIZERS:
        chosen = "fair_bedroc" if config.use_bedroc else "fair"

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

        if use_fold_honest_unimol:
            # Fold-local dataframe copy to avoid mutating global pool across folds.
            df_fold = df_pool.copy()
            unimol_sim = compute_fold_honest_unimol_similarity(
                unimol_emb_aligned,
                train_active_mask,
            )

            # Requested in spec: derive the used Uni-Mol name from config.modalities.
            unimol_col = config.modalities[unimol_feature_key][1]

            # Preserve existing global column values for debugging, when present.
            for col in [unimol_col, unimol_feature_key]:
                if col in df_fold.columns:
                    df_fold[f"{col}_global"] = df_fold[col]

            # Overwrite fold-local Uni-Mol similarity for this fold only.
            df_fold[unimol_col] = unimol_sim
            df_fold[unimol_feature_key] = unimol_sim

            if config.strict_cv:
                X, _, _ = normalize_modalities_cv(
                    df_fold,
                    effective_modalities,
                    config.norm_method,
                    exclude_mask=test_active_mask,
                )
            else:
                X, _, _ = normalize_modalities(
                    df_fold,
                    effective_modalities,
                    norm_method=config.norm_method,
                )
        else:
            if config.strict_cv:
                X, _, _ = normalize_modalities_cv(
                    df_pool,
                    effective_modalities,
                    config.norm_method,
                    exclude_mask=test_active_mask,
                )
            else:
                X = X_global

        all_methods = {}
        method_names = ['unconstrained', 'fair', 'entropy']
        if config.use_bedroc:
            method_names = ['unconstrained_bedroc', 'fair_bedroc', 'entropy_bedroc']
        for method_name in method_names:
            if verbose:
                print(f"  Running {method_name} optimization...")
            optimizer = OPTIMIZERS[method_name]
            weights = optimizer(X, train_active_mask, config)
            scores = X @ weights

            perf_train = evaluate_weights(weights, X, train_active_mask)
            perf_test = evaluate_weights(weights, X, test_active_mask)
            perf_full = evaluate_weights(weights, X, full_active_mask)
            mean_rank_test = _mean_rank(scores, test_active_mask)

            all_methods[method_name] = {
                'weights': weights,
                'perf_train': perf_train,
                'perf_test': perf_test,
                'perf_full': perf_full,
                'mean_rank_test': mean_rank_test,
            }

            method_row = {
                'fold': fold_idx,
                'method': method_name,
                'test_mean_rank': mean_rank_test,
            }
            for c in CUTOFFS:
                method_row[f'test_ef_{c}'] = perf_test[c]['ef']
                method_row[f'test_hits_{c}'] = perf_test[c]['hits']
            method_rows.append(method_row)

        chosen_data = all_methods[chosen]

        if config.report_extra_metrics:
            scores = X @ chosen_data['weights']
            extra_train = compute_extra_metrics(
                scores, train_active_mask, config.report_bedroc_alpha
            )
            extra_test = compute_extra_metrics(
                scores, test_active_mask, config.report_bedroc_alpha
            )
            extra_full = compute_extra_metrics(
                scores, full_active_mask, config.report_bedroc_alpha
            )
            extra_rows.append({
                'fold': fold_idx,
                'method': chosen,
                'train_auroc': extra_train['auroc'],
                'train_auprc': extra_train['auprc'],
                'train_bedroc': extra_train['bedroc'],
                'test_auroc': extra_test['auroc'],
                'test_auprc': extra_test['auprc'],
                'test_bedroc': extra_test['bedroc'],
                'full_auroc': extra_full['auroc'],
                'full_auprc': extra_full['auprc'],
                'full_bedroc': extra_full['bedroc'],
            })

        for c in CUTOFFS:
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

        if use_fold_honest_unimol:
            unimol_weight = float('nan')
            if unimol_display_name in mod_names:
                unimol_weight = float(weights[mod_names.index(unimol_display_name)])
            unimol_fold_rows.append({
                'fold': fold_idx,
                'method': chosen,
                'unimol_weight': unimol_weight,
                'unimol_sim_train_mean': float(np.mean(unimol_sim[train_active_mask])),
                'unimol_sim_test_mean': float(np.mean(unimol_sim[test_active_mask])) if A_test > 0 else float('nan'),
                'unimol_sim_full_mean': float(np.mean(unimol_sim)),
            })

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
            for c in CUTOFFS:
                row[f'train_ef_{c}'] = bt['performance'][c]['ef']
                row[f'train_hits_{c}'] = bt['performance'][c]['hits']
                row[f'test_ef_{c}'] = be['performance'][c]['ef']
                row[f'test_hits_{c}'] = be['performance'][c]['hits']
                row[f'full_ef_{c}'] = bf['performance'][c]['ef']
                row[f'full_hits_{c}'] = bf['performance'][c]['hits']
            if config.report_extra_metrics:
                w_base = be['weights']
                if w_base is not None:
                    scores = X @ w_base
                    extra = compute_extra_metrics(
                        scores, test_active_mask, config.report_bedroc_alpha
                    )
                    row['test_auroc'] = extra['auroc']
                    row['test_auprc'] = extra['auprc']
                    row['test_bedroc'] = extra['bedroc']
                else:
                    row['test_auroc'] = float('nan')
                    row['test_auprc'] = float('nan')
                    row['test_bedroc'] = float('nan')
            baseline_rows.append(row)

        for name in mod_names:
            key = f'individual_{name}'
            if key not in baselines_test:
                continue
            be = baselines_test[key]
            row = {'fold': fold_idx, 'modality': name}
            for c in CUTOFFS:
                row[f'test_ef_{c}'] = be['performance'][c]['ef']
                row[f'test_hits_{c}'] = be['performance'][c]['hits']
            if config.report_extra_metrics:
                w_mod = be['weights']
                if w_mod is not None:
                    scores = X @ w_mod
                    extra = compute_extra_metrics(
                        scores, test_active_mask, config.report_bedroc_alpha
                    )
                    row['test_auroc'] = extra['auroc']
                    row['test_auprc'] = extra['auprc']
                    row['test_bedroc'] = extra['bedroc']
                else:
                    row['test_auroc'] = float('nan')
                    row['test_auprc'] = float('nan')
                    row['test_bedroc'] = float('nan')
            row['test_mean_rank'] = _mean_rank(X @ be['weights'], test_active_mask)
            indiv_rows.append(row)

    perf_df = pd.DataFrame(perf_rows)
    weights_df = pd.DataFrame(weight_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    fold_info_df = pd.DataFrame(fold_info_rows)
    indiv_df = pd.DataFrame(indiv_rows)
    method_df = pd.DataFrame(method_rows)
    extra_df = pd.DataFrame(extra_rows)

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

    # Aggregate fold-level weights into a single reusable mean-weights table.
    mean_weights_df = (
        weights_df.groupby(['column', 'modality'], as_index=False)
        .agg(
            weight=('weight', 'mean'),
            weight_std=('weight', 'std'),
            weight_pct=('weight_pct', 'mean'),
        )
        .sort_values('weight', ascending=False)
    )
    mean_rank_indiv_df = (
        indiv_df.groupby('modality', as_index=False)
        .agg(
            test_mean_rank_mean=('test_mean_rank', 'mean'),
            test_mean_rank_std=('test_mean_rank', 'std'),
        )
    )
    chosen_mean_rank_df = (
        method_df[method_df['method'] == chosen]
        .groupby('method', as_index=False)
        .agg(
            test_mean_rank_mean=('test_mean_rank', 'mean'),
            test_mean_rank_std=('test_mean_rank', 'std'),
        )
    )
    cwra_mean_rank_df = pd.DataFrame(
        {
            'method': ['CWRA'],
            'test_mean_rank_mean': [float(chosen_mean_rank_df['test_mean_rank_mean'].iloc[0]) if not chosen_mean_rank_df.empty else float('nan')],
            'test_mean_rank_std': [float(chosen_mean_rank_df['test_mean_rank_std'].iloc[0]) if not chosen_mean_rank_df.empty else float('nan')],
        }
    )
    mean_rank_summary_df = pd.concat(
        [
            mean_rank_indiv_df.rename(columns={'modality': 'method'}),
            cwra_mean_rank_df,
        ],
        ignore_index=True,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = config.output_prefix
        cv_prefix = prefix if prefix.endswith('_cv') else f'{prefix}_cv'

        perf_df.to_csv(output_dir / f'{cv_prefix}_folds_performance.csv', index=False)
        weights_df.to_csv(output_dir / f'{cv_prefix}_folds_weights.csv', index=False)
        mean_weights_df.to_csv(output_dir / f'{cv_prefix}_mean_weights.csv', index=False)
        baseline_df.to_csv(output_dir / f'{cv_prefix}_folds_baselines.csv', index=False)
        fold_info_df.to_csv(output_dir / f'{cv_prefix}_folds_info.csv', index=False)
        summary_df.to_csv(output_dir / f'{cv_prefix}_summary.csv', index=False)
        indiv_df.to_csv(output_dir / f'{cv_prefix}_folds_individual.csv', index=False)
        method_df.to_csv(output_dir / f'{cv_prefix}_folds_methods.csv', index=False)
        mean_rank_summary_df.to_csv(output_dir / f'{cv_prefix}_mean_rank_summary.csv', index=False)
        if not extra_df.empty:
            extra_df.to_csv(output_dir / f'{cv_prefix}_folds_extra_metrics.csv', index=False)
        if use_fold_honest_unimol and unimol_fold_rows:
            pd.DataFrame(unimol_fold_rows).to_csv(
                output_dir / f'{cv_prefix}_folds_unimol_fold_honest.csv', index=False
            )
        if filter_report is not None:
            pd.DataFrame([filter_report]).to_csv(
                output_dir / f'{prefix}_filter_report.csv', index=False
            )
        if prune_report is not None:
            pd.DataFrame(prune_report).to_csv(
                output_dir / f'{prefix}_pruning_report.csv', index=False
            )

        # Build and save the paper-ready table
        paper_df = build_paper_table(
            indiv_df, baseline_df, method_df, chosen, extra_df=extra_df
        )
        paper_df.to_csv(output_dir / f'{prefix}_paper_table.csv', index=False)

        if verbose:
            print(f"\nResults saved to {output_dir}/")
            print(f"Paper table: {output_dir / f'{prefix}_paper_table.csv'}")
            print(f"Mean weights: {output_dir / f'{cv_prefix}_mean_weights.csv'}")
            print(f"Mean rank summary: {output_dir / f'{cv_prefix}_mean_rank_summary.csv'}")

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
            c = float(row['cutoff_pct'])
            c_str = f"{c:g}"
            tr = _fmt_pm(row['train_ef_mean'], row['train_ef_std'])
            te = _fmt_pm(row['test_ef_mean'], row['test_ef_std'])
            fu = _fmt_pm(row['full_ef_mean'], row['full_ef_std'])
            tr_hits = f"{row['train_hits_mean']:.1f}"
            te_hits = f"{row['test_hits_mean']:.1f}"
            fu_hits = f"{row['full_hits_mean']:.1f}"
            print(
                f"  @{c_str:>4}%   "
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
        eq_ef1 = float(eq_row['test_ef_1_mean'].iloc[0]) if not eq_row.empty else 0.0
        # Use per-cutoff best individual definition (for EF@1 here) for consistency.
        bi_ef1 = float("nan")
        if (not indiv_df.empty) and ("test_ef_1" in indiv_df.columns):
            bi_ef1 = float(indiv_df.groupby('modality')['test_ef_1'].mean().max())
        if np.isnan(bi_ef1):
            bi_row = baseline_summary[baseline_summary['baseline'].str.startswith('Best Individual')]
            bi_ef1 = float(bi_row['test_ef_1_mean'].iloc[0]) if not bi_row.empty else 0.0
        te_ef1 = float(summary_df.loc[summary_df['cutoff_pct'] == 1, 'test_ef_mean'].iloc[0])

        method_ef1 = (
            method_df.groupby('method', as_index=False)
            .agg(test_ef_1_mean=('test_ef_1', 'mean'))
        )
        uncon_key = 'unconstrained_bedroc' if config.use_bedroc else 'unconstrained'
        entropy_key = 'entropy_bedroc' if config.use_bedroc else 'entropy'

        if eq_ef1 > 0:
            print(f"  CWRA vs equal-weight: {100*(te_ef1/eq_ef1 - 1):+.1f}%")
        if bi_ef1 > 0:
            print(f"  CWRA vs best-indiv:   {100*(te_ef1/bi_ef1 - 1):+.1f}%")
            for method_key, method_label in [
                (uncon_key, 'DE Unconstrained'),
                (entropy_key, 'DE Entropy'),
            ]:
                mrow = method_ef1[method_ef1['method'] == method_key]
                if mrow.empty:
                    continue
                m_ef1 = float(mrow['test_ef_1_mean'].iloc[0])
                print(f"  {method_label} vs best-indiv: {100*(m_ef1/bi_ef1 - 1):+.1f}%")
            all_ef_parts = []
            for c in CUTOFFS:
                c_str = f"{c:g}"
                col = f'test_ef_{c}'
                row_c = summary_df.loc[summary_df['cutoff_pct'] == c]
                if row_c.empty:
                    continue
                # Compare against best individual modality at this cutoff (per-cutoff best).
                bi_mean_c = float("nan")
                if (not indiv_df.empty) and (col in indiv_df.columns):
                    bi_mean_c = float(indiv_df.groupby('modality')[col].mean().max())
                elif col in baseline_df.columns:
                    # Fallback for backward compatibility if individual table is unavailable.
                    bi_grp = baseline_df[baseline_df['baseline'].str.startswith('Best Individual')]
                    if not bi_grp.empty:
                        bi_mean_c = float(bi_grp[col].mean())
                if np.isnan(bi_mean_c):
                    continue
                if bi_mean_c <= 0:
                    continue
                cwra_mean_c = float(row_c['test_ef_mean'].iloc[0])
                delta_pct = 100 * (cwra_mean_c / bi_mean_c - 1)
                all_ef_parts.append(f"EF@{c_str}% {delta_pct:+.1f}%")
            if all_ef_parts:
                print(f"  CWRA vs best-indiv (all EF): {', '.join(all_ef_parts)}")

        print(f"\nOptimized weights (mean across folds):")
        for _, row in mean_weights_df.iterrows():
            weight = row['weight']
            print(
                f"  {row['modality']:>20s}: {weight:.4f} "
                f"({row['weight_pct']:.1f}%)"
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
                    _fmt_pm(grp['test_ef_2.5'].mean(), grp['test_ef_2.5'].std(ddof=0)) if 'test_ef_2.5' in grp.columns else 'N/A',
                    _fmt_pm(grp['test_ef_5'].mean(), grp['test_ef_5'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_10'].mean(), grp['test_ef_10'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_20'].mean(), grp['test_ef_20'].std(ddof=0)) if 'test_ef_20' in grp.columns else 'N/A',
                ])
            rows.sort(key=lambda r: float(r[1].split()[0]), reverse=True)
            _print_table(
                "Individual modality performance (test set, mean +/- std)",
                ["Modality", "EF@1%", "EF@2.5%", "EF@5%", "EF@10%", "EF@20%"],
                rows,
            )

        # Combination methods (test set)
        combo_rows = []
        method_labels = {
            'fair': 'DE Fair',
            'unconstrained': 'DE Unconstrained',
            'entropy': 'DE Entropy',
        }
        method_keys = ['fair', 'unconstrained', 'entropy']
        if config.use_bedroc:
            method_labels = {
                'fair_bedroc': 'DE Fair (BEDROC)',
                'unconstrained_bedroc': 'DE Unconstrained (BEDROC)',
                'entropy_bedroc': 'DE Entropy (BEDROC)',
            }
            method_keys = ['fair_bedroc', 'unconstrained_bedroc', 'entropy_bedroc']
        for key in method_keys:
            grp = method_df[method_df['method'] == key]
            if grp.empty:
                continue
            combo_rows.append([
                method_labels[key],
                _fmt_pm(grp['test_ef_1'].mean(), grp['test_ef_1'].std(ddof=0)),
                _fmt_pm(grp['test_ef_2.5'].mean(), grp['test_ef_2.5'].std(ddof=0)) if 'test_ef_2.5' in grp.columns else 'N/A',
                _fmt_pm(grp['test_ef_5'].mean(), grp['test_ef_5'].std(ddof=0)),
                _fmt_pm(grp['test_ef_10'].mean(), grp['test_ef_10'].std(ddof=0)),
                _fmt_pm(grp['test_ef_20'].mean(), grp['test_ef_20'].std(ddof=0)) if 'test_ef_20' in grp.columns else 'N/A',
            ])

        for baseline_name in baseline_df['baseline'].unique():
            if baseline_name == 'Equal Weights' or baseline_name.startswith('Random'):
                grp = baseline_df[baseline_df['baseline'] == baseline_name]
                combo_rows.append([
                    baseline_name,
                    _fmt_pm(grp['test_ef_1'].mean(), grp['test_ef_1'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_2.5'].mean(), grp['test_ef_2.5'].std(ddof=0)) if 'test_ef_2.5' in grp.columns else 'N/A',
                    _fmt_pm(grp['test_ef_5'].mean(), grp['test_ef_5'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_10'].mean(), grp['test_ef_10'].std(ddof=0)),
                    _fmt_pm(grp['test_ef_20'].mean(), grp['test_ef_20'].std(ddof=0)) if 'test_ef_20' in grp.columns else 'N/A',
                ])

        if combo_rows:
            _print_table(
                "Combination methods (test set, mean +/- std)",
                ["Method", "EF@1%", "EF@2.5%", "EF@5%", "EF@10%", "EF@20%"],
                combo_rows,
            )

        if config.report_extra_metrics and not extra_df.empty:
            print(f"\n--- Extra metrics (test set, mean +/- std) ---")
            auroc_mean = extra_df['test_auroc'].mean()
            auroc_std = extra_df['test_auroc'].std(ddof=0)
            auprc_mean = extra_df['test_auprc'].mean()
            auprc_std = extra_df['test_auprc'].std(ddof=0)
            bed_mean = extra_df['test_bedroc'].mean()
            bed_std = extra_df['test_bedroc'].std(ddof=0)
            print(f"AUROC:  {_fmt_pm(auroc_mean, auroc_std)}")
            print(f"AUPRC:  {_fmt_pm(auprc_mean, auprc_std)}")
            print(f"BEDROC: {_fmt_pm(bed_mean, bed_std)}")

        print(f"\nDone! Elapsed: {elapsed:.1f}s")

    return perf_df, summary_df


# =============================================================================
# PAPER TABLE BUILDER
# =============================================================================


def build_paper_table(
    indiv_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    method_df: pd.DataFrame,
    chosen_method: str = "fair",
    extra_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aggregate per-fold CV results into a single paper-ready table.

    Returns a DataFrame with columns:
        method, ef_1, ef_2.5, ..., ef_20, hits_1, ..., hits_20
        (and their _std variants)

    Rows: one per individual modality, then Equal-weight, then CWRA.
    Sorted by EF@1% descending within each group.
    """
    rows = []

    # --- Individual modalities ---
    for name, grp in indiv_df.groupby("modality"):
        row = {"method": name, "kind": "individual"}
        for c in CUTOFFS:
            ef_col = f"test_ef_{c}"
            hits_col = f"test_hits_{c}"
            if ef_col in grp.columns:
                row[f"ef_{c}"] = grp[ef_col].mean()
                row[f"ef_{c}_std"] = grp[ef_col].std(ddof=0)
            if hits_col in grp.columns:
                row[f"hits_{c}"] = grp[hits_col].mean()
                row[f"hits_{c}_std"] = grp[hits_col].std(ddof=0)
        if "test_auroc" in grp.columns:
            row["auroc"] = grp["test_auroc"].mean()
            row["auroc_std"] = grp["test_auroc"].std(ddof=0)
        if "test_auprc" in grp.columns:
            row["auprc"] = grp["test_auprc"].mean()
            row["auprc_std"] = grp["test_auprc"].std(ddof=0)
        if "test_bedroc" in grp.columns:
            row["bedroc"] = grp["test_bedroc"].mean()
            row["bedroc_std"] = grp["test_bedroc"].std(ddof=0)
        rows.append(row)

    # --- Equal-weight baseline ---
    eq = baseline_df[baseline_df["baseline"] == "Equal Weights"]
    if not eq.empty:
        row = {"method": "Equal-weight", "kind": "baseline"}
        for c in CUTOFFS:
            ef_col = f"test_ef_{c}"
            hits_col = f"test_hits_{c}"
            if ef_col in eq.columns:
                row[f"ef_{c}"] = eq[ef_col].mean()
                row[f"ef_{c}_std"] = eq[ef_col].std(ddof=0)
            if hits_col in eq.columns:
                row[f"hits_{c}"] = eq[hits_col].mean()
                row[f"hits_{c}_std"] = eq[hits_col].std(ddof=0)
        if "test_auroc" in eq.columns:
            row["auroc"] = eq["test_auroc"].mean()
            row["auroc_std"] = eq["test_auroc"].std(ddof=0)
        if "test_auprc" in eq.columns:
            row["auprc"] = eq["test_auprc"].mean()
            row["auprc_std"] = eq["test_auprc"].std(ddof=0)
        if "test_bedroc" in eq.columns:
            row["bedroc"] = eq["test_bedroc"].mean()
            row["bedroc_std"] = eq["test_bedroc"].std(ddof=0)
        rows.append(row)

    # --- CWRA (chosen method) ---
    cwra = method_df[method_df["method"] == chosen_method]
    if not cwra.empty:
        row = {"method": "CWRA", "kind": "cwra"}
        for c in CUTOFFS:
            ef_col = f"test_ef_{c}"
            hits_col = f"test_hits_{c}"
            if ef_col in cwra.columns:
                row[f"ef_{c}"] = cwra[ef_col].mean()
                row[f"ef_{c}_std"] = cwra[ef_col].std(ddof=0)
            if hits_col in cwra.columns:
                row[f"hits_{c}"] = cwra[hits_col].mean()
                row[f"hits_{c}_std"] = cwra[hits_col].std(ddof=0)
        if extra_df is not None and not extra_df.empty and "test_auroc" in extra_df.columns:
            extra_sel = extra_df[extra_df["method"] == chosen_method]
            if not extra_sel.empty:
                row["auroc"] = extra_sel["test_auroc"].mean()
                row["auroc_std"] = extra_sel["test_auroc"].std(ddof=0)
                row["auprc"] = extra_sel["test_auprc"].mean()
                row["auprc_std"] = extra_sel["test_auprc"].std(ddof=0)
                row["bedroc"] = extra_sel["test_bedroc"].mean()
                row["bedroc_std"] = extra_sel["test_bedroc"].std(ddof=0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort: individuals by EF@1% desc, then baselines, then CWRA last
    kind_order = {"individual": 0, "baseline": 1, "cwra": 2}
    df["_sort_kind"] = df["kind"].map(kind_order)
    df = df.sort_values(
        ["_sort_kind", "ef_1"], ascending=[True, False]
    ).drop(columns=["_sort_kind"]).reset_index(drop=True)

    return df


def paper_table_to_latex(
    df: pd.DataFrame,
    caption: str = (
        "Virtual screening performance of individual scoring methods and "
        "fusion baselines on the VDR ligand discovery task (5-fold CV, "
        "mean $\\pm$ std over test folds). "
        "EF: Enrichment Factor; Hits: number of active compounds retrieved. "
        "\\textbf{Bold}: best; \\underline{underlined}: second best."
    ),
    label: str = "tab:fusion_performance",
    show_std: bool = True,
    show_extra_metrics: bool = False,
) -> str:
    """
    Convert the paper table DataFrame to LaTeX source.

    Parameters
    ----------
    df : DataFrame from build_paper_table()
    caption : LaTeX caption string
    label : LaTeX label string
    show_std : if True, format as mean±std; if False, mean only

    Returns
    -------
    LaTeX string ready for inclusion in a .tex file.
    """
    cutoffs = CUTOFFS

    # Identify best and second-best per column
    def _rank_col(col_name):
        vals = df[col_name].values
        order = np.argsort(-vals)  # descending
        return order[0], order[1] if len(order) > 1 else (order[0], order[0])

    best_idx = {}
    second_idx = {}
    for c in cutoffs:
        for prefix in ["ef", "hits"]:
            col = f"{prefix}_{c}"
            if col in df.columns:
                b, s = _rank_col(col)
                best_idx[col] = b
                second_idx[col] = s
    if show_extra_metrics:
        for col in ["auroc", "auprc", "bedroc"]:
            if col in df.columns:
                b, s = _rank_col(col)
                best_idx[col] = b
                second_idx[col] = s

    # Method name formatting for LaTeX
    latex_names = {
        "GraphDTA_Kd": r"GraphDTA $K_{d}$",
        "GraphDTA_Ki": r"GraphDTA $K_{i}$",
        "GraphDTA_IC50": r"GraphDTA $IC_{50}$",
        "MLTLE_pKd": r"MLTLE $pK_d$",
        "Vina": "AutoDock Vina",
        "Boltz_affinity": "Boltz-2 affinity",
        "TankBind": "TankBind",
        "DrugBAN": "DrugBAN",
        "MolTrans": "MolTrans",
        "Equal-weight": "Equal-weight",
        "CWRA": "CWRA",
    }

    def _fmt_val(row_idx, col, is_hits=False):
        val = df.loc[row_idx, col]
        std_col = f"{col}_std"
        std = df.loc[row_idx, std_col] if show_std and std_col in df.columns else None

        if is_hits:
            txt = f"{val:.0f}"
            if show_std and std is not None and std > 0:
                txt = f"{val:.0f}$\\pm${std:.0f}"
        else:
            txt = f"{val:.2f}"
            if show_std and std is not None and std > 0:
                txt = f"{val:.2f}$\\pm${std:.2f}"

        if best_idx.get(col) == row_idx:
            txt = f"\\textbf{{{txt}}}"
        elif second_idx.get(col) == row_idx:
            txt = f"\\underline{{{txt}}}"
        return txt

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table*}[htbp!]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")

    n_c = len(cutoffs)
    col_spec = "l" + "r" * n_c + "|" + "r" * n_c
    if show_extra_metrics:
        col_spec += "|" + "r" * 3
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1
    ef_header = f"\\multicolumn{{{n_c}}}{{c|}}{{Enrichment Factor (EF)}}"
    hits_header = f"\\multicolumn{{{n_c}}}{{c}}{{Hits}}"
    if show_extra_metrics:
        extra_header = "\\multicolumn{3}{c}{Extra Metrics}"
        lines.append(f"& {ef_header} & {hits_header} & {extra_header} \\\\")
    else:
        lines.append(f"& {ef_header} & {hits_header} \\\\")

    # Header row 2
    ef_cols = " & ".join([f"@{c}\\%" for c in cutoffs])
    hits_cols = " & ".join([f"@{c}\\%" for c in cutoffs])
    if show_extra_metrics:
        lines.append(f"Method & {ef_cols} & {hits_cols} & AUROC & AUPRC & BEDROC \\\\")
    else:
        lines.append(f"Method & {ef_cols} & {hits_cols} \\\\")
    lines.append(r"\midrule")

    # Data rows
    prev_kind = None
    for idx, row in df.iterrows():
        kind = row["kind"]
        if prev_kind == "individual" and kind != "individual":
            lines.append(r"\midrule")
        prev_kind = kind

        name = latex_names.get(row["method"], row["method"])

        ef_cells = []
        hits_cells = []
        for c in cutoffs:
            ef_col = f"ef_{c}"
            hits_col = f"hits_{c}"
            ef_cells.append(_fmt_val(idx, ef_col, is_hits=False))
            hits_cells.append(_fmt_val(idx, hits_col, is_hits=True))

        cells = ef_cells + hits_cells
        if show_extra_metrics:
            for col in ["auroc", "auprc", "bedroc"]:
                if col in df.columns:
                    cells.append(_fmt_val(idx, col, is_hits=False))
                else:
                    cells.append("N/A")
        lines.append(f"{name} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CWRA with Train/Test Split for Honest Evaluation'
    )
    # Example:
    # python cwra.py -i data/composed_modalities_with_rdkit.csv -o results/bedroc \
    #   --norm minmax --seed 42 --bedroc --latex --extra-metrics --strict-cv \
    #   --fold-honest-unimol --unimol-embeddings data/unimol_embeddings.npz

    parser.add_argument('--input', '-i', default='data/composed_modalities_with_rdkit.csv',
        help='Input CSV file (default: data/composed_modalities_with_rdkit.csv)',
    )
    parser.add_argument('--output', '-o', default='cwra_cv_results', help='Output directory')
    parser.add_argument('--method', '-m', default='fair', choices=['unconstrained', 'fair', 'entropy'], help='Optimization method')
    parser.add_argument('--min-weight', type=float, default=0.03)
    parser.add_argument('--max-weight', type=float, default=0.25)
    parser.add_argument('--norm', default='minmax', choices=['minmax', 'rank', 'robust']) 
    parser.add_argument('--objective', default='default', choices=['default', 'sharp', 'top_heavy', 'balanced', 'custom'])
    parser.add_argument('--bedroc', action='store_true', default=False) 
    parser.add_argument('--bedroc-alpha', type=float, default=80.0)
    parser.add_argument('--max-mw', type=float, default=600) 
    parser.add_argument('--max-rotb', type=int, default=15)
    parser.add_argument('--smiles-col', type=str, default='smiles')
    parser.add_argument(
        '--drop-modalities',
        nargs='+',
        default=[],
        help="Modalities to drop by key and/or display name (e.g. unimol_similarity or UniMol_sim)",
    )
    parser.add_argument('--auto-prune', type=float, default=0.0)
    parser.add_argument('--de-maxiter', type=int, default=1000)
    parser.add_argument('--de-seeds', type=int, default=1)
    parser.add_argument(
        '--de-workers',
        type=int,
        default=-1,
        help='SciPy differential_evolution workers (-1 = all CPUs, 1 = single process)',
    )
    parser.add_argument('--strict-cv', dest='strict_cv', action='store_true')
    parser.add_argument('--no-strict-cv', dest='strict_cv', action='store_false')
    parser.set_defaults(strict_cv=False)
    parser.add_argument('--fold-honest-unimol', action='store_true', default=False, help='Recompute Uni-Mol similarity per CV fold using training-fold actives only.')
    parser.add_argument('--unimol-embeddings', type=str, default=None, help="Path to Uni-Mol embeddings .npz containing keys 'smiles' and 'emb'.")
    parser.add_argument('--extra-metrics', dest='report_extra_metrics', action='store_true')
    parser.add_argument('--no-extra-metrics', dest='report_extra_metrics', action='store_false')
    parser.set_defaults(report_extra_metrics=True)
    parser.add_argument('--report-bedroc-alpha', type=float, default=80.0)
    parser.add_argument(
        '--include-newref-137-as-active',
        action='store_true',
        default=False,
        help="Treat source 'newRef_137' as active (remove from exclude_sources and add to active_sources).",
    )
    parser.add_argument('--self-test', action='store_true', default=False)
    parser.add_argument('--train-frac', type=float, default=0.7, help='Fraction of actives for training (default: 0.7)')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of repeated active train/test splits (independent of --train-frac)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for both DE and active split')
    parser.add_argument('--latex', action='store_true', default=False, help='Generate LaTeX table source file')
    parser.add_argument('--no-std', action='store_true', default=False, help='Omit ±std in LaTeX table (show means only)')

    args = parser.parse_args()

    if args.fold_honest_unimol and not args.unimol_embeddings:
        parser.error(
            "--fold-honest-unimol requires --unimol-embeddings <path_to_npz>. "
            "Example: --fold-honest-unimol --unimol-embeddings data/unimol_embeddings.npz"
        )

    if args.self_test:
        rng = np.random.RandomState(0)
        v = rng.randn(9)
        w = project_to_capped_simplex(v, 0.03, 0.4)
        assert abs(w.sum() - 1.0) < 1e-10
        assert (w >= 0.03 - 1e-12).all() and (w <= 0.4 + 1e-12).all()
        try:
            project_to_capped_simplex(np.ones(9), 0.2, 1.0)
            raise AssertionError("Expected infeasible bounds to raise.")
        except ValueError:
            pass
        df_test = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0, 100.0]})
        mods = {'x': ('high', 'X')}
        exclude = np.array([False, False, False, False, True])
        X_excl, _, _ = normalize_modalities_cv(df_test, mods, 'rank', exclude_mask=exclude)
        X_full, _, _ = normalize_modalities_cv(df_test, mods, 'rank', exclude_mask=None)
        assert X_excl[3, 0] > X_full[3, 0]
        print("SELF-TESTS PASSED")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"cwra_cv_run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    default_cfg = CWRAConfig()
    active_sources = list(default_cfg.active_sources)
    exclude_sources = list(default_cfg.exclude_sources)
    if args.include_newref_137_as_active:
        if "newRef_137" not in active_sources:
            active_sources.append("newRef_137")
        exclude_sources = [s for s in exclude_sources if s != "newRef_137"]

    config = CWRAConfig(
        active_sources=active_sources,
        exclude_sources=exclude_sources,
        method=args.method,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        norm_method=args.norm,
        objective_preset=args.objective,
        use_bedroc=args.bedroc,
        bedroc_alpha=args.bedroc_alpha,
        max_mw=args.max_mw,
        max_rotb=args.max_rotb,
        smiles_col=args.smiles_col,
        drop_modalities=args.drop_modalities,
        auto_prune_threshold=args.auto_prune,
        de_maxiter=args.de_maxiter,
        de_n_seeds=args.de_seeds,
        de_workers=args.de_workers,
        strict_cv=args.strict_cv,
        fold_honest_unimol=args.fold_honest_unimol,
        unimol_embeddings_path=args.unimol_embeddings,
        report_extra_metrics=args.report_extra_metrics,
        report_bedroc_alpha=args.report_bedroc_alpha,
        de_seed=args.seed,
        split_seed=args.seed,
        train_frac=args.train_frac,
    )

    with open(log_path, 'w', encoding='utf-8') as log_f:
        tee_out = TeeStream(sys.stdout, log_f)
        tee_err = TeeStream(sys.stderr, log_f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Logging console output to {log_path}")
            df = pd.read_csv(args.input)
            print(f"Loaded {len(df):,} compounds")
            if args.include_newref_137_as_active:
                print("Including 'newRef_137' as active source.")

            if args.cv_folds and args.cv_folds > 1:
                run_cross_validation(df, config, args.cv_folds, output_dir=output_dir, verbose=True)

                if args.latex:
                    prefix = config.output_prefix
                    paper_csv = output_dir / f'{prefix}_paper_table.csv'
                    if paper_csv.exists():
                        paper_df = pd.read_csv(paper_csv)
                        latex_src = paper_table_to_latex(paper_df, show_std=not args.no_std)
                        latex_path = output_dir / f'{prefix}_table.tex'
                        with open(latex_path, 'w', encoding='utf-8') as f:
                            f.write(latex_src)
                        print(f"LaTeX table written to {latex_path}")
            else:
                cwra = CWRATrainTest(config)
                cwra.fit(df)

                df_ranked = cwra.transform(df)
                df_ranked.to_csv(output_dir / f'{config.output_prefix}_rankings.csv', index=False)
                cwra.save_results(output_dir)

                print("\nDone!")


if __name__ == '__main__':
    main()
