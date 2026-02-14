#!/usr/bin/env python3
"""
Step B (bypass): compute CWRA consensus scores and output meta_scores.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from cwra_cv_v2 import CWRAConfig, OPTIMIZERS, normalize_modalities

logger = logging.getLogger(__name__)

DEFAULT_MODALITIES: Dict[str, Tuple[str, str]] = {
    "graphdta_kd": ("high", "GraphDTA_Kd"),
    "graphdta_ki": ("high", "GraphDTA_Ki"),
    "graphdta_ic50": ("high", "GraphDTA_IC50"),
    "mltle_pKd": ("high", "MLTLE_pKd"),
    "vina_score": ("low", "Vina"),
    "boltz_affinity": ("low", "Boltz_affinity"),
    "tankbind_affinity": ("low", "TankBind"),
    "drugban_affinity": ("low", "DrugBAN"),
    "moltrans_affinity": ("low", "MolTrans"),
    "unimol_similarity": ("high", "UniMol_sim"),
    "boltz_confidence": ("high", "Boltz_confidence"),
}


def _choose_smiles_col(df: pd.DataFrame) -> Optional[str]:
    if "smiles" in df.columns:
        return "smiles"
    if "SMILES" in df.columns:
        return "SMILES"
    return None


def _align_labels(labels_df: pd.DataFrame, df_pool: pd.DataFrame) -> pd.Series:
    if "index" in labels_df.columns:
        labels_df = labels_df.set_index("index")
        labels_df = labels_df.reindex(df_pool.index)
        if labels_df["pu_label"].isna().any():
            raise ValueError("pu_labels index mismatch with df_pool.")
        return labels_df["pu_label"].astype(int)
    if "id" in labels_df.columns and "id" in df_pool.columns:
        labels_df = labels_df.set_index("id")
        labels_df = labels_df.reindex(df_pool["id"].values)
        if labels_df["pu_label"].isna().any():
            raise ValueError("pu_labels id mismatch with df_pool.")
        return labels_df["pu_label"].astype(int).reset_index(drop=True)
    if len(labels_df) != len(df_pool):
        raise ValueError("pu_labels length does not match df_pool.")
    return labels_df["pu_label"].astype(int)


def _load_modalities_json(path: str) -> Dict[str, Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("modalities JSON must be a dict of column -> (direction, name).")
    modalities: Dict[str, Tuple[str, str]] = {}
    for col, val in data.items():
        if isinstance(val, (list, tuple)) and len(val) == 2:
            direction, name = val
        elif isinstance(val, dict) and "direction" in val and "name" in val:
            direction, name = val["direction"], val["name"]
        else:
            raise ValueError(
                f"Invalid modality entry for '{col}'. Expected [direction, name] or "
                f"{{'direction': ..., 'name': ...}}."
            )
        modalities[str(col)] = (str(direction), str(name))
    return modalities


def _build_weight_vector(
    weights_df: pd.DataFrame,
    modalities: Dict[str, Tuple[str, str]],
    available_cols: list[str],
) -> np.ndarray:
    display_to_col = {name: col for col, (_, name) in modalities.items()}
    weights_by_col: Dict[str, float] = {}

    if "modality" in weights_df.columns:
        for _, row in weights_df.iterrows():
            name = str(row["modality"])
            weight = float(row["weight"])
            if name in display_to_col:
                col = display_to_col[name]
            elif name in modalities:
                col = name
            else:
                raise ValueError(f"Unknown modality in weights: {name}")
            if col in weights_by_col:
                raise ValueError(f"Duplicate modality in weights: {name}")
            weights_by_col[col] = weight
    elif "column" in weights_df.columns:
        for _, row in weights_df.iterrows():
            col = str(row["column"])
            weight = float(row["weight"])
            if col not in modalities:
                raise ValueError(f"Unknown column in weights: {col}")
            if col in weights_by_col:
                raise ValueError(f"Duplicate column in weights: {col}")
            weights_by_col[col] = weight
    else:
        raise ValueError("weights CSV must contain 'modality' or 'column' and 'weight' columns.")

    missing = [c for c in available_cols if c not in weights_by_col]
    if missing:
        raise ValueError(f"Missing weights for modalities: {missing}")

    extra = [c for c in weights_by_col if c not in available_cols]
    if extra:
        raise ValueError(f"Weights provided for unavailable modalities: {extra}")

    return np.array([weights_by_col[c] for c in available_cols], dtype=float)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step B (bypass): CWRA consensus scores.")
    parser.add_argument("--input", default="data/composed_modalities_with_rdkit.csv")
    parser.add_argument("--pu-labels", required=True, help="PU labels CSV from Step A")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--method",
        choices=["fair", "unconstrained", "entropy"],
        default="fair",
        help="CWRA optimization method",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--norm",
        choices=["minmax", "rank", "robust"],
        default="minmax",
        help="Normalization method (must match CWRA CV configuration).",
    )
    parser.add_argument(
        "--weights-csv",
        default=None,
        help="Path to pre-computed weights CSV (columns: modality/column, weight).",
    )
    parser.add_argument(
        "--weights-json",
        default=None,
        help="Inline JSON dict of modality display names -> weights.",
    )
    parser.add_argument(
        "--modalities-json",
        default=None,
        help="Optional JSON file to override default modalities mapping.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = CWRAConfig()
    modalities = DEFAULT_MODALITIES
    if args.modalities_json:
        modalities = _load_modalities_json(args.modalities_json)

    df = pd.read_csv(args.input)
    df_pool = df[~df["source"].isin(config.exclude_sources)].copy()

    labels_df = pd.read_csv(args.pu_labels)
    pu_label = _align_labels(labels_df, df_pool)

    X_mod, available_cols, mod_names = normalize_modalities(
        df_pool, modalities, norm_method=args.norm
    )
    logger.info("X_mod shape: %s, modalities: %s", X_mod.shape, mod_names)

    active_mask = df_pool["source"].isin(config.active_sources).values
    logger.info("Active count: %d", int(active_mask.sum()))

    weights_source = None
    weights = None
    if args.weights_csv:
        weights_df = pd.read_csv(args.weights_csv)
        if "weight" not in weights_df.columns:
            raise ValueError("weights CSV must contain a 'weight' column.")
        weights = _build_weight_vector(weights_df, modalities, available_cols)
        weights_source = f"weights CSV: {args.weights_csv}"
    elif args.weights_json:
        try:
            weights_map = json.loads(args.weights_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --weights-json: {exc}") from exc
        weights_df = pd.DataFrame(
            [{"modality": k, "weight": v} for k, v in weights_map.items()]
        )
        weights = _build_weight_vector(weights_df, modalities, available_cols)
        weights_source = "inline weights JSON"

    if weights_source:
        weights = np.asarray(weights, dtype=float)
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative.")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("Sum of weights must be positive.")
        weights = weights / total
        logger.info("Using pre-computed weights from %s", weights_source)
        for name, w in zip(mod_names, weights):
            logger.info("Weights: %s: %.4f (%.1f%%)", name, w, w * 100)
        logger.info("Skipping DE optimization.")
    else:
        optimizer = OPTIMIZERS[args.method]
        config_opt = CWRAConfig()
        config_opt.de_seed = args.seed
        config_opt.method = args.method
        weights = optimizer(X_mod, active_mask, config_opt)
        logger.info(
            "Optimized weights (%s): %s",
            args.method,
            dict(zip(mod_names, [f"{w:.4f}" for w in weights])),
        )

    scores = X_mod @ weights
    logger.info(
        "Score range: [%.4f, %.4f], median=%.4f",
        float(scores.min()),
        float(scores.max()),
        float(np.median(scores)),
    )

    for label, name in [(1, "positives"), (0, "RN"), (-1, "unlabeled")]:
        mask = pu_label.values == label
        if mask.any():
            logger.info(
                "  %s: median=%.4f, mean=%.4f, min=%.4f, max=%.4f",
                name,
                float(np.median(scores[mask])),
                float(scores[mask].mean()),
                float(scores[mask].min()),
                float(scores[mask].max()),
            )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_scores = pd.DataFrame(
        {
            "meta_score": scores,
            "pu_label": pu_label.values,
            "source": df_pool["source"].values,
        }
    )
    if "id" in df_pool.columns:
        meta_scores.insert(0, "id", df_pool["id"].values)
    else:
        meta_scores.insert(0, "index", df_pool.index.values)

    smiles_col = _choose_smiles_col(df_pool)
    if smiles_col:
        meta_scores[smiles_col] = df_pool[smiles_col].values

    meta_scores.to_csv(output_dir / "meta_scores.csv", index=False)

    weights_info = {
        "method": args.method,
        "seed": args.seed,
        "modality_names": mod_names,
        "modality_keys": available_cols,
        "weights": weights.tolist(),
    }
    with open(output_dir / "cwra_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_info, f, indent=2)

    logger.info("Wrote outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
