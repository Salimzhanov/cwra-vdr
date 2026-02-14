#!/usr/bin/env python3
"""
Step B: select and clean precomputed descriptor columns (no RDKit usage).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTOR_COLS = [
    "MW",
    "cLogP",
    "tPSA",
    "HBD",
    "HBA",
    "RotB",
    "RingCount",
    "AromaticRingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "FormalCharge",
    "MR",
    "NumStereocenters",
    "BertzCT",
    "LabuteASA",
    "QED",
    "SAScore",
]


def extract_precomputed_descriptors(
    df: pd.DataFrame,
    descriptor_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Extract and coerce precomputed descriptor columns to numeric.

    Returns:
        DataFrame with selected descriptor columns, or empty DataFrame if none exist.
    """
    if descriptor_cols is None:
        cols = [c for c in DEFAULT_DESCRIPTOR_COLS if c in df.columns]
    else:
        cols = [c for c in descriptor_cols if c in df.columns]

    if not cols:
        logger.warning("No descriptor columns found; returning empty DataFrame.")
        return pd.DataFrame(index=df.index)

    logger.info("Using %d descriptor columns: %s", len(cols), ", ".join(cols))
    desc_df = df[cols].copy()
    for c in cols:
        desc_df[c] = pd.to_numeric(desc_df[c], errors="coerce")
    nan_counts = desc_df.isna().sum()
    if nan_counts.any():
        logger.info("Descriptor NaN counts: %s", nan_counts[nan_counts > 0].to_dict())
    return desc_df


def compute_physchem_descriptors(
    df: pd.DataFrame,
    smiles_col: Optional[str] = None,
    descriptor_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for precomputed descriptor extraction.
    This function intentionally ignores smiles_col and does not use RDKit.
    """
    return extract_precomputed_descriptors(df, descriptor_cols=descriptor_cols)
