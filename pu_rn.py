#!/usr/bin/env python3
"""
Step A core logic: build reliable negatives for PU learning.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_reliable_negative_mask(
    df_pool,
    active_mask,
    X_mod,                     # numpy array [N, M] modality features already normalized to [0,1]
    neg_pos_ratio: int = 10,
    bottom_q: float = 0.20,
    seed: int = 42,
    smiles_col: Optional[str] = None,
    use_rdkit: bool = True,
    sim_max: float = 0.35,
    fp_radius: int = 2,
    fp_nbits: int = 2048,
):
    """
    Build a mask for reliable negatives (RN) among unlabeled samples.

    Returns:
        rn_mask (np.ndarray[bool]): length N
    """
    logger.info(
        "Building reliable negatives: neg_pos_ratio=%s bottom_q=%s seed=%s use_rdkit=%s sim_max=%s",
        neg_pos_ratio,
        bottom_q,
        seed,
        use_rdkit,
        sim_max,
    )
    X_mod = np.asarray(X_mod)
    active_mask = np.asarray(active_mask, dtype=bool)
    n_total = len(active_mask)

    if X_mod.shape[0] != n_total:
        raise ValueError("X_mod and active_mask must have the same length.")

    n_pos = int(active_mask.sum())
    logger.info("Total=%d, positives=%d, unlabeled=%d", n_total, n_pos, n_total - n_pos)
    if n_pos == 0:
        logger.warning("No positives found; returning empty RN mask.")
        return np.zeros(n_total, dtype=bool)

    unlabeled_mask = ~active_mask
    if not np.any(unlabeled_mask):
        logger.warning("No unlabeled samples found; returning empty RN mask.")
        return np.zeros(n_total, dtype=bool)

    s_simple = X_mod.mean(axis=1)
    q = np.quantile(s_simple[unlabeled_mask], bottom_q)
    candidate_mask = unlabeled_mask & (s_simple <= q)
    candidate_idx = np.where(candidate_mask)[0]
    logger.info("Candidate RN pool (bottom_q): %d", candidate_idx.size)

    if candidate_idx.size == 0:
        logger.warning("No candidates in RN pool after bottom-q filter.")
        return np.zeros(n_total, dtype=bool)

    target_size = min(candidate_idx.size, int(neg_pos_ratio * n_pos))
    logger.info("RN target size: %d", target_size)
    if target_size <= 0:
        return np.zeros(n_total, dtype=bool)

    if use_rdkit and smiles_col and smiles_col in df_pool.columns:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit import DataStructs
        except Exception as exc:
            logger.warning("RDKit not available; skipping similarity filtering. (%s)", exc)
        else:
            try:
                from rdkit.Chem import rdFingerprintGenerator
                fp_gen = rdFingerprintGenerator.GetMorganGenerator(
                    radius=fp_radius, fpSize=fp_nbits
                )

                def _fp(mol):
                    return fp_gen.GetFingerprint(mol)

            except Exception:
                logger.warning("MorganGenerator not available; using legacy Morgan fingerprints.")

                def _fp(mol):
                    return AllChem.GetMorganFingerprintAsBitVect(
                        mol, fp_radius, nBits=fp_nbits
                    )

            pos_idx = np.where(active_mask)[0]
            pos_fps = []
            for smi in df_pool.iloc[pos_idx][smiles_col].tolist():
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    logger.warning("Invalid SMILES in positives; skipping: %s", smi)
                    continue
                pos_fps.append(_fp(mol))

            if not pos_fps:
                logger.warning("No valid positive fingerprints; skipping similarity filtering.")
            else:
                kept_idx = []
                for idx in candidate_idx:
                    smi = df_pool.iloc[idx][smiles_col]
                    mol = Chem.MolFromSmiles(str(smi))
                    if mol is None:
                        logger.warning("Invalid SMILES in candidates; excluding: %s", smi)
                        continue
                    fp = _fp(mol)
                    sims = DataStructs.BulkTanimotoSimilarity(fp, pos_fps)
                    if sims and max(sims) <= sim_max:
                        kept_idx.append(idx)

                candidate_idx = np.array(kept_idx, dtype=int)
                logger.info("Candidate RN pool after similarity filter: %d", candidate_idx.size)
                if candidate_idx.size < target_size:
                    logger.warning(
                        "Similarity filtering reduced RN pool below target (%d < %d); using available.",
                        candidate_idx.size,
                        target_size,
                    )
                    target_size = candidate_idx.size
    else:
        if use_rdkit:
            logger.warning("SMILES column missing; skipping similarity filtering.")

    if target_size == 0 or candidate_idx.size == 0:
        return np.zeros(n_total, dtype=bool)

    rng = np.random.RandomState(seed)
    if candidate_idx.size <= target_size:
        selected = candidate_idx
    else:
        selected = rng.choice(candidate_idx, size=target_size, replace=False)

    rn_mask = np.zeros(n_total, dtype=bool)
    rn_mask[selected] = True
    logger.info("Selected RN count: %d", int(rn_mask.sum()))
    return rn_mask
