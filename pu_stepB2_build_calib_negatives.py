#!/usr/bin/env python3
"""
Step B2: build calibration negatives from unlabeled compounds.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _choose_smiles_col(df: pd.DataFrame) -> Optional[str]:
    if "smiles" in df.columns:
        return "smiles"
    if "SMILES" in df.columns:
        return "SMILES"
    return None


def _load_rdkit():
    try:
        from rdkit import Chem, DataStructs  # type: ignore
        from rdkit.Chem import AllChem, rdFingerprintGenerator  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "RDKit is required for --sim-max filtering. "
            "Install RDKit or run without --sim-max."
        ) from exc
    return Chem, AllChem, DataStructs, rdFingerprintGenerator


def _make_fp_func(rdFingerprintGenerator, AllChem):
    if hasattr(rdFingerprintGenerator, "GetMorganGenerator"):
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        def _fp(mol):
            return gen.GetFingerprint(mol)

        return _fp

    def _fp(mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    return _fp


def _align_labels(labels_df: pd.DataFrame, df_raw: pd.DataFrame) -> pd.Series:
    if "index" in labels_df.columns:
        idx = labels_df["index"].values
        df_pool = df_raw.loc[idx]
        labels_df = labels_df.set_index("index").reindex(df_pool.index)
        if labels_df["pu_label"].isna().any():
            raise ValueError("pu_labels index mismatch with input rows.")
        return labels_df["pu_label"].astype(int)
    if len(labels_df) == len(df_raw):
        if "pu_label" not in labels_df.columns:
            raise ValueError("pu_labels must contain pu_label column.")
        return pd.Series(labels_df["pu_label"].values, index=df_raw.index, dtype=int)
    raise ValueError(
        "pu_labels must include an index column or be row-aligned with input."
    )


def _align_meta_scores(meta_scores: Optional[str], df_raw: pd.DataFrame) -> Optional[pd.Series]:
    if meta_scores is None:
        return None
    ms = pd.read_csv(meta_scores)
    if "meta_score" not in ms.columns:
        logger.warning("meta_scores missing meta_score column; skipping.")
        return None
    if "index" in ms.columns:
        ms = ms.set_index("index")
        return ms["meta_score"]
    if "id" in ms.columns and "id" in df_raw.columns:
        ms = ms.set_index("id")
        series = df_raw["id"].map(ms["meta_score"])
        series.index = df_raw.index
        return series
    if len(ms) == len(df_raw):
        return pd.Series(ms["meta_score"].values, index=df_raw.index)
    logger.warning("Could not align meta_scores; skipping meta_score in output.")
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build calibration negatives from unlabeled pool."
    )
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--pu-labels", required=True, help="PU labels CSV from Step A")
    parser.add_argument("--meta-scores", required=False, help="meta_scores.csv from Step B")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-cal", type=int, required=True, help="Number of calibration negatives")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-max", type=float, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df_raw = pd.read_csv(args.input)
    smiles_col = _choose_smiles_col(df_raw)
    if smiles_col is None:
        raise ValueError("Input CSV must contain a smiles or SMILES column.")

    labels_df = pd.read_csv(args.pu_labels)
    pu_label = _align_labels(labels_df, df_raw)

    pos_idx = pu_label.index[pu_label.values == 1].to_numpy()
    unl_idx = pu_label.index[pu_label.values == -1].to_numpy()
    logger.info("Unlabeled pool size: %d", len(unl_idx))
    logger.info("Positive count: %d", len(pos_idx))

    candidate_idx = unl_idx
    if args.sim_max is not None:
        Chem, AllChem, DataStructs, rdFingerprintGenerator = _load_rdkit()
        fp_from_mol = _make_fp_func(rdFingerprintGenerator, AllChem)

        pos_fps = []
        for idx in pos_idx:
            smi = df_raw.loc[idx, smiles_col]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning("Invalid SMILES for positive index %s; skipping.", idx)
                continue
            pos_fps.append(fp_from_mol(mol))
        if len(pos_fps) == 0:
            raise ValueError("No valid positive SMILES available for similarity filtering.")

        kept = []
        for idx in candidate_idx:
            smi = df_raw.loc[idx, smiles_col]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning("Invalid SMILES for unlabeled index %s; skipping.", idx)
                continue
            fp = fp_from_mol(mol)
            sims = DataStructs.BulkTanimotoSimilarity(fp, pos_fps)
            if max(sims) <= args.sim_max:
                kept.append(idx)
        candidate_idx = np.array(kept, dtype=int)
        logger.info(
            "Unlabeled after sim-filter (<= %.3f): %d",
            args.sim_max,
            len(candidate_idx),
        )

    if len(candidate_idx) == 0:
        raise ValueError("Calibration candidate pool is empty.")

    n_cal = min(int(args.n_cal), len(candidate_idx))
    if len(candidate_idx) < int(args.n_cal):
        logger.warning(
            "Candidate pool (%d) smaller than n_cal (%d); using all candidates.",
            len(candidate_idx),
            int(args.n_cal),
        )

    rng = np.random.RandomState(args.seed)
    if n_cal > 0:
        selected_idx = rng.choice(candidate_idx, size=n_cal, replace=False)
    else:
        selected_idx = np.array([], dtype=int)

    meta_scores = _align_meta_scores(args.meta_scores, df_raw)
    out_df = pd.DataFrame({"index": selected_idx, "is_calib_neg": 1})
    if meta_scores is not None:
        out_df["meta_score"] = meta_scores.loc[selected_idx].values

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "calib_negatives.csv", index=False)

    report = {
        "n_unlabeled_total": int(len(unl_idx)),
        "n_pos": int(len(pos_idx)),
        "n_unlabeled_after_simfilter": int(len(candidate_idx)),
        "n_selected": int(len(selected_idx)),
        "sim_max": None if args.sim_max is None else float(args.sim_max),
        "n_cal_requested": int(args.n_cal),
        "seed": int(args.seed),
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Selected %d calibration negatives.", int(len(selected_idx)))
    logger.info("Wrote outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
