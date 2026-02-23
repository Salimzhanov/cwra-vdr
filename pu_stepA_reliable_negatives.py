#!/usr/bin/env python3
"""
CLI wrapper for Step A: build reliable negatives and PU labels.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from cwra import CWRAConfig, normalize_modalities
from pu_rn import build_reliable_negative_mask

logger = logging.getLogger(__name__)

BASE_MODALITIES = {
    "graphdta_kd": ("high", "GraphDTA_Kd"),
    "graphdta_ki": ("high", "GraphDTA_Ki"),
    "graphdta_ic50": ("high", "GraphDTA_IC50"),
    "mltle_pKd": ("high", "MLTLE_pKd"),
    "vina_score": ("low", "Vina"),
    "boltz_affinity": ("low", "Boltz_affinity"),
    "tankbind_affinity": ("low", "TankBind"),
    "drugban_affinity": ("low", "DrugBAN"),
    "moltrans_affinity": ("low", "MolTrans"),
}


def _choose_smiles_col(df: pd.DataFrame) -> Optional[str]:
    if "smiles" in df.columns:
        return "smiles"
    if "SMILES" in df.columns:
        return "SMILES"
    return None


def _configure_active_sources(config: CWRAConfig, include_newref_137_as_active: bool) -> None:
    if not include_newref_137_as_active:
        return
    if "newRef_137" not in config.active_sources:
        config.active_sources.append("newRef_137")
    config.exclude_sources = [s for s in config.exclude_sources if s != "newRef_137"]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step A: Reliable negatives for PU learning.")
    parser.add_argument(
        "--input",
        default="data/composed_modalities_with_rdkit.csv",
        help="Input CSV",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--neg-pos-ratio", type=int, default=10)
    parser.add_argument("--bottom-q", type=float, default=0.20)
    parser.add_argument("--sim-max", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rdkit", action="store_true", help="Disable RDKit similarity filtering")
    parser.add_argument(
        "--include-boltz-confidence",
        action="store_true",
        help="Include Boltz-2 confidence modality in Step A baseline scoring.",
    )
    parser.add_argument(
        "--include-unimol",
        action="store_true",
        help="Include UniMol similarity modality in Step A baseline scoring.",
    )
    parser.add_argument(
        "--include-newref-137-as-active",
        action="store_true",
        default=False,
        help="Treat source 'newRef_137' as active (remove from exclude_sources).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    config = CWRAConfig()
    _configure_active_sources(config, args.include_newref_137_as_active)

    smiles_col = _choose_smiles_col(df)

    df_pool = df[~df["source"].isin(config.exclude_sources)].copy()
    active_mask = df_pool["source"].isin(config.active_sources).values
    logger.info("Loaded %d rows, pool after exclude_sources: %d", len(df), len(df_pool))
    logger.info("Active count: %d", int(active_mask.sum()))
    if args.include_newref_137_as_active:
        logger.info("Including 'newRef_137' as active source.")

    modalities_to_use = dict(BASE_MODALITIES)
    if args.include_boltz_confidence:
        modalities_to_use["boltz_confidence"] = ("high", "Boltz_confidence")
    if args.include_unimol:
        modalities_to_use["unimol_similarity"] = ("high", "UniMol_sim")
    logger.info("Using %d modalities for X_mod", len(modalities_to_use))
    X_mod, _, _ = normalize_modalities(df_pool, modalities_to_use)

    rn_mask = build_reliable_negative_mask(
        df_pool=df_pool,
        active_mask=active_mask,
        X_mod=X_mod,
        neg_pos_ratio=args.neg_pos_ratio,
        bottom_q=args.bottom_q,
        seed=args.seed,
        smiles_col=smiles_col,
        use_rdkit=not args.no_rdkit,
        sim_max=args.sim_max,
    )

    pu_label = pd.Series(-1, index=df_pool.index, dtype=int)
    pu_label.loc[active_mask] = 1
    pu_label.loc[rn_mask] = 0

    if "id" in df_pool.columns:
        id_col = "id"
        id_vals = df_pool[id_col]
    else:
        id_col = "index"
        id_vals = df_pool.index

    labels_df = pd.DataFrame(
        {
            id_col: id_vals,
            "source": df_pool["source"].values,
            "pu_label": pu_label.values,
        }
    )

    labels_df.to_csv(output_dir / "pu_labels.csv", index=False)

    report = {
        "N": int(len(df_pool)),
        "n_pos": int(active_mask.sum()),
        "n_rn": int(rn_mask.sum()),
        "bottom_q": float(args.bottom_q),
        "sim_max": float(args.sim_max),
        "neg_pos_ratio": int(args.neg_pos_ratio),
        "seed": int(args.seed),
        "used_modality_keys": list(modalities_to_use.keys()),
    }
    with open(output_dir / "pu_stepA_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "PU labels: pos=%d rn=%d unlabeled=%d",
        int((pu_label == 1).sum()),
        int((pu_label == 0).sum()),
        int((pu_label == -1).sum()),
    )
    logger.info("Wrote outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
