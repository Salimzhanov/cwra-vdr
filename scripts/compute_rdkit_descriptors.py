#!/usr/bin/env python3
"""Compute RDKit physicochemical descriptors for compounds in a CSV.

Reads an input CSV containing a SMILES column, computes all descriptors
with RDKit for every row (no fallback, no skip-if-present), and writes
an output CSV with the descriptor columns appended/overwritten.

Descriptors (exact column names):
    MW               – Molecular weight (Descriptors.ExactMolWt)
    cLogP            – Crippen logP
    tPSA             – Topological polar surface area
    HBD              – H-bond donors  (Lipinski definition)
    HBA              – H-bond acceptors (Lipinski definition)
    RotB             – Rotatable bonds (Lipinski definition)
    RingCount        – Total ring count (SSSR)
    AromaticRingCount– Aromatic ring count
    FractionCSP3     – Fraction of sp3 carbons
    HeavyAtomCount   – Heavy (non-hydrogen) atom count
    FormalCharge      – Net formal charge (Chem.GetFormalCharge)
    MR               – Molar refractivity (Crippen)

Invalid / empty SMILES produce NaN for all descriptor columns.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors, QED
from rdkit.Chem import GraphDescriptors

# Suppress RDKit parse warnings (invalid SMILES); we handle them explicitly.
RDLogger.logger().setLevel(RDLogger.ERROR)


def _sa_score(mol: Chem.Mol) -> float:
    """Synthetic Accessibility score (Ertl & Schuffenhauer, J. Cheminform. 2009).

    Uses RDKit's contrib implementation.  Returns a float in [1, 10] where
    1 = easy to synthesise, 10 = hard.
    """
    try:
        from rdkit.Chem import RDConfig
        import os, importlib.util
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
        spec = importlib.util.spec_from_file_location("sascorer", sa_path)
        sascorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sascorer)
        return sascorer.calculateScore(mol)
    except Exception:
        return float("nan")

DESCRIPTOR_COLUMNS: List[str] = [
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


def compute_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Compute all descriptors for a valid RDKit Mol object."""
    return {
        "MW": Descriptors.ExactMolWt(mol),
        "cLogP": Crippen.MolLogP(mol),
        "tPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "MR": Crippen.MolMR(mol),
        "NumStereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "BertzCT": GraphDescriptors.BertzCT(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "QED": QED.qed(mol),
        "SAScore": _sa_score(mol),
    }


def smiles_to_descriptors(smiles) -> Optional[Dict[str, float]]:
    """Parse a single SMILES and return descriptor dict, or None on failure."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    smi = str(smiles).strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return compute_descriptors(mol)


def compute_rdkit_descriptors(
    input_csv: str,
    output_csv: str,
    smiles_column: str = "smiles",
) -> pd.DataFrame:
    """Read *input_csv*, compute descriptors for every row, write *output_csv*.

    Always recomputes all rows — no "only missing" fallback.
    """
    df = pd.read_csv(input_csv)

    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    n_total = len(df)
    print(f"Computing RDKit descriptors for {n_total} rows …")
    t0 = time.time()

    # Vectorised: apply returns a Series of dicts (or None).
    raw = df[smiles_column].apply(smiles_to_descriptors)

    # Replace None with a dict of NaNs so pd.DataFrame can expand uniformly.
    nan_row = {c: np.nan for c in DESCRIPTOR_COLUMNS}
    desc_df = pd.DataFrame(
        [d if d is not None else nan_row for d in raw],
        columns=DESCRIPTOR_COLUMNS,
        index=df.index,
    )

    # Drop old descriptor columns if they existed, then join fresh ones.
    df = df.drop(columns=[c for c in DESCRIPTOR_COLUMNS if c in df.columns])
    df = pd.concat([df, desc_df], axis=1)

    invalid_count = int(desc_df[DESCRIPTOR_COLUMNS[0]].isna().sum())
    elapsed = time.time() - t0

    df.to_csv(output_csv, index=False)

    print(f"Done in {elapsed:.1f}s  |  valid: {n_total - invalid_count}  |  "
          f"invalid SMILES (NaN): {invalid_count}")
    print(f"Saved: {output_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RDKit physicochemical descriptors for a CSV of SMILES."
    )
    parser.add_argument(
        "--input",
        default="data/composed_modalities.csv",
        help="Input CSV file (default: data/composed_modalities.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/composed_modalities_with_rdkit.csv",
        help="Output CSV file (default: data/composed_modalities_with_rdkit.csv)",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="Name of the SMILES column (default: smiles)",
    )
    args = parser.parse_args()

    compute_rdkit_descriptors(
        input_csv=args.input,
        output_csv=args.output,
        smiles_column=args.smiles_column,
    )


if __name__ == "__main__":
    main()
