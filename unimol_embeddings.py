#!/usr/bin/env python3
"""Build Uni-Mol embeddings (.npz) for fold-honest CWRA CV."""

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from unimol_tools import UniMolRepr

INPUT_CSV = "data/composed_modalities_with_rdkit.csv"
OUTPUT_NPZ = "data/unimol_embeddings.npz"
SMILES_COL = "smiles"
BATCH_SIZE = 64


def _rdkit_valid_smiles(smi: str) -> bool:
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


def _ordered_unique(values):
    seen = set()
    out = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    if SMILES_COL not in df.columns:
        raise ValueError(f"Missing SMILES column '{SMILES_COL}' in {INPUT_CSV}")

    smiles_all = _ordered_unique(df[SMILES_COL].dropna().astype(str).str.strip().tolist())
    print(f"Loaded {len(df):,} rows, {len(smiles_all):,} unique non-null SMILES")

    # Fast prefilter so one illegal SMILES does not crash an entire Uni-Mol batch.
    smiles_valid = [s for s in smiles_all if _rdkit_valid_smiles(s)]
    smiles_invalid = [s for s in smiles_all if not _rdkit_valid_smiles(s)]
    print(f"RDKit-valid SMILES: {len(smiles_valid):,}")
    if smiles_invalid:
        print(f"RDKit-invalid SMILES skipped: {len(smiles_invalid):,}")

    model = UniMolRepr(data_type="molecule", remove_hs=False)
    kept_smiles = []
    emb_rows = []
    unimol_failed = []

    for i in tqdm(range(0, len(smiles_valid), BATCH_SIZE), desc="UniMol embeddings"):
        batch = smiles_valid[i : i + BATCH_SIZE]
        try:
            batch_repr = model.get_repr(batch)
            if len(batch_repr) != len(batch):
                raise RuntimeError(
                    f"Uni-Mol returned {len(batch_repr)} embeddings for {len(batch)} SMILES."
                )
            for smi, vec in zip(batch, batch_repr):
                kept_smiles.append(smi)
                emb_rows.append(vec)
        except Exception:
            # Fallback to per-SMILES handling to isolate failures.
            for smi in batch:
                try:
                    vec = model.get_repr([smi])[0]
                    kept_smiles.append(smi)
                    emb_rows.append(vec)
                except Exception as exc:
                    unimol_failed.append((smi, str(exc)))

    if not emb_rows:
        raise RuntimeError("No Uni-Mol embeddings were generated.")

    emb = np.asarray(emb_rows, dtype=np.float32)
    np.savez_compressed(
        OUTPUT_NPZ,
        smiles=np.asarray(kept_smiles, dtype=object),
        emb=emb,
    )
    print(f"Saved {OUTPUT_NPZ} with shape {emb.shape}")

    if smiles_invalid or unimol_failed:
        failed_path = OUTPUT_NPZ.replace(".npz", "_failed_smiles.txt")
        with open(failed_path, "w", encoding="utf-8") as f:
            if smiles_invalid:
                f.write("[RDKit invalid]\n")
                for smi in smiles_invalid:
                    f.write(f"{smi}\n")
            if unimol_failed:
                f.write("\n[Uni-Mol failed]\n")
                for smi, reason in unimol_failed:
                    f.write(f"{smi}\t{reason}\n")
        print(f"Wrote failed SMILES report: {failed_path}")

    print(
        f"Summary: embedded={len(kept_smiles):,}, "
        f"rdkit_invalid={len(smiles_invalid):,}, unimol_failed={len(unimol_failed):,}"
    )


if __name__ == "__main__":
    main()
