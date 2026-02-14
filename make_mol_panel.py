#!/usr/bin/env python3
import argparse
import math
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_desc_line(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return f"MW={mw:.1f} logP={logp:.2f} TPSA={tpsa:.1f} HBD={hbd} HBA={hba} RotB={rotb}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to final_selected.csv")
    ap.add_argument("--out", default="panel_5x5.pdf", help="Output image filename (pdf or png)")
    ap.add_argument("--tau", type=float, default=0.001, help="p-value cutoff (used if 'selected' col missing)")
    ap.add_argument("--n", type=int, default=50, help="How many molecules to plot")
    ap.add_argument("--per-row", type=int, default=5, help="Molecules per row")
    ap.add_argument("--sort", choices=["pval", "meta"], default="pval",
                    help="Sort selected molecules by p-value (asc) or meta_score (desc)")
    ap.add_argument("--cell", type=int, default=320, help="Pixel size of each cell (e.g., 260-400)")
    ap.add_argument(
        "--reference-csv",
        default="data/composed_modalities_with_rdkit.csv",
        help="CSV used to find reference molecule (default: main dataset).",
    )
    ap.add_argument(
        "--reference-source",
        default="calcitriol",
        help="Value in source column to render as the reference molecule.",
    )
    ap.add_argument(
        "--no-reference",
        action="store_true",
        help="Disable rendering the reference molecule row.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    smiles_col = pick_col(df, ["smiles", "SMILES", "canonical_smiles", "rdkit_smiles"])
    if smiles_col is None:
        raise ValueError(f"Could not find a SMILES column in {list(df.columns)}")

    pcol = pick_col(df, ["pval_unweighted", "pval", "p_value"])
    if pcol is None:
        raise ValueError(f"Could not find a p-value column in {list(df.columns)}")

    meta_col = pick_col(df, ["meta_score", "score", "cwra_score"])
    id_col = pick_col(df, ["molecule_id", "mol_id", "id", "name", "inchikey", "source"])

    # Filter to the shortlist
    if "selected" in df.columns:
        df_sel = df[df["selected"] == True].copy()
        # if selection mode was BH/top-k, this is fine; if not, fall back to tau as well
        if df_sel.empty:
            df_sel = df[df[pcol] <= args.tau].copy()
    else:
        df_sel = df[df[pcol] <= args.tau].copy()

    if df_sel.empty:
        raise RuntimeError("Selection is empty. Check tau or whether the CSV contains the expected p-values.")

    # Sorting
    if args.sort == "pval":
        df_sel = df_sel.sort_values(pcol, ascending=True)
    else:
        if meta_col is None:
            raise ValueError("Requested sort=meta but no meta_score column was found.")
        df_sel = df_sel.sort_values(meta_col, ascending=False)

    df_sel = df_sel.head(args.n).copy()

    mols = []
    legends = []
    bad = 0
    ref_added = False

    if not args.no_reference:
        ref_row = None
        if "source" in df.columns:
            cand = df[df["source"] == args.reference_source]
            if not cand.empty:
                ref_row = cand.iloc[0]
        if ref_row is None:
            try:
                ref_df = pd.read_csv(args.reference_csv)
                if "source" in ref_df.columns:
                    cand = ref_df[ref_df["source"] == args.reference_source]
                    if not cand.empty:
                        ref_row = cand.iloc[0]
            except Exception:
                ref_row = None

        if ref_row is not None:
            ref_smiles_col = pick_col(pd.DataFrame([ref_row]), ["smiles", "SMILES", "canonical_smiles", "rdkit_smiles"])
            if ref_smiles_col is not None:
                ref_smi = ref_row[ref_smiles_col]
                ref_mol = Chem.MolFromSmiles(ref_smi) if isinstance(ref_smi, str) else None
                if ref_mol is not None:
                    ref_desc = _build_desc_line(ref_mol)
                    mols.append(ref_mol)
                    legends.append(f"REF | {args.reference_source}\n{ref_desc}")
                    # Reserve the rest of the first row for visual separation.
                    for _ in range(max(0, args.per_row - 1)):
                        mols.append(None)
                        legends.append("")
                    ref_added = True

    for i, row in enumerate(df_sel.itertuples(index=False), start=1):
        s = getattr(row, smiles_col)
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            bad += 1
            continue

        # Legend text
        pid = None
        if id_col is not None:
            pid = getattr(row, id_col)
        pval = getattr(row, pcol)
        meta = getattr(row, meta_col) if meta_col is not None else None

        desc_line = _build_desc_line(m)

        if meta is None:
            leg = f"{i}" + (f" | {pid}" if pid is not None else "") + f"\np={pval:.4g}\n{desc_line}"
        else:
            leg = (
                f"{i}"
                + (f" | {pid}" if pid is not None else "")
                + f"\np={pval:.4g}  s={meta:.3f}\n{desc_line}"
            )

        mols.append(m)
        legends.append(leg)

    if not mols:
        raise RuntimeError("No valid molecules to draw (all SMILES failed RDKit parsing).")

    # Ensure we draw exactly per-row layout (pad with None if needed)
    n_needed = int(math.ceil(len(mols) / args.per_row) * args.per_row)
    while len(mols) < n_needed:
        mols.append(None)
        legends.append("")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=args.per_row,
        subImgSize=(args.cell, args.cell),
        legends=legends,
        useSVG=False,
    )

    img.save(args.out)

    print(f"Wrote: {args.out}")
    drawn_total = len([m for m in mols if m is not None])
    print(f"Selected rows drawn: {len(df_sel) - bad} / {len(df_sel)}")
    if ref_added:
        print(f"Reference shown in first row: {args.reference_source}")
    else:
        print(f"Reference not shown: {args.reference_source} not found or invalid SMILES")
    print(f"Total molecules drawn (including reference): {drawn_total}")
    if bad:
        print(f"WARNING: {bad} selected molecules had invalid SMILES and were skipped.")


if __name__ == "__main__":
    main()
