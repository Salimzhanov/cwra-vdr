#!/usr/bin/env python3
"""
Generate PDB structures for top 100 G-group compounds bound to VDR (1DB1).

This script:
1. Reads final_selected.csv and selects top 100 G-group compounds (G1, G2, G3)
   in row order, excluding initial_370 and calcitriol sources
2. Uses the Boltz-2 predicted VDR protein template (1990 ATOM records)
3. Generates 3D ligand conformers from SMILES using RDKit
4. Places ligands into the VDR binding pocket (centered at VDX position)
5. Writes properly formatted PDB files with protein + ligand

Usage:
    python scripts/generate_top100_g_pdbs.py
    python scripts/generate_top100_g_pdbs.py --output pdb/top100_g_complexes --top-n 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms
except ImportError:
    print("ERROR: RDKit is required. Install with: pip install rdkit")
    sys.exit(1)

# VDR 1DB1 binding pocket center - from the VDX (calcitriol) ligand in 1DB1 crystal
VDR_POCKET_CENTER = np.array([11.5, 23.0, 34.5])

# Path to reference protein template PDB (any existing complex with the 1990-atom VDR)
PROTEIN_TEMPLATE_SOURCES = [
    "pdb/flexible_complexes/calcitriol_unified/0_cmp_00000.pdb",
    "pdb/flexible_complexes/vdr_overlap_G2/0_row_0.pdb",
    "pdb/topbottom_structs/top25/G2_9.pdb",
]


def extract_protein_template(template_path: str) -> str:
    """
    Extract the protein ATOM records + TER from a reference complex PDB.
    Returns the protein portion as a properly formatted string.
    """
    lines = []
    with open(template_path, 'r') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n').rstrip('\r')
            if line.startswith('ATOM'):
                # Ensure exactly 80 characters per PDB spec
                line = line.rstrip()
                if len(line) < 80:
                    line = line.ljust(80)
                elif len(line) > 80:
                    line = line[:80]
                lines.append(line)
            elif line.startswith('TER'):
                # TER record: just "TER" is acceptable, but proper format is
                # TER   serial      resName chainID resSeq
                lines.append("TER")
                break  # Stop after TER - don't include HETATM from template
    return '\n'.join(lines)


def smiles_to_3d_mol(smiles: str, num_confs: int = 10, seed: int = 42) -> Optional[Chem.Mol]:
    """
    Convert SMILES to a 3D RDKit molecule with the best conformer.
    Uses ETKDGv3 with multiple conformers and MMFF optimization.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate multiple conformers (10 is sufficient for initial pocket placement)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.useSmallRingTorsions = True

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(conf_ids) == 0:
        # Fallback: try simpler embedding
        params2 = AllChem.ETKDGv3()
        params2.randomSeed = seed
        params2.useRandomCoords = True
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params2)
        if len(conf_ids) == 0:
            return None

    # Optimize with MMFF94 (limited iterations for speed; these are pre-docking poses)
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
        if results:
            # Pick lowest energy conformer
            energies = []
            for i, r in enumerate(results):
                if r[0] == 0:  # converged
                    energies.append((r[1], i))
                else:
                    energies.append((r[1] + 1e6, i))  # penalize non-converged
            if energies:
                energies.sort()
                best_idx = energies[0][1]
                # Remove all conformers except the best
                conf_ids_to_remove = [i for i in range(mol.GetNumConformers()) if i != best_idx]
                for i in sorted(conf_ids_to_remove, reverse=True):
                    mol.RemoveConformer(mol.GetConformer(i).GetId())
    except Exception:
        # If MMFF fails, just keep the first conformer
        while mol.GetNumConformers() > 1:
            mol.RemoveConformer(mol.GetConformer(1).GetId())

    return mol


def center_mol_at_pocket(mol: Chem.Mol, pocket_center: np.ndarray) -> Chem.Mol:
    """
    Translate the molecule so its centroid is at the binding pocket center.
    """
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    centroid = positions.mean(axis=0)
    translation = pocket_center - centroid

    for i in range(mol.GetNumAtoms()):
        x, y, z = positions[i]
        conf.SetAtomPosition(i, (x + translation[0],
                                  y + translation[1],
                                  z + translation[2]))
    return mol


def mol_to_hetatm_lines(mol: Chem.Mol, start_serial: int = 1991,
                         chain: str = "L", res_name: str = "UNL",
                         res_seq: int = 1) -> list[str]:
    """
    Convert an RDKit molecule with 3D coordinates to PDB HETATM lines.

    Produces properly formatted 80-character PDB lines following the standard:
      Columns  1- 6: Record type "HETATM"
      Columns  7-11: Atom serial number (right-justified)
      Column  12:    Space
      Columns 13-16: Atom name (left-justified if 1 char element, else starts col 13)
      Column  17:    Alternate location indicator (blank)
      Columns 18-20: Residue name (right-justified)
      Column  21:    Space
      Column  22:    Chain ID
      Columns 23-26: Residue sequence number (right-justified)
      Column  27:    Code for insertion of residues (blank)
      Columns 28-30: Spaces
      Columns 31-38: X coordinate (8.3f)
      Columns 39-46: Y coordinate (8.3f)
      Columns 47-54: Z coordinate (8.3f)
      Columns 55-60: Occupancy (6.2f)
      Columns 61-66: Temperature factor (6.2f)
      Columns 67-76: Spaces
      Columns 77-78: Element symbol (right-justified)
      Columns 79-80: Charge (blank)
    """
    conf = mol.GetConformer()
    lines = []
    serial = start_serial

    # Build unique atom names using element + counter
    element_counts = {}

    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        elem = atom.GetSymbol()

        # Skip explicit hydrogens for cleaner PDB (standard practice for HETATM)
        if elem == 'H':
            continue

        pos = conf.GetAtomPosition(i)

        # Generate unique atom name
        if elem not in element_counts:
            element_counts[elem] = 1
        else:
            element_counts[elem] += 1
        count = element_counts[elem]

        # Atom name formatting (columns 13-16, 4 chars):
        # - 1-char elements: space + element + number (e.g., " C1 ", " C12")
        # - 2-char elements: element + number (e.g., "CL1 ", "BR1 ")
        if len(elem) == 1:
            atom_name = f" {elem}{count:<2d}"  # e.g., " C1 ", " C12"
            if len(atom_name) > 4:
                atom_name = atom_name[:4]
        else:
            atom_name = f"{elem}{count:<2d}"  # e.g., "CL1 "
            if len(atom_name) > 4:
                atom_name = atom_name[:4]

        # Ensure atom_name is exactly 4 chars
        atom_name = f"{atom_name:<4s}"

        # Element symbol for columns 77-78 (right-justified in 2 chars)
        elem_col = f"{elem:>2s}"

        # Build the line character by character following PDB format
        # Cols 1-6: record type
        record = "HETATM"
        # Cols 7-11: serial (right-justified, 5 chars)
        serial_str = f"{serial:5d}"
        # Col 12: space
        # Cols 13-16: atom name (4 chars)
        # Col 17: altLoc (blank)
        # Cols 18-20: resName (3 chars, right-justified)
        res_name_str = f"{res_name:>3s}"
        # Col 21: space
        # Col 22: chainID
        # Cols 23-26: resSeq (right-justified, 4 chars)
        res_seq_str = f"{res_seq:4d}"
        # Col 27: iCode (blank)
        # Cols 28-30: spaces
        # Cols 31-38: X (8.3f)
        x_str = f"{pos.x:8.3f}"
        # Cols 39-46: Y (8.3f)
        y_str = f"{pos.y:8.3f}"
        # Cols 47-54: Z (8.3f)
        z_str = f"{pos.z:8.3f}"
        # Cols 55-60: occupancy (6.2f)
        occ_str = f"{1.00:6.2f}"
        # Cols 61-66: bfactor (6.2f)
        bfac_str = f"{0.00:6.2f}"
        # Cols 67-76: spaces (10 chars)
        # Cols 77-78: element
        # Cols 79-80: charge (blank)

        line = (f"{record}{serial_str} {atom_name} {res_name_str} "
                f"{chain}{res_seq_str}    "
                f"{x_str}{y_str}{z_str}{occ_str}{bfac_str}"
                f"          {elem_col}  ")

        # Verify exactly 80 chars
        assert len(line) == 80, f"Line length {len(line)} != 80: '{line}'"

        lines.append(line)
        serial += 1

    return lines


def validate_ligand_in_pocket(
    mol: Chem.Mol,
    protein_template: str,
    pocket_center: np.ndarray,
    max_pocket_distance: float = 15.0,
    min_protein_distance: float = 1.5,
    max_ligand_span: float = 30.0,
) -> dict:
    """
    Validate that the ligand is properly placed inside the protein binding pocket.

    Checks performed:
    1. Ligand centroid within max_pocket_distance of pocket_center
    2. No ligand atom closer than min_protein_distance to any protein atom (steric clash)
    3. At least one ligand atom within 6Å of a protein atom (actual interaction)
    4. Ligand size is reasonable (span not too large)

    Returns:
        dict with 'valid' (bool), 'warnings' (list[str]), 'metrics' (dict)
    """
    conf = mol.GetConformer()
    lig_positions = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == 'H':
            continue
        pos = conf.GetAtomPosition(i)
        lig_positions.append(np.array([pos.x, pos.y, pos.z]))

    if not lig_positions:
        return {'valid': False, 'warnings': ['No heavy atoms in ligand'], 'metrics': {}}

    lig_positions = np.array(lig_positions)
    lig_centroid = lig_positions.mean(axis=0)

    warnings = []
    metrics = {}

    # 1. Check centroid distance to pocket center
    centroid_dist = np.linalg.norm(lig_centroid - pocket_center)
    metrics['centroid_to_pocket'] = round(float(centroid_dist), 2)
    if centroid_dist > max_pocket_distance:
        warnings.append(f"Ligand centroid {centroid_dist:.1f}Å from pocket center (max {max_pocket_distance}Å)")

    # 2. Check ligand span (max inter-atom distance)
    if len(lig_positions) > 1:
        from scipy.spatial.distance import pdist
        lig_span = float(pdist(lig_positions).max())
    else:
        lig_span = 0.0
    metrics['ligand_span'] = round(lig_span, 2)
    if lig_span > max_ligand_span:
        warnings.append(f"Ligand span {lig_span:.1f}Å exceeds {max_ligand_span}Å — may extend outside pocket")

    # 3. Parse protein atom positions from template
    prot_positions = []
    for line in protein_template.split('\n'):
        if line.startswith('ATOM') and len(line) >= 54:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                prot_positions.append(np.array([x, y, z]))
            except ValueError:
                continue

    if prot_positions:
        prot_positions = np.array(prot_positions)

        # Compute all pairwise distances (ligand × protein)
        from scipy.spatial.distance import cdist
        dists = cdist(lig_positions, prot_positions)

        min_dist = float(dists.min())
        metrics['min_protein_distance'] = round(min_dist, 2)

        # Check for steric clashes
        n_clashes = int((dists < min_protein_distance).sum())
        metrics['n_steric_clashes'] = n_clashes
        if n_clashes > 0:
            warnings.append(f"{n_clashes} ligand-protein atom pairs closer than {min_protein_distance}Å (steric clash)")

        # Check for interaction (at least one contact within 6Å)
        n_contacts = int((dists < 6.0).any(axis=1).sum())
        metrics['n_contact_atoms'] = n_contacts
        if n_contacts == 0:
            warnings.append("No ligand atom within 6Å of any protein atom — ligand may be outside pocket")

        # Check how many ligand atoms are near pocket residues (within 8Å of pocket center)
        pocket_mask = np.linalg.norm(prot_positions - pocket_center, axis=1) < 12.0
        if pocket_mask.any():
            pocket_dists = cdist(lig_positions, prot_positions[pocket_mask])
            n_pocket_contacts = int((pocket_dists < 5.0).any(axis=1).sum())
            metrics['n_pocket_contact_atoms'] = n_pocket_contacts
            if n_pocket_contacts == 0:
                warnings.append("No ligand atom within 5Å of pocket residues — ligand misplaced")

    valid = len(warnings) == 0
    metrics['n_heavy_atoms'] = len(lig_positions)

    return {'valid': valid, 'warnings': warnings, 'metrics': metrics}


def generate_complex_pdb(
    protein_template: str,
    smiles: str,
    compound_id: str,
    pocket_center: np.ndarray = VDR_POCKET_CENTER,
    validate: bool = True,
) -> Optional[str]:
    """
    Generate a complete protein-ligand complex PDB file.

    Args:
        protein_template: Pre-formatted protein ATOM + TER string
        smiles: Ligand SMILES
        compound_id: Identifier for REMARK
        pocket_center: 3D coordinates to center the ligand
        validate: If True, validate ligand placement in pocket

    Returns:
        Complete PDB file content as string, or None on failure
    """
    # Generate 3D conformer
    mol = smiles_to_3d_mol(smiles)
    if mol is None:
        print(f"  WARNING: Failed to generate 3D for {compound_id}: {smiles[:60]}")
        return None

    # Center at binding pocket
    mol = center_mol_at_pocket(mol, pocket_center)

    # Validate ligand placement before writing PDB
    if validate:
        val_result = validate_ligand_in_pocket(mol, protein_template, pocket_center)
        if val_result['warnings']:
            for w in val_result['warnings']:
                print(f"  VALIDATION WARNING: {w}")
        metrics = val_result['metrics']
        print(f"  Validation: centroid_dist={metrics.get('centroid_to_pocket', '?')}Å, "
              f"min_prot_dist={metrics.get('min_protein_distance', '?')}Å, "
              f"clashes={metrics.get('n_steric_clashes', '?')}, "
              f"pocket_contacts={metrics.get('n_pocket_contact_atoms', '?')}")

    # Generate HETATM lines (serial starts after 1990 protein atoms + TER)
    hetatm_lines = mol_to_hetatm_lines(mol, start_serial=1991)
    if not hetatm_lines:
        print(f"  WARNING: No heavy atoms for {compound_id}")
        return None

    # Build complete PDB
    parts = []
    parts.append(f"REMARK   1 VDR-ligand complex: {compound_id}")
    parts.append(f"REMARK   2 SMILES: {smiles}")
    parts.append(f"REMARK   3 Protein: VDR LBD (1DB1), Boltz-2 predicted structure")
    parts.append(f"REMARK   4 Ligand placed at binding pocket center "
                 f"({pocket_center[0]:.1f}, {pocket_center[1]:.1f}, {pocket_center[2]:.1f})")
    parts.append(protein_template)
    parts.extend(hetatm_lines)
    parts.append("END")

    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate top-100 G-group PDB complexes bound to VDR 1DB1"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/final_selected.csv",
        help="Input CSV (final_selected.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="pdb/top100_g_complexes",
        help="Output directory for PDB files"
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=100,
        help="Number of top G-group compounds to process (default: 100)"
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Path to protein template PDB (auto-detected if not given)"
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Load and filter data
    # ---------------------------------------------------------------
    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")

    # Keep only G-group sources (exclude initial_370, calcitriol)
    g_df = df[df['source'].isin(['G1', 'G2', 'G3'])].copy()
    print(f"  G-group rows: {len(g_df)}")

    # Take top N by row order (file is already sorted by meta_score descending)
    # Preserve the original CSV index for filename numbering
    top_n = g_df.head(args.top_n)
    print(f"  Selected top {len(top_n)} compounds")
    print(f"  Source distribution: {top_n['source'].value_counts().to_dict()}")

    # ---------------------------------------------------------------
    # 2. Load protein template
    # ---------------------------------------------------------------
    template_path = args.template
    if template_path is None:
        for p in PROTEIN_TEMPLATE_SOURCES:
            if os.path.exists(p):
                template_path = p
                break
    if template_path is None or not os.path.exists(template_path):
        print("ERROR: No protein template PDB found. Provide --template path.")
        return 1

    print(f"\nProtein template: {template_path}")
    protein_template = extract_protein_template(template_path)
    atom_count = protein_template.count('\nATOM')
    print(f"  Protein ATOM records: {atom_count + 1}")  # +1 for first line

    # ---------------------------------------------------------------
    # 3. Create output directory
    # ---------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old PDB files from output directory
    old_pdbs = list(output_dir.glob('*.pdb'))
    if old_pdbs:
        print(f"\nRemoving {len(old_pdbs)} old PDB files from {output_dir}")
        for p in old_pdbs:
            p.unlink()
    old_csv = output_dir / 'manifest.csv'
    if old_csv.exists():
        old_csv.unlink()

    print(f"\nOutput directory: {output_dir}")

    # ---------------------------------------------------------------
    # 4. Generate PDB files
    # ---------------------------------------------------------------
    manifest_rows = []
    success_count = 0
    fail_count = 0

    for seq_num, (csv_idx, row) in enumerate(top_n.iterrows(), 1):
        csv_row = csv_idx + 1  # 1-based row number in final_selected.csv
        source = row['source']
        smiles = row['smiles']
        meta_score = row['meta_score']

        compound_id = f"{source}_{csv_row}"
        filename = f"{compound_id}.pdb"

        print(f"\n[{seq_num:3d}/{len(top_n)}] {compound_id}  "
              f"csv_row={csv_row}  meta={meta_score:.4f}  smi={smiles[:50]}...")

        pdb_content = generate_complex_pdb(
            protein_template=protein_template,
            smiles=smiles,
            compound_id=compound_id,
            pocket_center=VDR_POCKET_CENTER,
        )

        if pdb_content is not None:
            out_path = output_dir / filename
            with open(out_path, 'w', newline='\n') as f:
                f.write(pdb_content)
                f.write('\n')
            success_count += 1
            status = "success"
            print(f"  -> Saved {out_path}")
        else:
            status = "failed"
            fail_count += 1
            print(f"  -> FAILED")

        manifest_rows.append({
            'seq': seq_num,
            'csv_row': csv_row,
            'source': source,
            'smiles': smiles,
            'meta_score': meta_score,
            'pdb_file': filename if status == "success" else "",
            'status': status,
        })

    # ---------------------------------------------------------------
    # 5. Save manifest
    # ---------------------------------------------------------------
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total compounds:  {len(top_n)}")
    print(f"  Success:          {success_count}")
    print(f"  Failed:           {fail_count}")
    print(f"  Output directory: {output_dir}")
    print(f"  Manifest:         {manifest_path}")

    # Per-source breakdown
    for src in ['G2', 'G3', 'G1']:
        src_rows = manifest_df[manifest_df['source'] == src]
        ok = (src_rows['status'] == 'success').sum()
        print(f"  {src}: {ok}/{len(src_rows)} success")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
