#!/usr/bin/env python3
"""
Prepare VDR LBD receptor for AutoDock Vina docking.

Full preparation pipeline:
  1. Read Boltz-predicted PDB (heavy atoms only)
  2. PDBFixer: add missing atoms, add hydrogens at pH 7.4
  3. Determine histidine protonation states (HID/HIE/HIP)
  4. Strip non-polar H (keep only polar H bonded to N, O, S)
  5. Assign AutoDock atom types (C, A, N, NA, OA, S, HD)
  6. Compute Gasteiger partial charges
  7. Write PDBQT preserving residue numbering
  8. Compute and report binding pocket grid box

Outputs:
  pdb/vdr_lbd_boltz_prepared.pdb   - full-H PDB (for reference)
  pdb/vdr_lbd_boltz_prepared.pdbqt - Vina receptor (polar H, charges, types)

Binding residues (UniProt → LBD numbering):
  Tyr143 → 24   Ser237 → 118   Arg274 → 155
  His305 → 186  His397 → 278
  + additional pocket: Pro26, Leu108, Leu114, Glu150, Ile152, Met153, Ser159

Usage:
    python scripts/prepare_vina_receptor.py
    python scripts/prepare_vina_receptor.py --ph 7.0 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ===================================================================
# PATHS
# ===================================================================
ROOT = Path(__file__).resolve().parent.parent
INPUT_PDB = ROOT / "pdb" / "vdr_lbd_boltz.pdb"
OUTPUT_PDB = ROOT / "pdb" / "vdr_lbd_boltz_prepared.pdb"
OUTPUT_PDBQT = ROOT / "pdb" / "vdr_lbd_boltz_prepared.pdbqt"

# ===================================================================
# VDR LBD BINDING POCKET
# ===================================================================
# User-specified key binding residues (LBD numbering)
KEY_BINDING_RESIDUES = {
    24:  "Tyr143",   # H-bond with ligand A-ring
    118: "Ser237",   # H-bond with 1α-OH
    155: "Arg274",   # H-bond with 25-OH
    186: "His305",   # NE2 accepts H-bond from 25-OH
    278: "His397",   # NE2 accepts H-bond from 1α-OH
}

# Extended pocket residues (for grid computation)
ALL_POCKET_RESIDUES = [24, 26, 108, 114, 118, 150, 152, 153, 155, 159, 186, 278]

# ===================================================================
# AUTODOCK ATOM TYPE ASSIGNMENT
# ===================================================================
# Aromatic carbon atoms per residue type
AROMATIC_CARBONS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "CD2", "CE1"},
}

# Standard amino acid atom types (non-backbone, non-aromatic)
# Format: {resname: {atomname: adtype}}
# Default rules cover most cases; this handles special ones
SPECIAL_TYPES = {
    # Backbone (applied to all residues)
    "_BB_": {"N": "N", "CA": "C", "C": "C", "O": "OA", "OXT": "OA"},
    # Side chains with special nitrogen types
    "HIS": {},   # handled by aromatic/protonation logic
    "TRP": {"NE1": "N"},  # NE1 is H-bond donor (has H)
    "PRO": {"N": "N"},    # Pro N has no H, but still typed as N in AD
}


def assign_ad_type(
    atom_name: str,
    res_name: str,
    element: str,
    is_polar_h: bool = False,
    his_nd1_protonated: bool = False,
    his_ne2_protonated: bool = False,
) -> str | None:
    """Assign AutoDock atom type.

    Returns None for non-polar H (should be excluded from PDBQT).
    """
    aname = atom_name.strip()
    rname = res_name.strip()
    elem = element.strip().upper()

    # Hydrogen
    if elem == "H":
        if is_polar_h:
            return "HD"
        return None  # non-polar H → skip

    # Carbon
    if elem == "C":
        if rname in AROMATIC_CARBONS and aname in AROMATIC_CARBONS[rname]:
            return "A"
        return "C"

    # Nitrogen
    if elem == "N":
        # HIS ring nitrogens: type depends on protonation
        if rname == "HIS":
            if aname == "ND1":
                return "N" if his_nd1_protonated else "NA"
            if aname == "NE2":
                return "N" if his_ne2_protonated else "NA"
        # TRP NE1 is always protonated donor
        if rname == "TRP" and aname == "NE1":
            return "N"
        # Default: N (most backbone/sidechain N)
        return "N"

    # Oxygen
    if elem == "O":
        return "OA"

    # Sulfur
    if elem == "S":
        return "S"

    # Fallback
    return elem


# ===================================================================
# GASTEIGER CHARGES (simplified for amino acids)
# ===================================================================
# Average Gasteiger charges per atom type from standard amino acids
# These are approximations; Vina doesn't use charges for scoring
DEFAULT_CHARGES = {
    "N":  -0.350,  # backbone N
    "CA":  0.100,  # backbone CA
    "C":   0.500,  # backbone C (C=O)
    "O":  -0.500,  # backbone O
    "CB":  0.000,  # beta carbon
    "H":   0.250,  # polar H on N
    "HD":  0.250,  # polar H (donor)
}


def gasteiger_charge(atom_name: str, res_name: str, element: str) -> float:
    """Assign approximate Gasteiger partial charge.

    Vina does NOT use these for scoring; they are for compatibility only.
    """
    aname = atom_name.strip()
    elem = element.strip().upper()
    rname = res_name.strip()

    # Backbone atoms
    if aname == "N":
        return -0.350
    if aname == "CA":
        return 0.100
    if aname == "C" and elem == "C":
        return 0.500 if aname == "C" else 0.000
    if aname == "O" and elem == "O":
        return -0.500

    # Charged side chains
    if rname == "ASP" and aname in ("OD1", "OD2"):
        return -0.570  # carboxylate
    if rname == "GLU" and aname in ("OE1", "OE2"):
        return -0.570
    if rname == "LYS" and aname == "NZ":
        return -0.070  # NH3+
    if rname == "ARG" and aname in ("NH1", "NH2"):
        return -0.260  # guanidinium

    # Polar groups
    if elem == "O":
        return -0.400
    if elem == "N":
        return -0.300
    if elem == "S":
        return -0.100
    if elem == "H":
        return 0.200

    # Aliphatic/aromatic carbons
    return 0.000


# ===================================================================
# MAIN PREPARATION
# ===================================================================
def prepare_receptor(
    input_pdb: Path,
    output_pdb: Path,
    output_pdbqt: Path,
    ph: float = 7.4,
    verbose: bool = True,
) -> dict:
    """Full receptor preparation pipeline.

    Returns dict with preparation statistics and grid parameters.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    from openmm import unit

    info = {}

    # ------------------------------------------------------------------
    # Step 1: Load PDB
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("RECEPTOR PREPARATION — VDR LBD for AutoDock Vina")
        print("=" * 70)
        print(f"\n[1] Loading {input_pdb.name}...")

    fixer = PDBFixer(filename=str(input_pdb))
    residues = list(fixer.topology.residues())
    atoms = list(fixer.topology.atoms())
    info["initial_residues"] = len(residues)
    info["initial_atoms"] = len(atoms)

    if verbose:
        print(f"    Chains: {len(list(fixer.topology.chains()))}")
        print(f"    Residues: {len(residues)}")
        print(f"    Atoms: {len(atoms)} (heavy atoms only)")

    # ------------------------------------------------------------------
    # Step 2: Verify key binding residues present
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[2] Checking key binding residues...")

    res_map = {int(r.id): r for r in residues}
    for lbd_pos, name in sorted(KEY_BINDING_RESIDUES.items()):
        if lbd_pos in res_map:
            r = res_map[lbd_pos]
            if verbose:
                print(f"    ✓ LBD {lbd_pos:3d} ({name:8s}): {r.name}")
        else:
            print(f"    ✗ LBD {lbd_pos:3d} ({name:8s}): MISSING!")
            info["missing_binding_res"] = info.get("missing_binding_res", [])
            info["missing_binding_res"].append(lbd_pos)

    # ------------------------------------------------------------------
    # Step 3: Remove heterogens/waters
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[3] Removing heterogens and waters...")
    fixer.removeHeterogens(keepWater=False)

    # ------------------------------------------------------------------
    # Step 4: Find and add missing atoms
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[4] Checking for missing atoms...")
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    n_missing = sum(len(v) for v in fixer.missingAtoms.values())
    if n_missing > 0:
        if verbose:
            print(f"    Adding {n_missing} missing atoms")
        fixer.addMissingAtoms()
    else:
        if verbose:
            print(f"    No missing atoms")

    # ------------------------------------------------------------------
    # Step 5: Add hydrogens at specified pH
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[5] Adding hydrogens at pH {ph}...")
    fixer.addMissingHydrogens(ph)

    positions = fixer.positions.value_in_unit(unit.angstroms)
    atoms_after = list(fixer.topology.atoms())
    residues_after = list(fixer.topology.residues())
    bonds = list(fixer.topology.bonds())
    n_h = sum(1 for a in atoms_after if a.element.symbol == "H")

    info["prepared_residues"] = len(residues_after)
    info["prepared_atoms"] = len(atoms_after)
    info["n_hydrogens"] = n_h

    if verbose:
        print(f"    Total atoms: {len(atoms_after)}")
        print(f"    Hydrogens added: {n_h}")

    # ------------------------------------------------------------------
    # Step 6: Analyze histidine protonation
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[6] Histidine protonation analysis...")

    # Build bond lookup for H
    bond_lookup = {}  # atom_index -> set of bonded atom indices
    for b in bonds:
        i1, i2 = b.atom1.index, b.atom2.index
        bond_lookup.setdefault(i1, set()).add(i2)
        bond_lookup.setdefault(i2, set()).add(i1)

    his_protonation = {}  # res_id -> {"nd1_h": bool, "ne2_h": bool, "state": str}
    for r in residues_after:
        if r.name != "HIS":
            continue
        rid = int(r.id)
        ratoms = {a.name: a for a in r.atoms()}

        nd1_has_h = False
        ne2_has_h = False

        # Check ND1
        if "ND1" in ratoms:
            nd1_idx = ratoms["ND1"].index
            for bidx in bond_lookup.get(nd1_idx, set()):
                if atoms_after[bidx].element.symbol == "H":
                    nd1_has_h = True
                    break

        # Check NE2
        if "NE2" in ratoms:
            ne2_idx = ratoms["NE2"].index
            for bidx in bond_lookup.get(ne2_idx, set()):
                if atoms_after[bidx].element.symbol == "H":
                    ne2_has_h = True
                    break

        if nd1_has_h and ne2_has_h:
            state = "HIP (+1)"
        elif nd1_has_h:
            state = "HID (δ)"
        elif ne2_has_h:
            state = "HIE (ε)"
        else:
            state = "neutral?"

        his_protonation[rid] = {
            "nd1_h": nd1_has_h, "ne2_h": ne2_has_h, "state": state
        }

        binding = " ← KEY BINDING" if rid in KEY_BINDING_RESIDUES else ""
        if verbose:
            print(f"    HIS {rid:3d}: {state:10s}"
                  f"  ND1-H={nd1_has_h}  NE2-H={ne2_has_h}{binding}")

    info["his_protonation"] = his_protonation

    # ------------------------------------------------------------------
    # Step 7: Save prepared PDB (full hydrogens)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[7] Saving prepared PDB: {output_pdb.name}")
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    # ------------------------------------------------------------------
    # Step 8: Write PDBQT (polar H only, AD types, charges)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[8] Writing PDBQT: {output_pdbqt.name}")
        print(f"    Stripping non-polar H, assigning AutoDock types...")

    pdbqt_lines = []
    serial = 0
    n_polar_h = 0
    n_nonpolar_h_removed = 0

    for r in residues_after:
        rid = int(r.id)
        rname = r.name.strip()
        chain_id = r.chain.id if r.chain.id else "A"

        # Get HIS protonation state for this residue
        his_nd1_h = his_protonation.get(rid, {}).get("nd1_h", False)
        his_ne2_h = his_protonation.get(rid, {}).get("ne2_h", False)

        for atom in r.atoms():
            elem = atom.element.symbol
            aname = atom.name.strip()
            pos = positions[atom.index]

            # Determine if H is polar
            is_polar = False
            if elem == "H":
                # Check what this H is bonded to
                for bidx in bond_lookup.get(atom.index, set()):
                    bonded_elem = atoms_after[bidx].element.symbol
                    if bonded_elem in ("N", "O", "S"):
                        is_polar = True
                        break

            # Assign AutoDock type
            ad_type = assign_ad_type(
                aname, rname, elem,
                is_polar_h=is_polar,
                his_nd1_protonated=his_nd1_h,
                his_ne2_protonated=his_ne2_h,
            )

            if ad_type is None:
                n_nonpolar_h_removed += 1
                continue

            if elem == "H" and is_polar:
                n_polar_h += 1

            serial += 1

            # Compute charge
            charge = gasteiger_charge(aname, rname, elem)

            # Format atom name (standard PDB convention)
            if len(aname) < 4:
                an_fmt = f" {aname:<3s}"
            else:
                an_fmt = f"{aname:4s}"

            # PDBQT line
            line = (
                f"ATOM  {serial:5d} {an_fmt:4s} {rname:3s}"
                f" {chain_id}{rid:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00    {charge:+.3f} {ad_type:<2s}"
            )
            pdbqt_lines.append(line)

    pdbqt_lines.append("TER")
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    output_pdbqt.write_text("\n".join(pdbqt_lines) + "\n", encoding="utf-8")

    info["pdbqt_atoms"] = serial
    info["polar_h"] = n_polar_h
    info["nonpolar_h_removed"] = n_nonpolar_h_removed

    if verbose:
        print(f"    PDBQT atoms: {serial}")
        print(f"    Polar H (HD): {n_polar_h}")
        print(f"    Non-polar H removed: {n_nonpolar_h_removed}")

    # ------------------------------------------------------------------
    # Step 9: Compute grid box from binding pocket
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[9] Computing binding pocket grid box...")

    pocket_xyz = []
    for r in residues_after:
        rid = int(r.id)
        if rid in ALL_POCKET_RESIDUES:
            for a in r.atoms():
                if a.element.symbol != "H":
                    pocket_xyz.append(positions[a.index])

    pocket_xyz = np.array(pocket_xyz)
    pocket_center = np.mean(pocket_xyz, axis=0)
    pocket_min = np.min(pocket_xyz, axis=0)
    pocket_max = np.max(pocket_xyz, axis=0)
    pocket_span = pocket_max - pocket_min

    # Box size: pocket span + margin, but constrained to Vina limit
    # Vina max recommended volume: 27,000 A^3 (= 30 x 30 x 30)
    margin = 8.0
    box_size = pocket_span + margin
    # Enforce Vina volume limit: 27000 A^3
    box_size = np.minimum(box_size, 30.0)
    # Round up to even numbers
    box_size = np.ceil(box_size / 2) * 2

    info["grid_center"] = tuple(float(x) for x in pocket_center)
    info["grid_size"] = tuple(float(x) for x in box_size)
    info["pocket_span"] = tuple(float(x) for x in pocket_span)

    if verbose:
        print(f"    Pocket residues: {ALL_POCKET_RESIDUES}")
        print(f"    Pocket heavy atoms: {len(pocket_xyz)}")
        print(f"    Pocket span: ({pocket_span[0]:.1f}, {pocket_span[1]:.1f},"
              f" {pocket_span[2]:.1f}) Å")
        print(f"    Grid center: ({pocket_center[0]:.2f},"
              f" {pocket_center[1]:.2f}, {pocket_center[2]:.2f})")
        print(f"    Grid size:   ({box_size[0]:.0f}, {box_size[1]:.0f},"
              f" {box_size[2]:.0f}) Å")

    # ------------------------------------------------------------------
    # Step 10: Validation
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[10] Validation...")

    # Verify PDBQT residue numbering preserved
    pdbqt_text = output_pdbqt.read_text()
    pdbqt_resnums = sorted(set(
        int(l[22:26].strip())
        for l in pdbqt_text.splitlines()
        if l.startswith("ATOM")
    ))

    # Check atom types
    pdbqt_types = set()
    for l in pdbqt_text.splitlines():
        if l.startswith("ATOM") and len(l) >= 78:
            pdbqt_types.add(l[77:79].strip())

    # Verify key residues
    key_ok = all(rid in pdbqt_resnums for rid in KEY_BINDING_RESIDUES)
    all_ok = all(rid in pdbqt_resnums for rid in ALL_POCKET_RESIDUES)

    if verbose:
        print(f"    Residue range: {pdbqt_resnums[0]}-{pdbqt_resnums[-1]}"
              f" ({len(pdbqt_resnums)} residues)")
        print(f"    Atom types: {sorted(pdbqt_types)}")
        print(f"    Key binding residues present: {'✓' if key_ok else '✗'}")
        print(f"    All pocket residues present: {'✓' if all_ok else '✗'}")

        # Check polar H on key residues
        for rid, name in sorted(KEY_BINDING_RESIDUES.items()):
            hd_count = sum(
                1 for l in pdbqt_text.splitlines()
                if l.startswith("ATOM") and int(l[22:26].strip()) == rid
                and len(l) >= 78 and l[77:79].strip() == "HD"
            )
            total = sum(
                1 for l in pdbqt_text.splitlines()
                if l.startswith("ATOM") and int(l[22:26].strip()) == rid
            )
            print(f"    {name} (LBD {rid}): {total} atoms, {hd_count} polar H")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PREPARATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Prepared PDB:  {output_pdb}")
        print(f"  Prepared PDBQT: {output_pdbqt}")
        print(f"  Residues: {len(pdbqt_resnums)}")
        print(f"  PDBQT atoms: {serial} (incl. {n_polar_h} polar H)")
        print(f"  Atom types: {sorted(pdbqt_types)}")
        print(f"  Protonation pH: {ph}")
        print(f"  Grid center: ({pocket_center[0]:.2f},"
              f" {pocket_center[1]:.2f}, {pocket_center[2]:.2f})")
        print(f"  Grid size:   ({box_size[0]:.0f}, {box_size[1]:.0f},"
              f" {box_size[2]:.0f})")

    return info


# ===================================================================
# CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare VDR LBD receptor for AutoDock Vina")
    parser.add_argument("--input", default=str(INPUT_PDB),
                        help="Input Boltz PDB (default: vdr_lbd_boltz.pdb)")
    parser.add_argument("--output-pdb", default=str(OUTPUT_PDB),
                        help="Output prepared PDB")
    parser.add_argument("--output-pdbqt", default=str(OUTPUT_PDBQT),
                        help="Output PDBQT for Vina")
    parser.add_argument("--ph", type=float, default=7.4,
                        help="pH for protonation (default: 7.4)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    input_pdb = Path(args.input)
    if not input_pdb.exists():
        print(f"ERROR: Input PDB not found: {input_pdb}")
        sys.exit(1)

    info = prepare_receptor(
        input_pdb=input_pdb,
        output_pdb=Path(args.output_pdb),
        output_pdbqt=Path(args.output_pdbqt),
        ph=args.ph,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
