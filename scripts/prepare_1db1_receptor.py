#!/usr/bin/env python3
"""
Prepare 1DB1 crystal receptor for AutoDock Vina docking.

Uses the experimental VDR LBD crystal structure (PDB: 1DB1, Rochel et al. 2000)
instead of a Boltz-predicted model.

1DB1 details:
  - Chain A: res 120-423 (250 residues, 2 disordered gaps: 165-215, 375-377)
  - Co-crystal ligand: VDX (1α,25-dihydroxyvitamin D3 / calcitriol)
  - Resolution: 1.80 Å

Pipeline:
  1. Extract chain A protein atoms (strip ligand VDX + waters HOH)
  2. PDBFixer: add missing heavy atoms within resolved residues
  3. Add hydrogens at pH 7.4 (OpenMM)
  4. Histidine protonation: HID for His305/His397 (NE2 = acceptor for calcitriol OH)
  5. Strip non-polar H (keep only HD bonded to N/O/S)
  6. Assign AutoDock atom types and Gasteiger charges
  7. Write PDBQT for Vina
  8. Compute binding pocket grid box from crystal coordinates

Outputs:
  pdb/vdr_1db1_prepared.pdb    - full-H PDB (for complex assembly)
  pdb/vdr_1db1_prepared.pdbqt  - Vina receptor (polar H, charges, types)

Usage:
    python scripts/prepare_1db1_receptor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT_PDB  = ROOT / "pdb" / "1DB1_original.pdb"
OUTPUT_PDB  = ROOT / "pdb" / "vdr_1db1_prepared.pdb"
OUTPUT_PDBQT = ROOT / "pdb" / "vdr_1db1_prepared.pdbqt"

# 1DB1 binding-pocket residues (original PDB numbering, chain A)
KEY_BINDING_RESIDUES = {
    143: "Tyr143",   # H-bond to 3β-OH
    145: "Pro145",
    227: "Leu227",
    233: "Leu233",
    237: "Ser237",   # H-bond to 1α-OH
    269: "Glu269",
    271: "Ile271",
    272: "Met272",
    274: "Arg274",   # H-bond to 1α-OH
    278: "Ser278",   # H-bond to 3β-OH
    305: "His305",   # NE2 H-bond to 25-OH
    397: "His397",   # NE2 H-bond to 25-OH
}

# Aromatic carbons per residue type (for AutoDock type assignment)
AROMATIC_CARBONS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "CD2", "CE1"},
}

# ── Flexible-residue docking (Vina --flex) ─────────────────────────
# 6 pharmacophore-anchor residues whose sidechains are set flexible
# so induced-fit effects are captured.  All form direct H-bonds with
# calcitriol in the 1DB1 crystal (Rochel et al. 2000).
FLEX_RESIDUES = [143, 237, 274, 278, 305, 397]

OUTPUT_RIGID = ROOT / "pdb" / "vdr_1db1_rigid.pdbqt"
OUTPUT_FLEX  = ROOT / "pdb" / "vdr_1db1_flex.pdbqt"

# Sidechain torsion trees for flex-PDBQT BRANCH generation.
# Each node: {"bond": (parent, child), "heavy": [atoms], "children": [...]}
# Polar-H atoms are auto-assigned to their nearest heavy atom.
FLEX_SIDECHAIN_TREES: dict[str, dict] = {
    "SER": {
        "bond": ("CA", "CB"),
        "heavy": ["CB", "OG"],
        "children": [],
    },
    "TYR": {
        "bond": ("CA", "CB"),
        "heavy": ["CB"],
        "children": [{
            "bond": ("CB", "CG"),
            "heavy": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            "children": [{
                "bond": ("CZ", "OH"),
                "heavy": ["OH"],
                "children": [],
            }],
        }],
    },
    "ARG": {
        "bond": ("CA", "CB"),
        "heavy": ["CB"],
        "children": [{
            "bond": ("CB", "CG"),
            "heavy": ["CG"],
            "children": [{
                "bond": ("CG", "CD"),
                "heavy": ["CD"],
                "children": [{
                    "bond": ("CD", "NE"),
                    "heavy": ["NE", "CZ", "NH1", "NH2"],
                    "children": [],
                }],
            }],
        }],
    },
    "HIS": {
        "bond": ("CA", "CB"),
        "heavy": ["CB"],
        "children": [{
            "bond": ("CB", "CG"),
            "heavy": ["CG", "ND1", "CD2", "CE1", "NE2"],
            "children": [],
        }],
    },
}


def extract_chain_a_protein(pdb_path: Path) -> list[str]:
    """Extract chain A ATOM lines only (no HETATM, no waters, no ligand)."""
    lines = pdb_path.read_text().splitlines()
    protein = []
    for line in lines:
        if line.startswith("ATOM") and line[21] == "A":
            protein.append(line)
    print(f"  Extracted {len(protein)} ATOM lines from chain A")
    return protein


def build_residue_map(pdb_lines: list[str]) -> dict[int, int]:
    """Build mapping from sequential 1..N numbering back to original 1DB1 numbers.

    1DB1 chain A: 120-164, 216-374, 378-423 (250 residues, 2 gaps)
    PDBFixer renumbers to 1-250 sequentially.
    """
    # Get original residue numbers in order
    seen = set()
    orig_order = []
    for line in pdb_lines:
        try:
            rn = int(line[22:26].strip())
            if rn not in seen:
                seen.add(rn)
                orig_order.append(rn)
        except (ValueError, IndexError):
            continue
    # new_num (1-based) -> original_num
    return {i + 1: orig for i, orig in enumerate(orig_order)}


def add_hydrogens_pdbfixer(pdb_lines: list[str], ph: float = 7.4) -> str:
    """Use PDBFixer + OpenMM to add missing atoms and hydrogens.

    Restores original 1DB1 residue numbering after PDBFixer renumbers.
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        print("ERROR: pdbfixer/openmm required. Install: pip install pdbfixer")
        sys.exit(1)

    import io, tempfile

    # Build renumbering map BEFORE PDBFixer modifies anything
    res_map = build_residue_map(pdb_lines)
    print(f"  Residue map: {len(res_map)} residues "
          f"(new 1-{max(res_map.keys())} -> orig {min(res_map.values())}-{max(res_map.values())})")

    # Write temp PDB
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    for line in pdb_lines:
        tmp.write(line + "\n")
    tmp.write("END\n")
    tmp.close()

    fixer = PDBFixer(filename=tmp.name)

    # Find and add missing atoms in resolved residues (not missing residues)
    fixer.findMissingResidues()
    # Don't add missing residues (gaps are expected in 1DB1)
    fixer.missingResidues = {}

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Add hydrogens at specified pH
    fixer.addMissingHydrogens(ph)

    # Write to string
    out = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, out)
    pdb_text = out.getvalue()

    Path(tmp.name).unlink(missing_ok=True)

    # Restore original residue numbering
    restored_lines = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                new_rn = int(line[22:26].strip())
                orig_rn = res_map.get(new_rn, new_rn)
                # Replace residue number, preserve chain A
                line = line[:21] + "A" + f"{orig_rn:4d}" + line[26:]
            except (ValueError, IndexError):
                pass
        restored_lines.append(line)

    restored_text = "\n".join(restored_lines)
    # Verify
    n_restored = sum(1 for l in restored_lines
                     if l.startswith("ATOM") and 300 <= int(l[22:26].strip()) <= 400)
    print(f"  Restored original numbering (res 300-400 atoms: {n_restored})")
    return restored_text


def is_polar_hydrogen(atom_name: str, bonded_to_element: str) -> bool:
    """Check if hydrogen is polar (bonded to N, O, or S)."""
    return bonded_to_element in ("N", "O", "S")


def determine_h_bonding(pdb_text: str) -> dict[str, dict]:
    """Determine which hydrogens are polar by proximity to N/O/S."""
    lines = pdb_text.splitlines()
    atoms = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                name = line[12:16].strip()
                elem = line[76:78].strip() if len(line) >= 78 else name[0]
                serial = int(line[6:11].strip())
                resnum = int(line[22:26].strip())
                atoms.append({"serial": serial, "name": name, "elem": elem,
                              "xyz": np.array([x, y, z]), "resnum": resnum,
                              "line": line})
            except (ValueError, IndexError):
                continue

    # For each H atom, find nearest heavy atom
    h_info = {}
    heavy = [a for a in atoms if a["elem"] != "H"]
    for a in atoms:
        if a["elem"] == "H" and heavy:
            dists = [np.linalg.norm(a["xyz"] - h["xyz"]) for h in heavy]
            nearest_idx = np.argmin(dists)
            nearest = heavy[nearest_idx]
            polar = nearest["elem"] in ("N", "O", "S")
            h_info[a["serial"]] = {"polar": polar, "bonded_elem": nearest["elem"]}

    return h_info


def assign_ad_type(atom_name: str, res_name: str, element: str,
                   is_polar_h: bool = False,
                   his_nd1_protonated: bool = False,
                   his_ne2_protonated: bool = False) -> str | None:
    """Assign AutoDock atom type. Returns None for non-polar H."""
    aname = atom_name.strip()
    rname = res_name.strip()
    elem = element.strip().upper()

    if elem == "H":
        return "HD" if is_polar_h else None

    if elem == "C":
        if rname in AROMATIC_CARBONS and aname in AROMATIC_CARBONS[rname]:
            return "A"
        return "C"

    if elem == "N":
        if rname == "HIS":
            if aname == "ND1":
                return "N" if his_nd1_protonated else "NA"
            if aname == "NE2":
                return "N" if his_ne2_protonated else "NA"
        if aname == "NE1" and rname == "TRP":
            return "N"
        if aname == "N":
            return "N"
        # Amide NH2 (ASN, GLN) and guanidinium (ARG) are donors
        if aname in ("NZ", "NH1", "NH2", "NE", "ND2", "NE2"):
            return "N"
        return "NA"  # acceptor nitrogen

    if elem == "O":
        return "OA"

    if elem == "S":
        return "S"

    if elem in ("F", "P", "I"):
        return elem
    if elem in ("CL", "BR"):
        return elem.capitalize()

    return "C"


def gasteiger_charge(atom_name: str, res_name: str, element: str) -> float:
    """Simplified Gasteiger partial charge assignment."""
    elem = element.strip().upper()
    aname = atom_name.strip()

    charges = {
        "N": -0.350, "O": -0.400, "OA": -0.350, "S": -0.200,
        "C": 0.100, "CA": 0.100, "H": 0.200, "HD": 0.250,
    }

    # Special cases
    if aname == "N" and elem == "N":
        return -0.350
    if aname in ("O", "OXT") and elem == "O":
        return -0.550
    if aname in ("OG", "OG1", "OH"):
        return -0.385
    if aname == "NZ":
        return 0.310
    if aname in ("NH1", "NH2"):
        return -0.235
    if aname in ("ND1", "NE2") and res_name.strip() == "HIS":
        return -0.360
    if elem == "H":
        return 0.211
    if elem == "C":
        return 0.066
    if elem == "N":
        return -0.316
    if elem == "O":
        return -0.389
    if elem == "S":
        return -0.185

    return 0.000


def write_pdbqt(pdb_text: str, output_pdbqt: Path, output_pdb: Path):
    """Convert PDB to PDBQT with proper atom types, charges, polar H only."""
    h_info = determine_h_bonding(pdb_text)

    lines = pdb_text.splitlines()
    pdbqt_lines = []
    pdb_full_lines = []  # full-H PDB for complex assembly
    n_polar_h = 0
    n_nonpolar_h = 0
    n_heavy = 0

    # 1DB1 His305 and His397: both HID (ND1 protonated, NE2 unprotonated = acceptor)
    # This matches the crystal H-bond network: NE2 accepts from calcitriol 25-OH

    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        try:
            serial = int(line[6:11].strip())
            aname = line[12:16].strip()
            resname = line[17:20].strip()
            resnum = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = line[76:78].strip() if len(line) >= 78 else aname[0]
        except (ValueError, IndexError):
            continue

        # Save full-H PDB
        pdb_full_lines.append(line)

        # Histidine protonation: HID for His305, His397
        # HID = ND1 has H (protonated), NE2 has lone pair (acceptor)
        his_nd1_prot = True   # δ-tautomer: ND1 protonated
        his_ne2_prot = False  # NE2 unprotonated = H-bond acceptor

        polar = False
        if elem == "H":
            info = h_info.get(serial, {})
            polar = info.get("polar", False)

        ad_type = assign_ad_type(aname, resname, elem,
                                 is_polar_h=polar,
                                 his_nd1_protonated=his_nd1_prot,
                                 his_ne2_protonated=his_ne2_prot)

        if ad_type is None:
            n_nonpolar_h += 1
            continue  # skip non-polar H

        if elem == "H":
            n_polar_h += 1
        else:
            n_heavy += 1

        charge = gasteiger_charge(aname, resname, elem)

        # Format PDBQT line (strict column format: charge cols 71-76, type cols 78-79)
        an = f" {aname:<3s}" if len(aname) < 4 else aname[:4]
        pdbqt_line = (
            f"ATOM  {serial:5d} {an:4s} {resname:>3s} A{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}    "
            f"{charge:+6.3f} {ad_type:<2s}"
        )
        pdbqt_lines.append(pdbqt_line)

    # Write PDBQT
    output_pdbqt.write_text("\n".join(pdbqt_lines) + "\n", encoding="utf-8")
    print(f"  PDBQT: {len(pdbqt_lines)} lines ({n_heavy} heavy + {n_polar_h} polar H)")
    print(f"  Stripped {n_nonpolar_h} non-polar H")

    # Write full-H PDB (for complex assembly)
    output_pdb.write_text("\n".join(pdb_full_lines) + "\nEND\n", encoding="utf-8")
    print(f"  PDB:   {len(pdb_full_lines)} lines (all H)")

    return pdbqt_lines


# ===================================================================
# RIGID + FLEX SPLIT (Vina flexible docking)
# ===================================================================
def split_rigid_flex(
    pdbqt_path: Path,
    rigid_out: Path,
    flex_out: Path,
    flex_resnums: list[int],
) -> tuple[int, int]:
    """Split receptor PDBQT into rigid + flexible parts for Vina --flex.

    Rigid PDBQT: full receptor with sidechain atoms of flex residues removed.
        Backbone (N, CA, C, O + amide H) of flex residues stays.
    Flex PDBQT:  CA (ROOT) + sidechain (BRANCH tree) for each flex residue.
        CA is duplicated in both files as the backbone anchor for Vina.

    Returns (n_rigid_atom_lines, n_flex_atoms).
    """
    flex_set = set(flex_resnums)
    BACKBONE_NAMES = {"N", "CA", "C", "O"}
    BACKBONE_H = {"H", "HN", "H1", "H2", "H3"}

    lines = pdbqt_path.read_text().splitlines()

    # ── Parse all atoms grouped by residue ──────────────────────────
    atoms_by_res: dict[int, list[dict]] = {}
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        try:
            aname = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resnum = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            ad_type = line[77:79].strip() if len(line) >= 79 else aname[0]
        except (ValueError, IndexError):
            continue
        atoms_by_res.setdefault(resnum, []).append({
            "name": aname, "resname": resname, "chain": chain,
            "resnum": resnum, "xyz": np.array([x, y, z]),
            "ad_type": ad_type, "line": line,
        })

    # ── Build rigid PDBQT ───────────────────────────────────────────
    rigid_lines = []
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        aname = line[12:16].strip()
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            rigid_lines.append(line)
            continue
        if resnum in flex_set:
            # Keep backbone + backbone H only
            if aname in BACKBONE_NAMES or aname in BACKBONE_H:
                rigid_lines.append(line)
        else:
            rigid_lines.append(line)
    rigid_out.write_text("\n".join(rigid_lines) + "\n", encoding="utf-8")

    # ── Build flex PDBQT ────────────────────────────────────────────
    flex_blocks: list[str] = []
    total_flex_atoms = 0

    for resnum in sorted(flex_set):
        if resnum not in atoms_by_res:
            print(f"    WARNING: Flex residue {resnum} not in PDBQT")
            continue

        res_atoms = atoms_by_res[resnum]
        resname = res_atoms[0]["resname"]
        chain = res_atoms[0]["chain"]

        tree = FLEX_SIDECHAIN_TREES.get(resname)
        if tree is None:
            print(f"    WARNING: No sidechain tree for {resname}{resnum}")
            continue

        # Separate: CA, sidechain heavy, sidechain H
        ca_atom = None
        sc_heavy: dict[str, dict] = {}
        sc_h_list: list[dict] = []

        for a in res_atoms:
            if a["name"] == "CA":
                ca_atom = a
            elif a["name"] in BACKBONE_NAMES or a["name"] in BACKBONE_H:
                continue  # backbone → already in rigid
            elif a["ad_type"].startswith("H"):
                sc_h_list.append(a)
            else:
                sc_heavy[a["name"]] = a

        if ca_atom is None:
            print(f"    WARNING: No CA for {resname}{resnum}")
            continue

        # Assign each sidechain H to nearest sidechain heavy atom
        h_by_heavy: dict[str, list[dict]] = {}
        for h in sc_h_list:
            best_d, best_hv = float("inf"), None
            for hv_name, hv in sc_heavy.items():
                d = float(np.linalg.norm(h["xyz"] - hv["xyz"]))
                if d < best_d:
                    best_d, best_hv = d, hv_name
            if best_hv:
                h_by_heavy.setdefault(best_hv, []).append(h)

        # ── Generate BEGIN_RES / END_RES block ──────────────────────
        block_lines: list[str] = []
        serial_ctr = [0]
        serial_map: dict[str, int] = {}

        def _next():
            serial_ctr[0] += 1
            return serial_ctr[0]

        def _fmt(orig_line: str, s: int) -> str:
            """Replace serial number in PDBQT atom line."""
            return f"ATOM  {s:5d}" + orig_line[11:]

        block_lines.append(f"BEGIN_RES {resname} {chain}{resnum:4d}")

        # ROOT: CA atom only (backbone anchor)
        block_lines.append("ROOT")
        ca_s = _next()
        serial_map["CA"] = ca_s
        block_lines.append(_fmt(ca_atom["line"], ca_s))
        block_lines.append("ENDROOT")

        def _emit(node):
            """Recursively emit BRANCH/ENDBRANCH for sidechain torsions."""
            parent_name, child_name = node["bond"]
            if parent_name not in serial_map or child_name not in sc_heavy:
                return
            parent_s = serial_map[parent_name]
            child_s = _next()
            serial_map[child_name] = child_s

            block_lines.append(f"BRANCH{parent_s:4d}{child_s:4d}")

            # Output all heavy atoms in this rigid group + their polar H
            for aname in node["heavy"]:
                if aname not in sc_heavy:
                    continue
                if aname == child_name:
                    a_s = child_s
                else:
                    a_s = _next()
                    serial_map[aname] = a_s
                block_lines.append(_fmt(sc_heavy[aname]["line"], a_s))
                # Polar H bonded to this heavy atom
                for h_atom in h_by_heavy.get(aname, []):
                    block_lines.append(_fmt(h_atom["line"], _next()))

            # Recurse into child branches
            for child_node in node.get("children", []):
                _emit(child_node)

            block_lines.append(f"ENDBRANCH{parent_s:4d}{child_s:4d}")

        _emit(tree)
        total_flex_atoms += serial_ctr[0]

        block_lines.append(f"END_RES {resname} {chain}{resnum:4d}")
        flex_blocks.append("\n".join(block_lines))
        print(f"    {resname}{resnum}: CA(ROOT) + "
              f"{len(sc_heavy)} heavy + {len(sc_h_list)} H "
              f"= {serial_ctr[0]} atoms")

    flex_out.write_text("\n".join(flex_blocks) + "\n", encoding="utf-8")
    print(f"  Rigid: {len(rigid_lines)} atom lines -> {rigid_out.name}")
    print(f"  Flex:  {total_flex_atoms} atoms in {len(flex_blocks)} residues"
          f" -> {flex_out.name}")
    return len(rigid_lines), total_flex_atoms


def compute_grid_box(pdbqt_lines: list[str], pocket_res: list[int],
                     padding: float = 8.0) -> tuple:
    """Compute Vina grid box from pocket residue coordinates."""
    pocket_xyz = []
    for line in pdbqt_lines:
        if not line.startswith("ATOM"):
            continue
        try:
            rn = int(line[22:26].strip())
            if rn in pocket_res:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                pocket_xyz.append([x, y, z])
        except (ValueError, IndexError):
            continue

    if not pocket_xyz:
        print("ERROR: No pocket atoms found!")
        return None, None

    arr = np.array(pocket_xyz)
    center = arr.mean(axis=0)
    span = arr.max(axis=0) - arr.min(axis=0)

    # Box = pocket span + 2*padding, rounded to int, clamped to Vina limit
    box_size = tuple(min(30.0, max(20.0, float(s + 2 * padding))) for s in span)
    box_center = tuple(float(c) for c in center)

    print(f"\n  Grid box center: ({box_center[0]:.2f}, {box_center[1]:.2f}, {box_center[2]:.2f})")
    print(f"  Grid box size:   ({box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f}) Å")
    vol = box_size[0] * box_size[1] * box_size[2]
    print(f"  Volume: {vol:.0f} ų")

    return box_center, box_size


def main():
    print("=" * 60)
    print("PREPARE 1DB1 CRYSTAL RECEPTOR FOR VINA")
    print("=" * 60)

    if not INPUT_PDB.exists():
        print(f"ERROR: {INPUT_PDB} not found")
        sys.exit(1)

    # Step 1: Extract chain A protein atoms
    print("\n1. Extracting chain A protein atoms...")
    protein_lines = extract_chain_a_protein(INPUT_PDB)

    # Step 2: PDBFixer + hydrogens
    print("\n2. Adding missing atoms + hydrogens (pH 7.4)...")
    pdb_text = add_hydrogens_pdbfixer(protein_lines, ph=7.4)

    # Count
    n_atoms = sum(1 for l in pdb_text.splitlines()
                  if l.startswith("ATOM") or l.startswith("HETATM"))
    print(f"  Total atoms after adding H: {n_atoms}")

    # Step 3: Write PDBQT + PDB
    print("\n3. Assigning atom types + charges, writing PDBQT...")
    pdbqt_lines = write_pdbqt(pdb_text, OUTPUT_PDBQT, OUTPUT_PDB)

    # Step 3b: Split into rigid + flex for Vina flexible docking
    print("\n3b. Splitting PDBQT for Vina flexible docking (--flex)...")
    flex_names = [KEY_BINDING_RESIDUES.get(r, str(r)) for r in FLEX_RESIDUES]
    print(f"    Flex residues: {flex_names}")
    n_rigid, n_flex = split_rigid_flex(
        OUTPUT_PDBQT, OUTPUT_RIGID, OUTPUT_FLEX, FLEX_RESIDUES)

    # Step 4: Compute grid box
    print("\n4. Computing binding pocket grid box...")
    pocket_res = list(KEY_BINDING_RESIDUES.keys())
    box_center, box_size = compute_grid_box(pdbqt_lines, pocket_res)

    if box_center:
        print(f"\n  Python constants for run_vina_docking.py:")
        print(f'  BOX_CENTER = ({box_center[0]:.2f}, {box_center[1]:.2f}, {box_center[2]:.2f})')
        print(f'  BOX_SIZE   = ({box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f})')

    # Verify binding residues
    print("\n5. Verifying binding residues in PDBQT...")
    found_res = set()
    for line in pdbqt_lines:
        if line.startswith("ATOM"):
            try:
                rn = int(line[22:26].strip())
                found_res.add(rn)
            except ValueError:
                continue

    for rn, label in sorted(KEY_BINDING_RESIDUES.items()):
        status = "OK" if rn in found_res else "MISSING"
        print(f"    {label}: {status}")

    # NE2 atoms for His305/His397
    print("\n  H-bond acceptors for calcitriol 25-OH:")
    for line in pdbqt_lines:
        if not line.startswith("ATOM"):
            continue
        aname = line[12:16].strip()
        rn = int(line[22:26].strip())
        if rn in (305, 397) and aname == "NE2":
            ad_type = line.split()[-1]
            print(f"    {KEY_BINDING_RESIDUES[rn]} NE2: type={ad_type}")

    print(f"\n  Output files:")
    print(f"    {OUTPUT_PDBQT}")
    print(f"    {OUTPUT_PDB}")
    print(f"    {OUTPUT_RIGID}  (rigid receptor for Vina --flex)")
    print(f"    {OUTPUT_FLEX}   (flex sidechains for Vina --flex)")
    print("\nDone!")


if __name__ == "__main__":
    main()
