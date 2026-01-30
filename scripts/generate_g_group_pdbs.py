"""
Generate docked PDB files for top5/bottom5 compounds per G group (G1, G2, G3).

This script:
1. Loads CWRA final rankings
2. Maps generators to G groups (G1=reference/reinvent/gmdldr, G2=hybrid, G3=transmol-reinvent-gmdldr)
3. Selects top 5 and bottom 5 per G group (excluding 'reference' generator)
4. Generates 3D conformers from SMILES using RDKit
5. Docks to VDR 1DB1 using AutoDock Vina
6. Saves PDB files organized by group and rank position
7. Includes calcitriol as reference

Usage:
    python scripts/generate_g_group_pdbs.py --input results/cwra_final/cwra_final_rankings.csv \
        --output results/cwra_final/g_group_pdbs --receptor pdb/1DB1.pdb
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False


# VDR 1DB1 binding site parameters (calcitriol binding pocket)
# Calculated from native VDX ligand coordinates in 1DB1 crystal structure
VDR_1DB1_CENTER = (11.5, 23.0, 34.5)
VDR_1DB1_BOX_SIZE = (24, 22, 16)  # Ligand span + 10 Angstrom margin

# Calcitriol SMILES (1,25-dihydroxyvitamin D3)
CALCITRIOL_SMILES = "C=C1/C(=C\\C=C2/CCC[C@]3(C)[C@@H]([C@H](C)CCCC(C)(C)O)CC[C@@H]23)C[C@@H](O)C[C@@H]1O"


@dataclass
class Compound:
    """Represents a compound for docking."""
    smiles: str
    rank: int
    source: str
    generator: str
    group: str
    category: str  # 'top' or 'bottom'


def map_generator_to_group(generator: str) -> str:
    """Map generator to G group."""
    gen_lower = str(generator).lower()
    
    # G3: Triple hybrid (transmol-reinvent-gmdldr)
    if 'transmol-reinvent-gmdldr' in gen_lower:
        return 'G3'
    
    # G2: Double hybrids
    if any(x in gen_lower for x in ['gmdldr_reinvent', 'reinvent_transmol', 'gmdldr_transmol',
                                      'reinvent_gmdldr', 'transmol_reinvent', 'transmol_gmdldr']):
        return 'G2'
    
    # G1: Single generators (reinvent, gmdldr, transmol) or reference
    return 'G1'


def is_reference(generator: str) -> bool:
    """Check if generator is a reference compound."""
    return str(generator).lower() in ['reference', 'calcitriol']


def smiles_to_3d_mol(smiles: str, num_confs: int = 10) -> Optional[Chem.Mol]:
    """Convert SMILES to 3D molecule with optimized conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    # Generate conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(conf_ids) == 0:
        # Fallback to basic embedding
        AllChem.EmbedMolecule(mol, randomSeed=42)
        if mol.GetNumConformers() == 0:
            return None
    
    # Optimize conformers with MMFF
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        if results:
            # Select lowest energy conformer
            energies = [r[1] if r[0] == 0 else float('inf') for r in results]
            best_conf = int(np.argmin(energies))
            
            # Keep only best conformer
            for i in range(mol.GetNumConformers() - 1, -1, -1):
                if i != best_conf:
                    mol.RemoveConformer(i)
    except Exception:
        pass
    
    return mol


def mol_to_pdbqt_string(mol: Chem.Mol) -> str:
    """Convert RDKit mol to PDBQT format string using meeko."""
    if MEEKO_AVAILABLE:
        try:
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)[0]
            pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]
            return pdbqt_string
        except Exception as e:
            print(f"  Meeko error: {e}, falling back to manual conversion")
    
    # Fallback: manual conversion (less accurate but works)
    pdb_block = Chem.MolToPDBBlock(mol)
    
    # Simple conversion: add AutoDock atom types with ROOT/ENDROOT
    lines = ["ROOT"]
    for line in pdb_block.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Parse atom info
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            
            # Assign AutoDock atom type
            if element == 'C':
                ad_type = 'C'
            elif element == 'N':
                ad_type = 'NA'  # Hydrogen bond acceptor nitrogen
            elif element == 'O':
                ad_type = 'OA'  # Hydrogen bond acceptor oxygen
            elif element == 'S':
                ad_type = 'SA'
            elif element == 'H':
                ad_type = 'HD' if 'HO' in atom_name or 'HN' in atom_name else 'H'
            elif element == 'F':
                ad_type = 'F'
            elif element == 'Cl':
                ad_type = 'Cl'
            elif element == 'Br':
                ad_type = 'Br'
            elif element == 'I':
                ad_type = 'I'
            else:
                ad_type = element
            
            # Format PDBQT line (charge = 0.0 for simplicity)
            new_line = line[:54] + '  0.00' + f'  {ad_type:>2s}'
            lines.append(new_line)
        elif line.startswith('TER'):
            pass  # Skip TER in ligand PDBQT
        # Skip END, CONECT, MASTER records (Vina doesn't support them)
    
    lines.append("ENDROOT")
    lines.append("TORSDOF 0")
    return '\n'.join(lines)


def run_vina_docking(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    exhaustiveness: int = 8,
    num_modes: int = 1
) -> Tuple[Optional[str], Optional[float]]:
    """Run AutoDock Vina docking and return best pose PDBQT and score."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ligand_path = Path(tmpdir) / "ligand.pdbqt"
        output_path = Path(tmpdir) / "output.pdbqt"
        
        ligand_path.write_text(ligand_pdbqt)
        
        cmd = [
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", str(ligand_path),
            "--out", str(output_path),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(box_size[0]),
            "--size_y", str(box_size[1]),
            "--size_z", str(box_size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Vina error: {result.stderr}")
                return None, None
            
            # Parse output
            if output_path.exists():
                output_pdbqt = output_path.read_text()
                
                # Extract score from output
                score = None
                for line in result.stdout.split('\n'):
                    if line.strip().startswith('1'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                score = float(parts[1])
                            except ValueError:
                                pass
                        break
                
                return output_pdbqt, score
            
        except subprocess.TimeoutExpired:
            print("Vina timeout")
        except FileNotFoundError:
            print("Vina not found in PATH")
        except Exception as e:
            print(f"Vina error: {e}")
    
    return None, None


def pdbqt_to_pdb(pdbqt_content: str) -> str:
    """Convert PDBQT to PDB format."""
    lines = []
    for line in pdbqt_content.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Remove charge and AD type columns
            pdb_line = line[:54] + line[70:78] if len(line) > 70 else line[:54]
            lines.append(pdb_line)
        elif line.startswith('END') or line.startswith('MODEL') or line.startswith('ENDMDL'):
            lines.append(line)
    return '\n'.join(lines)


def generate_3d_pdb_from_smiles(smiles: str, compound_name: str) -> Optional[str]:
    """Generate 3D PDB from SMILES without docking (fallback)."""
    mol = smiles_to_3d_mol(smiles)
    if mol is None:
        return None
    
    # Get PDB block
    pdb_block = Chem.MolToPDBBlock(mol)
    
    # Add header
    header = f"REMARK   Generated 3D structure for {compound_name}\n"
    header += f"REMARK   SMILES: {smiles}\n"
    
    return header + pdb_block


def prepare_receptor_pdbqt(pdb_path: str) -> str:
    """Prepare receptor PDBQT file path."""
    pdb_path = Path(pdb_path)
    pdbqt_path = pdb_path.with_suffix('.pdbqt')
    
    # Force regeneration - delete existing file
    if pdbqt_path.exists():
        pdbqt_path.unlink()
    
    # Try to convert using prepare_receptor if available (from ADT)
    try:
        cmd = ["prepare_receptor", "-r", str(pdb_path), "-o", str(pdbqt_path)]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if pdbqt_path.exists() and result.returncode == 0:
            return str(pdbqt_path)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Manual conversion with proper PDBQT format
    pdb_content = pdb_path.read_text()
    pdbqt_lines = []
    
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Skip ligand/water
            res_name = line[17:20].strip()
            if res_name in ['HOH', 'WAT', 'VDX', 'VD3']:  # Skip water and vitamin D ligand
                continue
            
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ''
            if not element:
                # Infer element from atom name
                element = atom_name[0] if atom_name[0].isalpha() else atom_name[1] if len(atom_name) > 1 else 'C'
            
            # AD4 atom type assignment for protein
            if element == 'C':
                ad_type = 'C'
            elif element == 'N':
                # Check if it's a hydrogen bond acceptor
                ad_type = 'NA' if atom_name in ['N', 'NE', 'NE2', 'ND1', 'ND2', 'NZ', 'NH1', 'NH2'] else 'N'
            elif element == 'O':
                ad_type = 'OA'  # Oxygen is always H-bond acceptor in proteins
            elif element == 'S':
                ad_type = 'SA'
            elif element == 'H':
                ad_type = 'H'
            else:
                ad_type = element[:2] if len(element) >= 2 else element
            
            # Build PDBQT line: columns 1-54 from PDB, then charge (55-60), then AD type (77-78)
            # PDBQT format: cols 1-54 (coords), 55-60 (partial charge), 67-68 or 77-78 (AD type)
            pdb_part = line[:54].ljust(54)
            charge = "  0.00"  # 6 chars for partial charge
            # Pad to column 77, then AD type
            new_line = pdb_part + charge + " " * 16 + f"{ad_type:>2s}"
            pdbqt_lines.append(new_line)
            
        elif line.startswith('TER'):
            pdbqt_lines.append('TER')
        # Skip END, CONECT, MASTER records - Vina doesn't support them
    
    pdbqt_content = '\n'.join(pdbqt_lines)
    pdbqt_path.write_text(pdbqt_content)
    
    return str(pdbqt_path)


def main():
    parser = argparse.ArgumentParser(description="Generate docked PDBs for G groups")
    parser.add_argument("--input", "-i", default="results/cwra_final/cwra_final_rankings.csv",
                        help="Input CWRA rankings CSV")
    parser.add_argument("--output", "-o", default="results/cwra_final/g_group_pdbs",
                        help="Output directory")
    parser.add_argument("--receptor", "-r", default="pdb/1DB1.pdb",
                        help="Receptor PDB file (VDR 1DB1)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top compounds per group")
    parser.add_argument("--bottom-n", type=int, default=5, help="Number of bottom compounds per group")
    parser.add_argument("--skip-docking", action="store_true", help="Skip docking, just generate 3D structures")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading rankings from {args.input}")
    df = pd.read_csv(args.input)
    
    # Ensure rank column
    if 'cwra_rank' not in df.columns:
        print("ERROR: cwra_rank column not found")
        return 1
    
    # Map generators to groups
    df['group'] = df['generator'].apply(map_generator_to_group)
    df['is_reference'] = df['generator'].apply(is_reference)
    
    # Filter out reference compounds
    df_filtered = df[~df['is_reference']].copy()
    
    print(f"\nTotal compounds (excluding reference): {len(df_filtered)}")
    print(f"Compounds per group:")
    print(df_filtered['group'].value_counts().sort_index())
    
    # Create output directories
    output_dir = Path(args.output)
    for group in ['G1', 'G2', 'G3']:
        (output_dir / 'top' / group).mkdir(parents=True, exist_ok=True)
        (output_dir / 'bottom' / group).mkdir(parents=True, exist_ok=True)
    (output_dir / 'reference').mkdir(parents=True, exist_ok=True)
    
    # Prepare receptor if docking
    receptor_pdbqt = None
    if not args.skip_docking:
        try:
            receptor_pdbqt = prepare_receptor_pdbqt(args.receptor)
            print(f"\nReceptor prepared: {receptor_pdbqt}")
        except Exception as e:
            print(f"Warning: Could not prepare receptor, skipping docking: {e}")
            args.skip_docking = True
    
    # Collect compounds to process
    compounds_to_process: List[Compound] = []
    
    for group in ['G1', 'G2', 'G3']:
        group_df = df_filtered[df_filtered['group'] == group].sort_values('cwra_rank')
        
        # Top N
        top_df = group_df.head(args.top_n)
        for _, row in top_df.iterrows():
            compounds_to_process.append(Compound(
                smiles=row['smiles'],
                rank=int(row['cwra_rank']),
                source=row['source'],
                generator=row['generator'],
                group=group,
                category='top'
            ))
        
        # Bottom N
        bottom_df = group_df.tail(args.bottom_n)
        for _, row in bottom_df.iterrows():
            compounds_to_process.append(Compound(
                smiles=row['smiles'],
                rank=int(row['cwra_rank']),
                source=row['source'],
                generator=row['generator'],
                group=group,
                category='bottom'
            ))
    
    print(f"\nProcessing {len(compounds_to_process)} compounds + calcitriol reference")
    
    # Add calcitriol as reference
    calcitriol = Compound(
        smiles=CALCITRIOL_SMILES,
        rank=0,
        source='calcitriol',
        generator='calcitriol',
        group='reference',
        category='reference'
    )
    
    # Process all compounds
    manifest_rows = []
    
    # Process calcitriol first
    print("\nProcessing calcitriol reference...")
    pdb_content = generate_3d_pdb_from_smiles(calcitriol.smiles, "calcitriol")
    if pdb_content:
        out_path = output_dir / 'reference' / 'calcitriol.pdb'
        out_path.write_text(pdb_content)
        manifest_rows.append({
            'set': 'reference',
            'group': 'reference',
            'rank': 0,
            'generator': 'calcitriol',
            'source': 'calcitriol',
            'smiles': calcitriol.smiles,
            'pdb_file': str(out_path.relative_to(output_dir)),
            'status': 'success'
        })
        print(f"  Saved: {out_path}")
    
    # Process G group compounds
    for i, comp in enumerate(compounds_to_process, 1):
        print(f"\n[{i}/{len(compounds_to_process)}] Processing {comp.group} {comp.category} rank {comp.rank}")
        
        compound_name = f"{comp.group}_{comp.category}_rank{comp.rank:05d}"
        
        # Generate 3D structure
        if args.skip_docking:
            pdb_content = generate_3d_pdb_from_smiles(comp.smiles, compound_name)
            status = 'success_3d' if pdb_content else 'failed'
        else:
            # Try docking
            mol = smiles_to_3d_mol(comp.smiles)
            if mol is None:
                print(f"  Failed to generate 3D structure")
                pdb_content = None
                status = 'failed_3d'
            else:
                ligand_pdbqt = mol_to_pdbqt_string(mol)
                docked_pdbqt, score = run_vina_docking(
                    ligand_pdbqt,
                    receptor_pdbqt,
                    VDR_1DB1_CENTER,
                    VDR_1DB1_BOX_SIZE,
                    exhaustiveness=args.exhaustiveness
                )
                
                if docked_pdbqt:
                    pdb_content = pdbqt_to_pdb(docked_pdbqt)
                    # Add header
                    header = f"REMARK   Docked structure for {compound_name}\n"
                    header += f"REMARK   SMILES: {comp.smiles}\n"
                    header += f"REMARK   Vina score: {score} kcal/mol\n" if score else ""
                    pdb_content = header + pdb_content
                    status = 'docked'
                    print(f"  Docked with score: {score}")
                else:
                    # Fallback to 3D without docking
                    pdb_content = generate_3d_pdb_from_smiles(comp.smiles, compound_name)
                    status = 'success_3d' if pdb_content else 'failed'
        
        if pdb_content:
            out_path = output_dir / comp.category / comp.group / f"rank{comp.rank:05d}_{comp.generator}.pdb"
            out_path.write_text(pdb_content)
            print(f"  Saved: {out_path}")
        else:
            out_path = None
            print(f"  FAILED")
        
        manifest_rows.append({
            'set': comp.category,
            'group': comp.group,
            'rank': comp.rank,
            'generator': comp.generator,
            'source': comp.source,
            'smiles': comp.smiles,
            'pdb_file': str(out_path.relative_to(output_dir)) if out_path else '',
            'status': status
        })
    
    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / 'manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for group in ['G1', 'G2', 'G3']:
        group_data = manifest_df[manifest_df['group'] == group]
        top_count = len(group_data[group_data['set'] == 'top'])
        bottom_count = len(group_data[group_data['set'] == 'bottom'])
        success_count = len(group_data[group_data['status'].str.contains('success|docked', na=False)])
        print(f"{group}: top={top_count}, bottom={bottom_count}, success={success_count}")
    
    print(f"\nCalcitriol reference: {'✓' if (output_dir / 'reference' / 'calcitriol.pdb').exists() else '✗'}")
    print(f"\nOutput directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
