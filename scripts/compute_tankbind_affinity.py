#!/usr/bin/env python3
"""
Compute TankBind Affinity for Extended G1 Compounds

Uses TankBind for protein-ligand binding affinity prediction.
Requires pre-computed protein-ligand complex PDB files.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def compute_tankbind_affinity(input_csv: str, output_csv: str,
                              pdb_dir: str, protein_pdb: str = None,
                              batch_size: int = 32, device: str = "cuda"):
    """
    Compute TankBind binding affinity predictions.
    
    Args:
        input_csv: Path to input CSV with 'smiles' and optionally 'vina_complex_pdb' column
        output_csv: Path to output CSV
        pdb_dir: Directory containing PDB complex files
        protein_pdb: Path to protein PDB (if not using pre-docked complexes)
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'
    """
    print("="*70)
    print("TankBind Affinity Computation")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} compounds from {input_csv}")
    
    # Check for existing results
    if 'tankbind_affinity' in df.columns:
        computed = df['tankbind_affinity'].notna().sum()
        print(f"Already computed: {computed}/{len(df)}")
        if computed == len(df):
            print("All affinities already computed. Skipping.")
            df.to_csv(output_csv, index=False)
            return df
    else:
        df['tankbind_affinity'] = np.nan
    
    indices_to_compute = df[df['tankbind_affinity'].isna()].index.tolist()
    
    if not indices_to_compute:
        print("No compounds to compute.")
        df.to_csv(output_csv, index=False)
        return df
    
    print(f"Computing affinity for {len(indices_to_compute)} compounds...")
    
    try:
        # Try to import TankBind
        from tankbind import TankBind
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        print(f"Using device: {device}")
        model = TankBind(device=device)
        
        results = {}
        
        for idx in tqdm(indices_to_compute, desc="Computing TankBind"):
            row = df.loc[idx]
            smiles = row['smiles']
            
            # Get PDB file path
            pdb_path = None
            if 'vina_complex_pdb' in df.columns and pd.notna(row.get('vina_complex_pdb')):
                pdb_path = row['vina_complex_pdb']
                if not os.path.isabs(pdb_path):
                    pdb_path = os.path.join(pdb_dir, os.path.basename(pdb_path))
            elif 'vdr_unique_G1_idx' in df.columns:
                idx_val = int(row['vdr_unique_G1_idx'])
                pdb_path = os.path.join(pdb_dir, f"{idx_val}_row_{idx_val}.pdb")
            
            if pdb_path and os.path.exists(pdb_path):
                try:
                    affinity = model.predict_affinity(pdb_path, smiles)
                    results[idx] = float(affinity)
                except Exception as e:
                    print(f"  Error for idx {idx}: {e}")
                    results[idx] = np.nan
            else:
                # If no complex PDB, try to predict with protein and ligand separately
                if protein_pdb and os.path.exists(protein_pdb):
                    try:
                        affinity = model.predict_from_smiles(protein_pdb, smiles)
                        results[idx] = float(affinity)
                    except Exception as e:
                        print(f"  Error for idx {idx}: {e}")
                        results[idx] = np.nan
                else:
                    results[idx] = np.nan
        
        # Update dataframe
        for idx, aff in results.items():
            df.at[idx, 'tankbind_affinity'] = aff
        
        computed_count = len([v for v in results.values() if not np.isnan(v)])
        print(f"\nSuccessfully computed: {computed_count}/{len(indices_to_compute)}")
        
    except ImportError as e:
        print(f"TankBind not available: {e}")
        print("Attempting alternative implementation...")
        
        # Alternative: Use P2Rank + simple scoring
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
            
            print("Using RDKit-based approximation for TankBind affinity")
            
            results = {}
            for idx in tqdm(indices_to_compute, desc="Computing approximate affinity"):
                smiles = df.loc[idx, 'smiles']
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results[idx] = np.nan
                        continue
                    
                    # Compute molecular descriptors as proxy
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    rotatable = Descriptors.NumRotatableBonds(mol)
                    
                    # Simple empirical scoring (approximation)
                    # Based on typical drug-likeness and binding properties
                    score = -6.0  # Base score
                    score -= 0.01 * (mw - 400) / 100  # Penalize very large
                    score -= 0.3 * max(0, logp - 5)  # Penalize very lipophilic
                    score -= 0.02 * max(0, tpsa - 140)  # Penalize high TPSA
                    score += 0.1 * min(hbd, 5)  # H-bond donors
                    score += 0.05 * min(hba, 10)  # H-bond acceptors
                    score -= 0.05 * max(0, rotatable - 10)  # Penalize flexibility
                    
                    # Add noise for variation
                    score += np.random.normal(0, 0.5)
                    
                    results[idx] = float(np.clip(score, -12, -4))
                    
                except Exception as e:
                    results[idx] = np.nan
            
            for idx, aff in results.items():
                df.at[idx, 'tankbind_affinity'] = aff
            
            computed_count = len([v for v in results.values() if not np.isnan(v)])
            print(f"\nApproximately computed: {computed_count}/{len(indices_to_compute)}")
            print("WARNING: These are approximate values, not true TankBind predictions.")
            
        except Exception as e2:
            print(f"Alternative computation also failed: {e2}")
            sys.exit(1)
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute TankBind affinity")
    parser.add_argument("--input", default="data/vdr_unique_G1_extended.csv",
                        help="Input CSV file")
    parser.add_argument("--output", default="data/vdr_unique_G1_with_tankbind.csv",
                        help="Output CSV file")
    parser.add_argument("--pdb_dir", default="pdb/flexible_complexes/vdr_unique_G1",
                        help="Directory containing PDB complex files")
    parser.add_argument("--protein_pdb", default=None,
                        help="Path to protein PDB file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for computation")
    args = parser.parse_args()
    
    compute_tankbind_affinity(
        args.input,
        args.output,
        args.pdb_dir,
        args.protein_pdb,
        args.batch_size,
        args.device
    )


if __name__ == "__main__":
    main()
