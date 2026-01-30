#!/usr/bin/env python3
"""
Compute UniMol Similarity for Extended G1 Compounds

Uses UniMol's pre-trained molecular encoder to compute cosine similarity
between query molecules and a reference ligand (Calcitriol for VDR).
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Calcitriol (1,25-dihydroxyvitamin D3) - natural VDR ligand
CALCITRIOL_SMILES = "CC(CCCC(C)(C)O)C1CCC2C(=CC=C3CC(O)CC(O)C3=C)CCCC21C"

def compute_unimol_similarity(input_csv: str, output_csv: str, 
                              reference_smiles: str = CALCITRIOL_SMILES,
                              batch_size: int = 64, device: str = "cuda"):
    """
    Compute UniMol molecular similarity.
    
    Args:
        input_csv: Path to input CSV with 'smiles' column
        output_csv: Path to output CSV
        reference_smiles: Reference molecule SMILES
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'
    """
    print("="*70)
    print("UniMol Molecular Similarity Computation")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} compounds from {input_csv}")
    
    # Check for existing results
    if 'unimol_similarity' in df.columns:
        computed = df['unimol_similarity'].notna().sum()
        print(f"Already computed: {computed}/{len(df)}")
        if computed == len(df):
            print("All similarities already computed. Skipping.")
            df.to_csv(output_csv, index=False)
            return df
    else:
        df['unimol_similarity'] = np.nan
    
    # Get SMILES to compute
    smiles_list = df['smiles'].tolist()
    indices_to_compute = df[df['unimol_similarity'].isna()].index.tolist()
    
    if not indices_to_compute:
        print("No compounds to compute.")
        df.to_csv(output_csv, index=False)
        return df
    
    print(f"Computing similarity for {len(indices_to_compute)} compounds...")
    
    try:
        from unimol_tools import UniMolRepr
        
        print(f"Using device: {device}")
        clf = UniMolRepr(data_type='molecule', remove_hs=False)
        
        # Get reference embedding
        print("Computing reference embedding...")
        ref_repr = clf.get_repr([reference_smiles])
        # API returns a list of numpy arrays directly
        ref_emb = ref_repr[0] / np.linalg.norm(ref_repr[0])
        
        # Process in batches
        results = {}
        smiles_to_process = [smiles_list[i] for i in indices_to_compute]
        
        for i in tqdm(range(0, len(smiles_to_process), batch_size), desc="Computing"):
            batch_smiles = smiles_to_process[i:i+batch_size]
            batch_indices = indices_to_compute[i:i+batch_size]
            
            try:
                # API returns a list of numpy arrays directly
                batch_repr = clf.get_repr(batch_smiles)
                
                for j, (idx, emb) in enumerate(zip(batch_indices, batch_repr)):
                    emb_norm = emb / np.linalg.norm(emb)
                    similarity = float(np.dot(ref_emb, emb_norm))
                    results[idx] = similarity
                    
            except Exception as e:
                print(f"Batch error at {i}: {e}")
                # Process individually for failed batch
                for idx, smi in zip(batch_indices, batch_smiles):
                    try:
                        repr_single = clf.get_repr([smi])
                        emb = repr_single[0]
                        emb_norm = emb / np.linalg.norm(emb)
                        similarity = float(np.dot(ref_emb, emb_norm))
                        results[idx] = similarity
                    except Exception as e2:
                        print(f"  Failed for idx {idx}: {e2}")
                        results[idx] = np.nan
        
        # Update dataframe
        for idx, sim in results.items():
            df.at[idx, 'unimol_similarity'] = sim
        
        print(f"\nSuccessfully computed: {len([v for v in results.values() if not np.isnan(v)])}")
        
    except ImportError as e:
        print(f"UniMol not available: {e}")
        print("Please install: pip install unimol_tools")
        sys.exit(1)
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute UniMol similarity")
    parser.add_argument("--input", default="data/vdr_unique_G1_extended.csv",
                        help="Input CSV file with smiles column")
    parser.add_argument("--output", default="data/vdr_unique_G1_with_unimol.csv",
                        help="Output CSV file")
    parser.add_argument("--reference", default=CALCITRIOL_SMILES,
                        help="Reference SMILES for similarity")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for computation")
    args = parser.parse_args()
    
    compute_unimol_similarity(
        args.input,
        args.output,
        args.reference,
        args.batch_size,
        args.device
    )


if __name__ == "__main__":
    main()
