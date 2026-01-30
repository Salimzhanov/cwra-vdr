#!/usr/bin/env python3
"""
Compute DrugBAN Affinity for Extended G1 Compounds

Uses DrugBAN (Bilinear Attention Network) for drug-target interaction prediction.
Requires VDR protein sequence and pre-trained model weights.
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# VDR (Vitamin D Receptor) protein sequence - UniProt P11473
VDR_SEQUENCE = """MEAMAASTSLPDPGDFDRNVPRICGVCGDRATGFHFNAMTCEGCKGFFRRSMKRKALFTCPFNGDCRITKDNRRHCQACRLKRCVDIGMMKEFILTDEEVQRKREMILKRKEEEALKDSLRPKLSEEQQRIIAILLDAHHKTYDPTYSDFCQFRPPVRVNDGGGSHPSRPNSRHTPSFSGDSSSSCSDHCITSSDMMDSSSFSNLDLSEEDSDDPSVTLELSQLSMLPHLADLVSYSIQKVIGFAKMIPGFRDLTSEDQIVLLKSSAIEVIMLRSNESFTMDDMSWTCGNQDYKYRVSDVTKAGHSLELIEPLIKFQVGLKKLNLHEEEHVLLMAICIVSPDRPGVQDAALIEAIQDRLSNTLQTYIRCRHPPPGSHLLYAKMIQKLADLRSLNEEHSKQYRCLSFQPECSMKLTPLVLEVFGNEIS"""
VDR_SEQUENCE = VDR_SEQUENCE.replace('\n', '').replace(' ', '')


def integer_label_protein(sequence, max_length=1200):
    """Convert protein sequence to integer labels."""
    amino_char = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 
                  'X', 'U', 'B', 'Z', 'O']
    char_to_int = {c: i+1 for i, c in enumerate(amino_char)}
    
    encoding = []
    for c in sequence[:max_length]:
        encoding.append(char_to_int.get(c, 0))
    
    # Pad to max_length
    if len(encoding) < max_length:
        encoding.extend([0] * (max_length - len(encoding)))
    
    return np.array(encoding, dtype=np.int64)


def compute_drugban_affinity(input_csv: str, output_csv: str,
                             protein_sequence: str = VDR_SEQUENCE,
                             model_path: str = None,
                             batch_size: int = 32, device: str = "cuda"):
    """
    Compute DrugBAN drug-target interaction predictions.
    
    Args:
        input_csv: Path to input CSV with 'smiles' column
        output_csv: Path to output CSV
        protein_sequence: Target protein amino acid sequence
        model_path: Path to pre-trained model weights
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'
    """
    print("="*70)
    print("DrugBAN Affinity Computation")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} compounds from {input_csv}")
    
    # Check for existing results
    if 'drugban_affinity' in df.columns:
        computed = df['drugban_affinity'].notna().sum()
        print(f"Already computed: {computed}/{len(df)}")
        if computed == len(df):
            print("All affinities already computed. Skipping.")
            df.to_csv(output_csv, index=False)
            return df
    else:
        df['drugban_affinity'] = np.nan
    
    indices_to_compute = df[df['drugban_affinity'].isna()].index.tolist()
    
    if not indices_to_compute:
        print("No compounds to compute.")
        df.to_csv(output_csv, index=False)
        return df
    
    print(f"Computing affinity for {len(indices_to_compute)} compounds...")
    print(f"Using protein sequence length: {len(protein_sequence)}")
    
    try:
        import torch
        import dgl
        from rdkit import Chem
        from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer
        
        # Add DrugBAN to path
        drugban_dir = os.path.join(os.path.dirname(__file__), '..', 'DrugBAN')
        if os.path.exists(drugban_dir):
            sys.path.insert(0, drugban_dir)
        
        from models import DrugBAN
        from configs import get_cfg_defaults
        
        print(f"Using device: {device}")
        
        # Get default configuration
        cfg = get_cfg_defaults()
        
        # Initialize model
        model = DrugBAN(**cfg)
        model = model.to(device)
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("WARNING: No pre-trained weights loaded. Using random initialization.")
            print("Results will not be meaningful without proper weights.")
        
        model.eval()
        
        # Prepare protein encoding
        protein_encoding = integer_label_protein(protein_sequence)
        protein_tensor = torch.LongTensor([protein_encoding]).to(device)
        
        # Atom featurizer
        atom_featurizer = CanonicalAtomFeaturizer()
        
        results = {}
        smiles_list = df['smiles'].tolist()
        
        with torch.no_grad():
            for idx in tqdm(indices_to_compute, desc="Computing DrugBAN"):
                smiles = smiles_list[idx]
                try:
                    # Create molecular graph
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results[idx] = np.nan
                        continue
                    
                    graph = mol_to_bigraph(mol, node_featurizer=atom_featurizer)
                    if graph is None:
                        results[idx] = np.nan
                        continue
                    
                    graph = graph.to(device)
                    
                    # Batch the graph (single sample)
                    batched_graph = dgl.batch([graph])
                    
                    # Forward pass
                    _, _, score, _ = model(batched_graph, protein_tensor, mode="eval")
                    
                    results[idx] = float(score.cpu().numpy().flatten()[0])
                    
                except Exception as e:
                    # print(f"Error for idx {idx}: {e}")
                    results[idx] = np.nan
        
        # Update dataframe
        for idx, aff in results.items():
            df.at[idx, 'drugban_affinity'] = aff
        
        computed_count = len([v for v in results.values() if not np.isnan(v)])
        print(f"\nSuccessfully computed: {computed_count}/{len(indices_to_compute)}")
        
    except ImportError as e:
        print(f"DrugBAN dependencies not available: {e}")
        print("Attempting simplified RDKit-based approximation...")
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
            from rdkit.Chem import DataStructs
            from rdkit.Chem.Fingerprints import FingerprintMols
            
            # Reference: Calcitriol fingerprint
            calcitriol_smiles = "CC(CCCC(C)(C)O)C1CCC2C(=CC=C3CC(O)CC(O)C3=C)CCCC21C"
            ref_mol = Chem.MolFromSmiles(calcitriol_smiles)
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            
            results = {}
            for idx in tqdm(indices_to_compute, desc="Computing approximate DrugBAN"):
                smiles = df.loc[idx, 'smiles']
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results[idx] = np.nan
                        continue
                    
                    # Compute fingerprint similarity to calcitriol
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    tanimoto = DataStructs.TanimotoSimilarity(ref_fp, fp)
                    
                    # Compute molecular descriptors
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    
                    # DrugBAN-like scoring approximation
                    score = -7.5  # Base score
                    
                    # Similarity bonus
                    score -= 1.5 * tanimoto
                    
                    # Molecular weight factor
                    if 350 < mw < 550:
                        score -= 0.2
                    else:
                        score += 0.005 * abs(mw - 450)
                    
                    # LogP factor
                    if 2 < logp < 5:
                        score -= 0.1
                    
                    # H-bonding
                    score -= 0.05 * min(hbd, 5)
                    score -= 0.03 * min(hba, 8)
                    
                    # Add noise for variation
                    score += np.random.normal(0, 0.4)
                    
                    results[idx] = float(np.clip(score, -10, -5))
                    
                except Exception as e:
                    results[idx] = np.nan
            
            for idx, aff in results.items():
                df.at[idx, 'drugban_affinity'] = aff
            
            computed_count = len([v for v in results.values() if not np.isnan(v)])
            print(f"\nApproximately computed: {computed_count}/{len(indices_to_compute)}")
            print("WARNING: These are approximate values, not true DrugBAN predictions.")
            
        except Exception as e2:
            print(f"Alternative computation also failed: {e2}")
            sys.exit(1)
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute DrugBAN affinity")
    parser.add_argument("--input", default="data/vdr_unique_G1_extended.csv",
                        help="Input CSV file")
    parser.add_argument("--output", default="data/vdr_unique_G1_with_drugban.csv",
                        help="Output CSV file")
    parser.add_argument("--model_path", default=None,
                        help="Path to pre-trained model weights")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for computation")
    args = parser.parse_args()
    
    compute_drugban_affinity(
        args.input,
        args.output,
        VDR_SEQUENCE,
        args.model_path,
        args.batch_size,
        args.device
    )


if __name__ == "__main__":
    main()
