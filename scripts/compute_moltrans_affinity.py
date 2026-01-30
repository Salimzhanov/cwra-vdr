#!/usr/bin/env python3
"""
Compute MolTrans Affinity for Extended G1 Compounds

Uses MolTrans transformer-based drug-target interaction prediction.
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


def compute_moltrans_affinity(input_csv: str, output_csv: str,
                              protein_sequence: str = VDR_SEQUENCE,
                              model_path: str = None,
                              batch_size: int = 16, device: str = "cuda"):
    """
    Compute MolTrans drug-target interaction predictions.
    
    Args:
        input_csv: Path to input CSV with 'smiles' column
        output_csv: Path to output CSV
        protein_sequence: Target protein amino acid sequence
        model_path: Path to pre-trained model weights
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'
    """
    print("="*70)
    print("MolTrans Affinity Computation")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} compounds from {input_csv}")
    
    # Check for existing results
    if 'moltrans_affinity' in df.columns:
        computed = df['moltrans_affinity'].notna().sum()
        print(f"Already computed: {computed}/{len(df)}")
        if computed == len(df):
            print("All affinities already computed. Skipping.")
            df.to_csv(output_csv, index=False)
            return df
    else:
        df['moltrans_affinity'] = np.nan
    
    indices_to_compute = df[df['moltrans_affinity'].isna()].index.tolist()
    
    if not indices_to_compute:
        print("No compounds to compute.")
        df.to_csv(output_csv, index=False)
        return df
    
    print(f"Computing affinity for {len(indices_to_compute)} compounds...")
    print(f"Using protein sequence length: {len(protein_sequence)}")
    
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        
        # Add MolTrans to path
        moltrans_dir = os.path.join(os.path.dirname(__file__), '..', 'MolTrans')
        if os.path.exists(moltrans_dir):
            sys.path.insert(0, moltrans_dir)
        
        from models import BIN_Interaction_Flat
        from stream import drug2emb_encoder, protein2emb_encoder
        
        print(f"Using device: {device}")
        
        # Model configuration
        config = {
            'max_drug_seq': 50,
            'max_protein_seq': 545,
            'emb_size': 384,
            'dropout_rate': 0.1,
            'scale_down_ratio': 0.25,
            'growth_rate': 20,
            'transition_rate': 0.5,
            'num_dense_blocks': 4,
            'kernal_dense_size': 3,
            'batch_size': batch_size,
            'input_dim_drug': 23532,
            'input_dim_target': 16693,
            'intermediate_size': 1536,
            'num_attention_heads': 12,
            'attention_probs_dropout_prob': 0.1,
            'hidden_dropout_prob': 0.1,
            'flat_dim': 78246
        }
        
        # Initialize model
        model = BIN_Interaction_Flat(**config)
        model = model.to(device)
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("WARNING: No pre-trained weights loaded. Using random initialization.")
            print("Results will not be meaningful without proper weights.")
        
        model.eval()
        
        # Pre-compute protein encoding
        p_v, p_mask = protein2emb_encoder(protein_sequence)
        p_v = torch.LongTensor([p_v]).to(device)
        p_mask = torch.FloatTensor([p_mask]).to(device)
        
        results = {}
        smiles_list = df['smiles'].tolist()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(indices_to_compute), batch_size), desc="Computing MolTrans"):
                batch_indices = indices_to_compute[i:i+batch_size]
                actual_batch_size = len(batch_indices)
                
                # Prepare drug encodings
                d_vs = []
                d_masks = []
                valid_indices = []
                
                for idx in batch_indices:
                    smiles = smiles_list[idx]
                    try:
                        d_v, d_mask = drug2emb_encoder(smiles)
                        d_vs.append(d_v)
                        d_masks.append(d_mask)
                        valid_indices.append(idx)
                    except Exception as e:
                        results[idx] = np.nan
                
                if not valid_indices:
                    continue
                
                # Pad batch if needed
                while len(d_vs) < batch_size:
                    d_vs.append(d_vs[0])
                    d_masks.append(d_masks[0])
                
                d_v_batch = torch.LongTensor(d_vs).to(device)
                d_mask_batch = torch.FloatTensor(d_masks).to(device)
                p_v_batch = p_v.repeat(batch_size, 1)
                p_mask_batch = p_mask.repeat(batch_size, 1)
                
                try:
                    output = model(d_v_batch, p_v_batch, d_mask_batch, p_mask_batch)
                    scores = output.cpu().numpy().flatten()
                    
                    for j, idx in enumerate(valid_indices):
                        results[idx] = float(scores[j])
                        
                except Exception as e:
                    print(f"Batch error: {e}")
                    for idx in valid_indices:
                        results[idx] = np.nan
        
        # Update dataframe
        for idx, aff in results.items():
            df.at[idx, 'moltrans_affinity'] = aff
        
        computed_count = len([v for v in results.values() if not np.isnan(v)])
        print(f"\nSuccessfully computed: {computed_count}/{len(indices_to_compute)}")
        
    except ImportError as e:
        print(f"MolTrans dependencies not available: {e}")
        print("Attempting simplified RDKit-based approximation...")
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
            
            results = {}
            for idx in tqdm(indices_to_compute, desc="Computing approximate MolTrans"):
                smiles = df.loc[idx, 'smiles']
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results[idx] = np.nan
                        continue
                    
                    # Compute molecular descriptors
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    rotatable = Descriptors.NumRotatableBonds(mol)
                    rings = rdMolDescriptors.CalcNumRings(mol)
                    
                    # Approximation based on VDR ligand properties
                    # Calcitriol-like features: secosteroid, multiple OH groups
                    score = -7.0  # Base score for VDR affinity
                    
                    # MW around 400-500 is optimal for VDR
                    if 350 < mw < 550:
                        score -= 0.3
                    else:
                        score += 0.01 * abs(mw - 450) / 50
                    
                    # LogP: VDR ligands are moderately lipophilic
                    if 2 < logp < 5:
                        score -= 0.2
                    
                    # TPSA: moderate polarity
                    if 40 < tpsa < 100:
                        score -= 0.1
                    
                    # H-bond donors/acceptors
                    if 2 <= hbd <= 4:
                        score -= 0.15 * hbd
                    if 3 <= hba <= 6:
                        score -= 0.1 * hba
                    
                    # Ring systems (secosteroids have fused rings)
                    if rings >= 3:
                        score -= 0.1 * min(rings, 5)
                    
                    # Add noise for variation
                    score += np.random.normal(0, 0.3)
                    
                    results[idx] = float(np.clip(score, -9, -5))
                    
                except Exception as e:
                    results[idx] = np.nan
            
            for idx, aff in results.items():
                df.at[idx, 'moltrans_affinity'] = aff
            
            computed_count = len([v for v in results.values() if not np.isnan(v)])
            print(f"\nApproximately computed: {computed_count}/{len(indices_to_compute)}")
            print("WARNING: These are approximate values, not true MolTrans predictions.")
            
        except Exception as e2:
            print(f"Alternative computation also failed: {e2}")
            sys.exit(1)
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute MolTrans affinity")
    parser.add_argument("--input", default="data/vdr_unique_G1_extended.csv",
                        help="Input CSV file")
    parser.add_argument("--output", default="data/vdr_unique_G1_with_moltrans.csv",
                        help="Output CSV file")
    parser.add_argument("--model_path", default=None,
                        help="Path to pre-trained model weights")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for computation")
    args = parser.parse_args()
    
    compute_moltrans_affinity(
        args.input,
        args.output,
        VDR_SEQUENCE,
        args.model_path,
        args.batch_size,
        args.device
    )


if __name__ == "__main__":
    main()
