#!/usr/bin/env python3
"""
Compute All Remaining Modalities for Extended Dataset

This script computes:
1. Merges boltz_affinity and boltz_confidence from G1 source
2. Computes MolTrans affinity using the local MolTrans model
3. Computes DrugBAN affinity if available
4. Computes TankBind affinity if structures are available
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
CWRA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CWRA_DIR, 'data')
G1_SOURCE = "c:/Users/abyla/Downloads/weighted rank/weighted rank/vdr_unique_G1.csv"

# VDR protein sequence for DTI models
VDR_SEQUENCE = "MEAMAASTSLPDPGDFDRNVPRICGVCGDRATGFHFNAMTCEGCKGFFRRSMKRKALFTCPFNGDCRITKDNRRHCQACRLKRCVDIGMMKEFILTDEEVQRKREMILKRKEEEALKDSLRPKLSEEQQRIIAILLDAHHKTYDPTYSDFCQFRPPVRVNDGGGSHPSRPNSRHTPSFSGDSSSSCSDHCITSSDMMDSSSFSNLDLSEEDSDDPSVTLELSQLSMLPHLADLVSYSIQKVIGFAKMIPGFRDLTSEDQIVLLKSSAIEVIMLRSNESFTMDDMSWTCGNQDYKYRVSDVTKAGHSLELIEPLIKFQVGLKKLNLHEEEHVLLMAICIVSPDRPGVQDAALIEAIQDRLSNTLQTYIRCRHPPPGSHLLYAKMIQKLADLRSLNEEHSKQYRCLSFQPECSMKLTPLVLEVFGNEIS"


def merge_boltz_from_source(main_df: pd.DataFrame) -> pd.DataFrame:
    """Merge boltz_affinity and boltz_confidence from G1 source file."""
    print("\n" + "="*60)
    print("Merging Boltz data from G1 source")
    print("="*60)
    
    if not os.path.exists(G1_SOURCE):
        print(f"G1 source file not found: {G1_SOURCE}")
        return main_df
    
    g1_df = pd.read_csv(G1_SOURCE)
    print(f"Loaded G1 source: {len(g1_df)} compounds")
    
    # Create SMILES to boltz mapping
    boltz_aff_map = dict(zip(g1_df['smiles'], g1_df['boltz_affinity_pred_value']))
    boltz_conf_map = dict(zip(g1_df['smiles'], g1_df['boltz_confidence_score']))
    
    # Update main dataframe
    updated_aff = 0
    updated_conf = 0
    
    for idx, row in main_df.iterrows():
        smiles = row['smiles']
        
        # Update boltz_affinity if missing
        if pd.isna(row['boltz_affinity']) and smiles in boltz_aff_map:
            val = boltz_aff_map[smiles]
            if pd.notna(val):
                main_df.at[idx, 'boltz_affinity'] = val
                updated_aff += 1
        
        # Update boltz_confidence if missing
        if pd.isna(row['boltz_confidence']) and smiles in boltz_conf_map:
            val = boltz_conf_map[smiles]
            if pd.notna(val):
                main_df.at[idx, 'boltz_confidence'] = val
                updated_conf += 1
    
    print(f"Updated boltz_affinity: {updated_aff}")
    print(f"Updated boltz_confidence: {updated_conf}")
    print(f"Total boltz_affinity coverage: {main_df['boltz_affinity'].notna().sum()}/{len(main_df)}")
    
    return main_df


def compute_moltrans_affinity(main_df: pd.DataFrame, batch_size: int = 32, device: str = "cuda") -> pd.DataFrame:
    """Compute MolTrans affinity predictions."""
    print("\n" + "="*60)
    print("Computing MolTrans Affinity")
    print("="*60)
    
    # Check for missing values
    missing_mask = main_df['moltrans_affinity'].isna()
    indices_to_compute = main_df[missing_mask].index.tolist()
    
    if not indices_to_compute:
        print("All MolTrans affinities already computed.")
        return main_df
    
    print(f"Need to compute: {len(indices_to_compute)} compounds")
    
    try:
        import torch
        
        # Change to MolTrans directory for proper imports
        moltrans_dir = os.path.join(CWRA_DIR, 'MolTrans')
        original_dir = os.getcwd()
        os.chdir(moltrans_dir)
        sys.path.insert(0, moltrans_dir)
        
        from models import BIN_Interaction_Flat
        from stream import drug2emb_encoder, protein2emb_encoder
        
        print(f"Using device: {device}")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA not available, using CPU")
        
        # Model configuration (from MolTrans config)
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
        
        # Check for pre-trained weights
        weight_paths = [
            os.path.join(moltrans_dir, 'best_model.pt'),
            os.path.join(moltrans_dir, 'model.pt'),
            os.path.join(moltrans_dir, 'moltrans.pt'),
        ]
        
        weights_loaded = False
        for wp in weight_paths:
            if os.path.exists(wp):
                print(f"Loading weights from {wp}")
                model.load_state_dict(torch.load(wp, map_location=device))
                weights_loaded = True
                break
        
        if not weights_loaded:
            print("WARNING: No pre-trained weights found. Using random initialization.")
            print("Results may not be meaningful.")
        
        model.eval()
        
        # Pre-compute protein encoding
        p_v, p_mask = protein2emb_encoder(VDR_SEQUENCE)
        p_v = torch.LongTensor([p_v]).to(device)
        p_mask = torch.FloatTensor([p_mask]).to(device)
        
        smiles_list = main_df['smiles'].tolist()
        results = {}
        
        with torch.no_grad():
            for i in tqdm(range(0, len(indices_to_compute), batch_size), desc="MolTrans"):
                batch_indices = indices_to_compute[i:i+batch_size]
                
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
                
                actual_batch_size = len(valid_indices)
                
                d_v_batch = torch.LongTensor(d_vs).to(device)
                d_mask_batch = torch.FloatTensor(d_masks).to(device)
                p_v_batch = p_v.repeat(actual_batch_size, 1)
                p_mask_batch = p_mask.repeat(actual_batch_size, 1)
                
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
        for idx, score in results.items():
            main_df.at[idx, 'moltrans_affinity'] = score
        
        os.chdir(original_dir)
        
        computed = len([v for v in results.values() if pd.notna(v)])
        print(f"Successfully computed: {computed}/{len(indices_to_compute)}")
        print(f"Total coverage: {main_df['moltrans_affinity'].notna().sum()}/{len(main_df)}")
        
    except Exception as e:
        print(f"MolTrans computation failed: {e}")
        import traceback
        traceback.print_exc()
        if 'original_dir' in dir():
            os.chdir(original_dir)
    
    return main_df


def compute_drugban_affinity(main_df: pd.DataFrame, batch_size: int = 32, device: str = "cuda") -> pd.DataFrame:
    """Compute DrugBAN affinity predictions."""
    print("\n" + "="*60)
    print("Computing DrugBAN Affinity")
    print("="*60)
    
    # Check for missing values
    missing_mask = main_df['drugban_affinity'].isna()
    indices_to_compute = main_df[missing_mask].index.tolist()
    
    if not indices_to_compute:
        print("All DrugBAN affinities already computed.")
        return main_df
    
    print(f"Need to compute: {len(indices_to_compute)} compounds")
    
    try:
        import torch
        
        drugban_dir = os.path.join(CWRA_DIR, 'DrugBAN')
        original_dir = os.getcwd()
        os.chdir(drugban_dir)
        sys.path.insert(0, drugban_dir)
        
        from models import DrugBAN
        from utils import graph_collate_func
        from dataloader import DTIDataset
        
        print(f"Using device: {device}")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA not available, using CPU")
        
        # Initialize model with default config
        model = DrugBAN()
        model = model.to(device)
        
        # Check for pre-trained weights
        weight_paths = [
            os.path.join(drugban_dir, 'best_model.pt'),
            os.path.join(drugban_dir, 'model.pt'),
            os.path.join(drugban_dir, 'drugban.pt'),
        ]
        
        weights_loaded = False
        for wp in weight_paths:
            if os.path.exists(wp):
                print(f"Loading weights from {wp}")
                checkpoint = torch.load(wp, map_location=device)
                model.load_state_dict(checkpoint)
                weights_loaded = True
                break
        
        if not weights_loaded:
            print("WARNING: No pre-trained weights found. Skipping DrugBAN.")
            os.chdir(original_dir)
            return main_df
        
        model.eval()
        smiles_list = main_df['smiles'].tolist()
        results = {}
        
        # Simplified inference without full dataset setup
        from rdkit import Chem
        from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
        
        atom_featurizer = CanonicalAtomFeaturizer()
        bond_featurizer = CanonicalBondFeaturizer()
        
        with torch.no_grad():
            for idx in tqdm(indices_to_compute, desc="DrugBAN"):
                smiles = smiles_list[idx]
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        results[idx] = np.nan
                        continue
                    
                    graph = mol_to_bigraph(mol, 
                                          node_featurizer=atom_featurizer,
                                          edge_featurizer=bond_featurizer)
                    
                    # This would need proper batching with protein features
                    # Simplified placeholder
                    results[idx] = np.nan
                    
                except Exception as e:
                    results[idx] = np.nan
        
        os.chdir(original_dir)
        
    except ImportError as e:
        print(f"DrugBAN import failed: {e}")
        print("DrugBAN requires dgl and dgllife which have compatibility issues.")
    except Exception as e:
        print(f"DrugBAN computation failed: {e}")
        if 'original_dir' in dir():
            os.chdir(original_dir)
    
    return main_df


def compute_tankbind_affinity(main_df: pd.DataFrame, pdb_dir: str = None) -> pd.DataFrame:
    """Compute TankBind affinity from structure predictions."""
    print("\n" + "="*60)
    print("Computing TankBind Affinity")
    print("="*60)
    
    # Check for missing values
    missing_mask = main_df['tankbind_affinity'].isna()
    indices_to_compute = main_df[missing_mask].index.tolist()
    
    if not indices_to_compute:
        print("All TankBind affinities already computed.")
        return main_df
    
    print(f"Need to compute: {len(indices_to_compute)} compounds")
    
    if pdb_dir is None:
        pdb_dir = os.path.join(CWRA_DIR, 'pdb', 'flexible_complexes', 'vdr_unique_G1')
    
    if not os.path.exists(pdb_dir):
        print(f"PDB directory not found: {pdb_dir}")
        return main_df
    
    try:
        # TankBind typically outputs affinity predictions during docking
        # Check if there are pre-computed results in PDB files or accompanying data
        
        # Look for tankbind output files
        tankbind_results = {}
        
        for pdb_file in os.listdir(pdb_dir):
            if pdb_file.endswith('.pdb'):
                # Try to extract affinity from PDB header or filename
                pdb_path = os.path.join(pdb_dir, pdb_file)
                # This is a placeholder - actual implementation depends on TankBind output format
                pass
        
        print("TankBind requires pre-computed docking results or running the TankBind pipeline.")
        print("Skipping TankBind computation for now.")
        
    except Exception as e:
        print(f"TankBind computation failed: {e}")
    
    return main_df


def main():
    """Main function to compute all modalities."""
    print("="*70)
    print("Computing All Remaining Modalities")
    print("="*70)
    
    # Load main data
    main_csv = os.path.join(DATA_DIR, 'labeled_raw_modalities.csv')
    print(f"\nLoading data from {main_csv}")
    main_df = pd.read_csv(main_csv)
    print(f"Loaded {len(main_df)} compounds")
    
    # Initial status
    print("\nInitial modality coverage:")
    for col in ['boltz_affinity', 'boltz_confidence', 'tankbind_affinity', 
                'drugban_affinity', 'moltrans_affinity']:
        coverage = main_df[col].notna().sum()
        print(f"  {col}: {coverage}/{len(main_df)} ({100*coverage/len(main_df):.1f}%)")
    
    # 1. Merge Boltz data from source
    main_df = merge_boltz_from_source(main_df)
    
    # 2. Compute MolTrans
    main_df = compute_moltrans_affinity(main_df, batch_size=32, device="cuda")
    
    # 3. Attempt DrugBAN (likely to fail due to DGL issues)
    main_df = compute_drugban_affinity(main_df, batch_size=32, device="cuda")
    
    # 4. Attempt TankBind
    main_df = compute_tankbind_affinity(main_df)
    
    # Save results
    main_df.to_csv(main_csv, index=False)
    print(f"\nResults saved to {main_csv}")
    
    # Final status
    print("\nFinal modality coverage:")
    for col in ['boltz_affinity', 'boltz_confidence', 'tankbind_affinity', 
                'drugban_affinity', 'moltrans_affinity']:
        coverage = main_df[col].notna().sum()
        print(f"  {col}: {coverage}/{len(main_df)} ({100*coverage/len(main_df):.1f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
