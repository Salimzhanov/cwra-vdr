# Data Directory

This directory contains the input datasets and precomputed embeddings for the
CWRA virtual screening pipeline.

## Files

| File | Description | Rows | Key Columns |
|------|-------------|------|-------------|
| `composed_modalities_with_rdkit.csv` | Main dataset with all 11 modality scores and RDKit descriptors | 16,059 | `smiles`, `source`, `generator`, modality columns |
| `composed_modalities.csv` | Dataset without RDKit descriptors | 16,059 | `smiles`, `source`, modality columns |
| `composed_modalities_tankbind.csv` | Dataset variant with TankBind scores | 16,059 | `smiles`, `tankbind` |
| `labeled_raw_modalities.csv` | Raw modality scores before composition | 16,059 | `smiles`, `source`, individual modalities |
| `labeled_raw_modalities_extended.csv` | Extended raw modalities | 16,059 | As above, extended |
| `cwra.csv` | Pre-computed CWRA rankings | 16,059 | `smiles`, `cwra_score`, `cwra_rank` |
| `final_selected.csv` | PU-conformal selected compounds | ~50–200 | `smiles`, `p_value`, `source` |
| `unimol_embeddings.npz` | Uni-Mol 3D molecular embeddings (NumPy compressed) | 16,059 | Array of shape `(N, D)` |
| `unimol_embeddings_failed_smiles.txt` | SMILES that failed Uni-Mol embedding | — | One SMILES per line |
| `VDR_Molecular_Chemistry_Guide.md` | Background on VDR molecular chemistry | — | Markdown reference |

## Modality Columns

The 11 modality score columns in `composed_modalities_with_rdkit.csv`:

| Column | Direction | Description |
|--------|-----------|-------------|
| `graphdta_kd` | Higher = better | GraphDTA dissociation constant prediction |
| `graphdta_ki` | Higher = better | GraphDTA inhibition constant prediction |
| `graphdta_ic50` | Higher = better | GraphDTA IC50 prediction |
| `mltle_pKd` | Higher = better | MLT-LE binding affinity (pKd) |
| `vina_score` | Lower = better | AutoDock Vina docking score |
| `boltz_affinity` | Lower = better | Boltz-2 binding affinity |
| `boltz_confidence` | Higher = better | Boltz-2 confidence score |
| `unimol_similarity` | Higher = better | Uni-Mol similarity to known actives |
| `tankbind_affinity` | Lower = better | TankBind binding affinity |
| `drugban_affinity` | Lower = better | DrugBAN interaction prediction |
| `moltrans_affinity` | Lower = better | MolTrans interaction prediction |

## Source Labels

The `source` column indicates compound origin:

| Source | Description |
|--------|-------------|
| `initial_370` | Known VDR active ligands (366 after deduplication) |
| `calcitriol` | Reference VDR ligand (1,25-dihydroxyvitamin D3) |
| `G1` | Single-model generated candidates |
| `G2` | Two-model consensus generated candidates |
| `G3` | Three-model consensus generated candidates |

## Provenance

- Modality scores were computed using the scripts in `scripts/compute_*.py`
- Uni-Mol embeddings were generated with `unimol_embeddings.py`
- The VDR crystal structure used is PDB [1DB1](https://www.rcsb.org/structure/1DB1)
- Active compounds were curated from published VDR binding data

## Usage

```python
import pandas as pd

df = pd.read_csv("data/composed_modalities_with_rdkit.csv")
print(f"Compounds: {len(df)}")
print(f"Actives:   {(df['source'] == 'initial_370').sum()}")
print(f"Columns:   {list(df.columns)}")
```
