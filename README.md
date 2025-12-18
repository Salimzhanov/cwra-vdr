# CWRA: Calibrated Weighted Rank Aggregation for VDR Virtual Screening

A robust machine learning framework for combining multiple molecular docking and binding affinity prediction modalities to improve virtual screening performance for Vitamin D Receptor (VDR) ligands.

## Features

- **Multi-modal fusion**: Combines predictions from 9 different computational methods
- **Optimized weighting**: Uses nested cross-validation to learn optimal modality weights
- **Stability-focused**: Reduces overfitting through objective functions and CV design
- **Comprehensive evaluation**: Provides detailed performance metrics across multiple enrichment factors

## Modalities

The framework combines predictions from the following computational methods:

### GraphDTA Kd/Ki/IC50
**Description**: GraphDTA is a deep learning framework for drug-target affinity prediction using graph neural networks. It predicts binding affinities (Kd, Ki) and inhibition constants (IC50).

**Computation**: Models trained on binding affinity datasets. Kd and Ki are dissociation constants (lower = better binding), IC50 is inhibition concentration (lower = better).

**Installation**: Pre-trained models available via GraphDTA repository. Predictions computed offline.

**Parameters**: Default model architectures with graph convolution layers.

**Direction**: Kd/Ki: lower values better (negative oriented); IC50: lower values better (negative oriented).

### MLT-LE pKd
**Description**: Multi-task learning framework for drug-target affinity prediction, predicts pKd values (-log10(Kd)).

**Computation**: Multi-task neural network trained on multiple affinity datasets.

**Installation**: Available via MLT-LE repository.

**Parameters**: Multi-task architecture with shared and task-specific layers.

**Direction**: Higher pKd values better (positive oriented).

### AutoDock Vina
**Description**: Molecular docking software that predicts binding poses and scores ligands against protein structures.

**Computation**: Docking simulations using protein-ligand interaction scoring functions.

**Installation**: Open-source software available at autodock.scripps.edu.

**Parameters**: Default scoring function, exhaustiveness=8, energy_range=3.

**Direction**: Lower docking scores better (negative oriented).

### Boltz-2 Affinity/Confidence
**Description**: Physics-based molecular dynamics simulation for binding free energy prediction.

**Computation**: MD simulations with enhanced sampling to compute binding affinities.

**Installation**: Boltz-2 software suite.

**Parameters**: Default simulation parameters with 10ns production runs.

**Direction**: Affinity: lower values better (negative oriented); Confidence: higher values better (positive oriented).

### Uni-Mol Similarity
**Description**: Universal molecular representation learning framework for computing molecular similarities to known actives.

**Computation**: Transformer-based model trained on large molecular datasets, computes similarity scores to reference active compounds.

**Installation**: Available via Uni-Mol repository.

**Parameters**: Pre-trained model with 12-layer transformer.

**Direction**: Higher similarity scores better (positive oriented).

### TankBind Affinity
**Description**: Structure-based drug design tool for binding affinity prediction using geometric deep learning.

**Computation**: Predicts binding poses and affinities using 3D geometric learning on protein-ligand complexes.

**Installation**: Available via TankBind repository.

**Parameters**: Default model with geometric attention layers.

**Direction**: Lower affinity values better (negative oriented).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cwra-vdr-benchmark.git
cd cwra-vdr-benchmark

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- rdkit

## Usage

### Basic Run

```bash
python cwra.py --csv labeled_raw_modalities_with_tankbind.csv --focus early
```

### Advanced Options

```bash
python cwra.py \
    --csv labeled_raw_modalities_with_tankbind.csv \
    --outer_repeats 5 \
    --outer_splits 10 \
    --focus early \
    --aggregation weighted \
    --output_prefix my_results
```

### Command Line Arguments

- `--csv`: Path to input CSV with modalities and SMILES
- `--outer_splits`: Number of outer CV folds (default: 10)
- `--outer_repeats`: Number of CV repeats (default: 5)
- `--focus`: Optimization focus - 'early', 'balanced', or 'standard'
- `--aggregation`: Aggregation method - 'weighted', 'rrf', or 'power'
- `--output_prefix`: Prefix for output files

## Input Data Format

The input CSV should contain:
- `smiles`: SMILES strings for molecules
- `source`: Source identifier (e.g., 'initial_370' for actives)
- Modality columns: `graphdta_kd`, `graphdta_ki`, `graphdta_ic50`, `mltle_pKd`, `vina_score`, `boltz_affinity`, `boltz_confidence`, `unimol_similarity`, `tankbind_affinity`

## Output Files

- `{prefix}_table5_weights.csv`: Modality weights and individual performance
- `{prefix}_table6_performance.csv`: Comprehensive performance comparison
- `{prefix}_full_ranking.csv`: Complete ranking of all compounds
- `{prefix}_top{top_n}_G.csv`: Top generated compounds
- `{prefix}_bottom{top_n}_G.csv`: Bottom generated compounds

## Modalities

The framework combines predictions from:

1. **GraphDTA Kd/Ki/IC50**: Drug target affinity binding predictions
2. **MLTLE pKd**: Multi-task learning binding predictions
3. **AutoDock Vina**: Molecular docking free binding energy
4. **Boltz-1/2**: Physics-based binding affinity and confidence
5. **Uni-Mol Similarity**: Molecular similarity to known actives
6. **TankBind**: Structure-based binding affinity

## Performance Metrics

- **EF@k%**: Enrichment Factor at k% of database
- **Hits@k**: Number of actives in top k compounds
- **BEDROC**: Boltzmann-Enhanced Discrimination of ROC
- **Mean Rank**: Average rank of active compounds

## Citation

If you use this code, please cite

## License

MIT License - see LICENSE file for details.