# CWRA: Calibrated Weighted Rank Aggregation for VDR Virtual Screening

[![CI](https://github.com/Salimzhanov/cwra-vdr/actions/workflows/ci.yml/badge.svg)](https://github.com/Salimzhanov/cwra-vdr/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/cwra-vdr.svg)](https://pypi.org/project/cwra-vdr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A robust machine learning framework for combining multiple molecular docking and binding affinity prediction modalities to improve virtual screening performance for Vitamin D Receptor (VDR) ligands. The framework now supports 11 modalities including traditional docking scores, deep learning-based affinity predictions, and similarity-based methods.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modalities](#modalities)
- [Performance Metrics](#performance-metrics)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### From PyPI (Recommended)

```bash
pip install cwra-vdr
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Salimzhanov/cwra-vdr.git
cd cwra-vdr

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- rdkit>=2021.03.1

## Quick Start

### Basic Usage

```python
from cwra import main
import sys

# Run with default parameters
sys.argv = ['cwra', '--csv', 'data/labeled_raw_modalities.csv', '--focus', 'early']
main()
```

### Command Line

```bash
# Quick test run
cwra --csv data/labeled_raw_modalities.csv --outer_repeats 1 --outer_splits 3 --focus early

# Full production run
cwra --csv data/labeled_raw_modalities.csv --outer_repeats 5 --outer_splits 10 --focus early --output_prefix results
```

### Python API

```python
import pandas as pd
from cwra import CWRA

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize CWRA
cwra = CWRA(
    modalities=['graphdta_kd', 'vina_score', 'boltz_affinity'],
    focus='early',
    n_repeats=3,
    n_splits=5
)

# Run the analysis
results = cwra.fit_predict(df)
print(results.summary())
```

## Project Structure

```
cwra-vdr/
├── cwra/                          # Main package
│   ├── __init__.py
│   └── cwra.py                   # Core implementation
├── src/                           # Alternative modular implementation
│   └── cwra/
│       ├── io.py                 # Data I/O utilities
│       ├── metrics.py            # Evaluation metrics
│       ├── weighting.py          # Weight optimization
│       ├── aggregation.py        # Rank aggregation methods
│       ├── cv.py                 # Cross-validation utilities
│       └── scaffold.py           # Scaffold-based evaluation
├── analysis/                      # Analysis and comparison scripts
│   ├── analyze_vdr_ranking_modalities.py
│   ├── generate_vdr_comparison_tables.py
│   ├── create__vdr_table.py
│   └── cwra_reduce_variance_10_20_30_fixed_all7modalities.py
├── code_results/                  # Optimized implementations
│   └── cwra_improved.py          # Reduced hyperparameter grid
├── data/                          # Example datasets
│   └── labeled_raw_modalities.csv
├── examples/                      # Example scripts
│   ├── basic_example.py
│   ├── advanced_example.py
│   └── compare_aggregation.py
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
├── .github/                       # GitHub configuration
│   ├── workflows/
│   │   └── ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE/
├── tests/                         # Unit tests
├── pyproject.toml                 # Modern Python packaging
├── setup.py                       # Legacy packaging
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── CONTRIBUTING.md                # Contribution guidelines
├── CHANGELOG.md                   # Version history
└── LICENSE                        # MIT License
```

**Note**: The repository also includes `TankBind/` - a separate repository for structure-based binding affinity prediction that provides the TankBind modality values used in CWRA analysis.

**Computation**: MD simulations with binding affinities.

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

### DrugBAN Affinity
**Description**: Graph-based drug-target affinity prediction using Deep Graph Neural Networks.

**Computation**: Combines molecular graph representations with character-level SMILES encoding and molecular descriptors for binding affinity prediction.

**Installation**: Available via DrugBAN repository with DGL and dgllife dependencies.

**Parameters**: Pre-trained model with graph attention layers and molecular descriptors.

**Direction**: Lower affinity values better (negative oriented).

### MolTrans Affinity
**Description**: Transformer-based drug-target interaction prediction using subword tokenization.

**Computation**: Uses Byte-Pair Encoding (BPE) for SMILES and protein sequences, processed through multi-head attention transformer.

**Installation**: Available via MolTrans repository with subword-nmt for BPE encoding.

**Parameters**: Pre-trained transformer with BPE vocabulary files.

**Direction**: Lower affinity values better (negative oriented).

## Performance Results

The CWRA framework has been evaluated on an extended dataset with 11 modalities including the newly integrated DrugBAN and MolTrans methods. Performance results are reported as mean ± standard deviation across 3-fold CV repeats with scaffold-based grouping to prevent overfitting.

\begin{table*}[h]
\centering
\caption{Hit recovery under scaffold-grouped nested CV on initial\_370 actives. Values are mean $\pm$ std across 3-fold CV repeats. EF@k\% measures enrichment factor at top k\% of the ranked database.}
\label{tab:fusion_performance_scaffold}
\small
\begin{tabular}{@{}llccc@{}}
\toprule
Category & Method & EF@10\% & EF@20\% & EF@30\% \\
\midrule
\multirow{11}{*}{\rotatebox[origin=c]{90}{\textit{Single Modality}}}
 & GraphDTA-$K_\mathrm{d}$\textsuperscript{$\downarrow$} & 1.76 $\pm$ 0.71 & 1.65 $\pm$ 0.34 & 1.50 $\pm$ 0.27 \\
 & GraphDTA-$K_\mathrm{i}$\textsuperscript{$\downarrow$} & 1.87 $\pm$ 0.29 & 1.61 $\pm$ 0.16 & 1.56 $\pm$ 0.37 \\
 & GraphDTA-IC$_{50}$\textsuperscript{$\downarrow$} & 1.85 $\pm$ 0.51 & 1.66 $\pm$ 0.52 & 1.55 $\pm$ 0.36 \\
 & MLT-LE p$K_\mathrm{d}$\textsuperscript{$\downarrow$} & 1.47 $\pm$ 0.49 & 1.35 $\pm$ 0.18 & 1.38 $\pm$ 0.15 \\
 & AutoDock Vina\textsuperscript{$\downarrow$} & 0.53 $\pm$ 0.15 & 0.74 $\pm$ 0.14 & 0.80 $\pm$ 0.23 \\
 & Boltz-2 affinity\textsuperscript{$\downarrow$} & 1.66 $\pm$ 0.39 & 1.61 $\pm$ 0.36 & 1.43 $\pm$ 0.27 \\
 & Boltz-2 confidence\textsuperscript{$\uparrow$} & 1.74 $\pm$ 0.47 & 1.74 $\pm$ 0.38 & 1.57 $\pm$ 0.17 \\
 & Uni-Mol similarity\textsuperscript{$\uparrow$} & 1.94 $\pm$ 1.35 & 1.76 $\pm$ 0.88 & 1.54 $\pm$ 0.61 \\
 & TankBind affinity\textsuperscript{$\downarrow$} & 0.65 $\pm$ 0.35 & 1.08 $\pm$ 0.38 & 1.05 $\pm$ 0.42 \\
 & DrugBAN\textsuperscript{$\downarrow$} & 1.10 $\pm$ 0.65 & 1.04 $\pm$ 0.32 & 1.00 $\pm$ 0.17 \\
 & MolTrans\textsuperscript{$\downarrow$} & 1.36 $\pm$ 0.21 & 1.12 $\pm$ 0.17 & 1.14 $\pm$ 0.17 \\
\midrule
\multirow{2}{*}{\rotatebox[origin=c]{90}{\textit{Fusion}}}
 & Equal-weight & 1.84 $\pm$ 0.47 & 1.93 $\pm$ 0.45 & 1.62 $\pm$ 0.27 \\
 & CWRA-early & 1.98 $\pm$ 0.43 & 1.87 $\pm$ 0.39 & 1.67 $\pm$ 0.37 \\
\midrule
\multicolumn{2}{l}{\textit{Expected at random}} & 1.31 $\pm$ 0.12 & 1.17 $\pm$ 0.13 & 1.21 $\pm$ 0.12 \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\raggedright
\footnotesize
\textsuperscript{$\uparrow$}Higher values indicate stronger predicted binding. 
\textsuperscript{$\downarrow$}Lower (more negative) values indicate stronger predicted binding. 
\textbf{Bold} indicates best performance. CWRA: Calibrated Weighted Rank Aggregation.
\end{table*}

**Key Findings:**
- CWRA-early achieves strong performance with EF@10% = 1.98 ± 0.43, outperforming equal-weight fusion (1.84 ± 0.47)
- Among single modalities, UniMol similarity and GraphDTA variants show the strongest performance
- The framework focuses on stable EF@10% metrics for weight optimization, removing noisy early enrichment measures
- Calibrated weight optimization successfully integrates 11 diverse modalities with reduced hyperparameter complexity

## Modality Implementations

The CWRA toolbox includes multiple implementations for analyzing and comparing multiple molecular ranking modalities. The codebase provides several analysis and implementation scripts:

### Core Implementation (`src/cwra/`)

The `src/cwra/` directory contains the core modular implementation of CWRA algorithms:

- **`io.py`**: Data input/output utilities for loading and preprocessing modality data
- **`metrics.py`**: Virtual screening evaluation metrics (EF@k%, BEDROC, hits@k)
- **`weighting.py`**: CWRA weight optimization algorithms with correlation shrinkage
- **`aggregation.py`**: Rank aggregation methods (weighted, RRF, power transformation)
- **`cv.py`**: Cross-validation utilities for robust performance estimation
- **`scaffold.py`**: Scaffold-based grouping for unbiased evaluation

### Analysis Scripts (`analysis/`)

The `analysis/` directory contains comprehensive analysis tools:

- **`analyze_vdr_ranking_modalities.py`**: Statistical analysis of modality performance, correlation matrices, and distribution comparisons
- **`generate_vdr_comparison_tables.py`**: Detailed comparison tables and top-ranked compound analysis
- **`create_comprehensive_vdr_table.py`**: Comprehensive performance tables across all modalities
- **`cwra_reduce_variance_10_20_30_fixed_all7modalities.py`**: Variance reduction techniques for stable CWRA optimization
- **`vdr_glide_comprehensive_plot.py`**: Visualization tools for performance comparison

### Evaluation Metrics

The toolbox implements industry-standard virtual screening metrics:

- **Enrichment Factor (EF@k%)**: Ratio of actives found in top k% vs random selection
- **BEDROC (Boltzmann-Enhanced Discrimination of ROC)**: Early recognition metric with exponential weighting
- **Hits@k**: Raw count of actives in top k compounds
- **Mean Rank**: Average rank position of active compounds
- **Kendall Tau**: Rank correlation between modalities

### Weight Optimization

CWRA optimizes modality weights using:

- **Multi-objective optimization**: Balances EF, BEDROC, and mean rank metrics
- **Correlation shrinkage**: Penalizes redundant modalities based on Kendall tau
- **Risk aversion**: Balances expected performance vs variance
- **Regularization**: Prevents overfitting through uniform mixing

### Aggregation Methods

Three aggregation strategies are implemented:

- **Weighted Ranks**: Linear combination of normalized ranks
- **Reciprocal Rank Fusion (RRF)**: 1/(k+r) aggregation (k=60 default)
- **Power Transformation**: Non-linear rank combination with tunable exponents

### Alternative Implementations

The repository includes multiple CWRA implementations optimized for different use cases:

- **`cwra_improved.py`** (`code_results/`): Optimized version with reduced hyperparameter grid (1,728 vs 88,200 configurations) and empirically validated modality directions
- **`cwra.py`** (root): Full-featured implementation with CLI and all aggregation methods
- **`cwra/cwra.py`**: Modular package implementation suitable for integration

### Performance Baselines

The toolbox provides baseline comparisons:

- **Individual Modalities**: Performance of each ranking method alone
- **Random Ranking**: Random permutation baseline
- **Equal Weight**: Simple average across all modalities
- **CWRA Consensus**: Optimized weighted combination

## Usage

### Basic Run

```bash
python -m cwra --csv data/labeled_raw_modalities.csv --focus early
```

### Advanced Options

```bash
python -m cwra \
    --csv data/labeled_raw_modalities.csv \
    --outer_repeats 5 \
    --outer_splits 10 \
    --focus early \
    --aggregation weighted \
    --output_prefix my_results
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `labeled_raw_modalities.csv` | Input CSV with modalities + SMILES + source |
| `--outer_splits` | `10` | Number of outer CV folds |
| `--outer_repeats` | `5` | Number of outer CV repeats |
| `--seed` | `42` | Random seed for reproducibility |
| `--risk_beta` | `0.5` | Risk aversion parameter (mean - beta*std) |
| `--focus` | `early` | Optimization focus: 'early', 'balanced', or 'standard' |
| `--aggregation` | `weighted` | Aggregation method: 'weighted', 'rrf', or 'power' |
| `--output_prefix` | `cwrad` | Prefix for output files |
| `--top_n` | `25` | Number of top/bottom structures to extract |

### Input Data Format

The input CSV should contain:
- `smiles`: SMILES strings for molecules
- `source`: Source identifier (e.g., 'initial_370' for actives)
- Modality columns: `graphdta_kd`, `graphdta_ki`, `graphdta_ic50`, `mltle_pKd`, `vina_score`, `boltz_affinity`, `boltz_confidence`, `unimol_similarity`, `tankbind_affinity`, `drugban_affinity`, `moltrans_affinity`

### Output Files

- `{prefix}_table5_weights.csv`: Modality weights and individual performance
- `{prefix}_table6_performance.csv`: Comprehensive performance comparison
- `{prefix}_full_ranking.csv`: Complete ranking of all compounds
- `{prefix}_top{top_n}_G.csv`: Top generated compounds
- `{prefix}_bottom{top_n}_G.csv`: Bottom generated compounds

## Performance Metrics

- **EF@k%**: Enrichment Factor at k% of database (EF@1%, EF@5%, EF@10%, etc.)
- **Hits@k**: Number of actives in top k compounds
- **BEDROC**: Boltzmann-Enhanced Discrimination of ROC (early recognition metric)
- **Mean Rank**: Average rank of active compounds across all modalities

## API Reference

### Main Classes

#### `CWRA`

Main class for Calibrated Weighted Rank Aggregation.

**Parameters:**
- `modalities` (list): List of modality column names
- `focus` (str): Optimization focus ('early', 'balanced', 'standard')
- `aggregation` (str): Aggregation method ('weighted', 'rrf', 'power')
- `n_repeats` (int): Number of CV repeats
- `n_splits` (int): Number of CV folds
- `seed` (int): Random seed

**Methods:**
- `fit_predict(df)`: Run the complete CWRA analysis
- `summary()`: Return performance summary

### Utility Functions

- `compute_scaffold(smiles)`: Compute Murcko scaffold from SMILES
- `bedroc_from_x(x, alpha, A, N)`: Compute BEDROC score
- `reciprocal_rank_fusion(ranks, k=60.0)`: RRF aggregation

## Examples

See the `examples/` directory for complete usage examples:

- `basic_example.py`: Simple CWRA run
- `advanced_example.py`: Full production configuration
- `compare_aggregation.py`: Compare different aggregation methods

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Authors

- **Abylay Salimzhanov**  
  [ORCID: 0000-0001-6630-585X](https://orcid.org/0000-0001-6630-585X)

- **Ferdinand Molnár**   
  [ORCID: 0000-0001-9008-4233](https://orcid.org/0000-0001-9008-4233)

- **Siamac Fazli**  
  [ORCID: 0000-0003-3397-0647](https://orcid.org/0000-0003-3397-0647)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{salimzhanov2025cwra,
  title={CWRA: Calibrated Weighted Rank Aggregation for VDR Virtual Screening},
  author={Salimzhanov, Abylay and Molnár, Ferdinand and Fazli, Siamac},
  year={2025},
  url={https://github.com/Salimzhanov/cwra-vdr},
  version={1.1.0}
}
```

### BibTeX with ORCID

```bibtex
@software{salimzhanov2025cwra,
  title={CWRA: Calibrated Weighted Rank Aggregation for VDR Virtual Screening},
  author={
    Salimzhanov, Abylay and
    Molnár, Ferdinand and
    Fazli, Siamac
  },
  year={2025},
  url={https://github.com/Salimzhanov/cwra-vdr},
  version={1.1.0},
  orcid={
    0000-0001-6630-585X and
    0000-0001-9008-4233 and
    0000-0003-3397-0647
  }
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.