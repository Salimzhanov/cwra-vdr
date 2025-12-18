# CWRA: Calibrated Weighted Rank Aggregation for VDR Virtual Screening

[![CI](https://github.com/Salimzhanov/cwra-vdr/actions/workflows/ci.yml/badge.svg)](https://github.com/Salimzhanov/cwra-vdr/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/cwra-vdr.svg)](https://pypi.org/project/cwra-vdr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A robust machine learning framework for combining multiple molecular docking and binding affinity prediction modalities to improve virtual screening performance for Vitamin D Receptor (VDR) ligands.

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
- Modality columns: `graphdta_kd`, `graphdta_ki`, `graphdta_ic50`, `mltle_pKd`, `vina_score`, `boltz_affinity`, `boltz_confidence`, `unimol_similarity`, `tankbind_affinity`

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
  url={https://github.com/Salimzhanov/cwra-vdr-toolbox},
  version={1.0.0}
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
  url={https://github.com/Salimzhanov/cwra-vdr-toolbox},
  version={1.0.0},
  orcid={
    0000-0001-6630-585X and
    0000-0001-9008-4233 and
    0000-0003-3397-0647
  }
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.