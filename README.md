# CWRA: Calibrated Weighted Rank Aggregation for VDR Virtual Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A framework for combining multiple molecular docking and binding affinity prediction methods to improve virtual screening performance for Vitamin D Receptor (VDR) ligands. Supports 11 modalities including docking scores, deep learning-based affinity predictions, and similarity-based methods.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modalities](#modalities)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### From Source

```bash
git clone https://github.com/Salimzhanov/cwra-vdr.git
cd cwra-vdr
pip install -e ".[dev]"
```

### Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- rdkit>=2021.03.1

## Quick Start

### Command Line

```bash
# Quick test (narrow grid, few folds)
python -m cwra --csv data/labeled_raw_modalities.csv --grid_mode narrow --outer_splits 3 --outer_repeats 1

# Standard run (optimal grid, 5×5 CV)
python -m cwra --csv data/labeled_raw_modalities.csv --grid_mode optimal --outer_splits 5 --outer_repeats 5 --output_prefix results/results

# Apply predefined weights (skip CV)
python -m cwra --csv data/labeled_raw_modalities.csv --apply_weights --output_prefix results_direct
```

### Python API

```python
import pandas as pd
from cwra import CWRA

df = pd.read_csv('your_data.csv')

cwra = CWRA(
    modalities=['graphdta_kd', 'vina_score', 'boltz_affinity'],
    focus='early',
    n_repeats=3,
    n_splits=5
)

results = cwra.fit_predict(df)
print(results.summary())
```

## Project Structure

```
cwra-vdr/
├── cwra/
│   ├── __init__.py
│   └── cwra.py
├── docs/
├── scripts/
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE/
├── tests/
├── pyproject.toml
├── setup.py
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE
```

## Modalities

| Modality | Description | Source |
|----------|-------------|--------|
| GraphDTA-Kd | Graph neural network predicting dissociation constants from molecular graphs and protein sequences | [GitHub](https://github.com/thinng/GraphDTA)  |
| GraphDTA-Ki | Graph neural network predicting inhibition constants | [GitHub](https://github.com/thinng/GraphDTA)  |
| GraphDTA-IC50 | Graph neural network predicting half-maximal inhibitory concentrations | [GitHub](https://github.com/thinng/GraphDTA)  |
| MLT-LE pKd | Multi-task residual neural network for binding affinity prediction across pKd, pKi, pIC50 tasks | [GitHub](https://github.com/VeaLi/MLT-LE) |
| AutoDock Vina | Physics-based docking scoring function | [AutoDock Vina](https://vina.scripps.edu/)  |
| Boltz-2 affinity | Foundation model for biomolecular structure and binding affinity prediction | [GitHub](https://github.com/jwohlwend/boltz)  |
| Boltz-2 confidence | Binding likelihood score from Boltz-2 | [GitHub](https://github.com/jwohlwend/boltz) |
| Uni-Mol similarity | 3D molecular representation learning framework; similarity to reference actives | [GitHub](https://github.com/deepmodeling/Uni-Mol) |
| TankBind affinity | Trigonometry-aware neural network for binding structure and affinity prediction | [GitHub](https://github.com/luwei0917/TankBind)  |
| DrugBAN affinity | Bilinear attention network learning pairwise interactions from 2D molecular graphs and protein sequences | [GitHub](https://github.com/peizhenbai/DrugBAN)  |
| MolTrans affinity | Transformer using frequent consecutive subsequence mining for drug-target interaction prediction | [GitHub](https://github.com/kexinhuang12345/MolTrans)  |

## Performance

Cross-validated performance using scaffold-grouped nested CV (5 folds × 3 repeats). EF@k% measures enrichment factor at top k% of the ranked database. Values are mean ± standard deviation.

### Method Comparison (CV Results)

| Category | Method | EF@1% | EF@5% | EF@10% | EF@20% | EF@30% |
|----------|--------|-------|-------|--------|--------|--------|
| Fusion | **CWRA** | **3.78 ± 2.60** | **3.18 ± 1.65** | 2.47 ± 1.00 | 2.02 ± 0.55 | 1.75 ± 0.40 |
| Fusion | Equal-weight | 3.68 ± 1.91 | 2.89 ± 1.12 | **2.61 ± 0.64** | **2.08 ± 0.43** | **1.76 ± 0.30** |
| Single | UniMol similarity | 3.18 ± 2.94 | 2.53 ± 2.09 | 2.10 ± 1.50 | 1.93 ± 0.83 | 1.61 ± 0.49 |
| Single | MLTLE pKd | 3.21 ± 2.90 | 1.84 ± 0.74 | 1.55 ± 0.55 | 1.43 ± 0.41 | 1.46 ± 0.36 |
| Single | GraphDTA-Kd | 2.64 ± 1.55 | 2.30 ± 0.63 | 1.90 ± 0.58 | 1.75 ± 0.44 | 1.60 ± 0.39 |
| Single | GraphDTA-Ki | 2.46 ± 2.25 | 2.26 ± 0.67 | 2.02 ± 0.65 | 1.74 ± 0.41 | 1.64 ± 0.34 |
| Single | GraphDTA-IC50 | 2.40 ± 1.46 | 2.34 ± 0.83 | 1.98 ± 0.51 | 1.71 ± 0.40 | 1.63 ± 0.31 |
| Single | MolTrans | 2.04 ± 1.16 | 1.52 ± 0.58 | 1.46 ± 0.38 | 1.14 ± 0.24 | 1.20 ± 0.17 |
| Single | AutoDock Vina | 0.94 ± 1.24 | 1.87 ± 0.87 | 1.87 ± 0.79 | 1.80 ± 0.55 | 1.67 ± 0.38 |
| Single | Boltz-2 confidence | 1.14 ± 0.92 | 2.01 ± 0.63 | 1.71 ± 0.45 | 1.81 ± 0.22 | 1.66 ± 0.19 |
| Single | Boltz-2 affinity | 0.97 ± 0.89 | 1.57 ± 0.71 | 1.68 ± 0.63 | 1.64 ± 0.59 | 1.46 ± 0.45 |
| Single | DrugBAN | 1.27 ± 1.87 | 1.80 ± 0.94 | 1.51 ± 0.68 | 1.51 ± 0.47 | 1.44 ± 0.34 |
| Single | TankBind | 0.58 ± 0.73 | 0.82 ± 0.70 | 0.98 ± 0.65 | 1.16 ± 0.55 | 1.23 ± 0.52 |
| Baseline | Random | 1.12 ± 1.06 | 1.26 ± 0.47 | 1.30 ± 0.40 | 1.26 ± 0.23 | 1.25 ± 0.17 |

### Full Dataset Performance (No CV)

| Metric | CWRA | Equal-weight | Best Single (UniMol) |
|--------|------|--------------|---------------------|
| EF@1% | 3.56 | 1.64 | 1.91 |
| EF@5% | 2.63 | 2.30 | 1.97 |
| EF@10% | 2.05 | 1.81 | 1.56 |
| Hits@10% | 75 | 66 | 57 |
| Hits@20% | 123 | 115 | 105 |
| Hits@30% | 168 | 159 | 146 |

**Key Results:**
- CWRA achieves **EF@1% = 3.78 ± 2.60** in CV, outperforming all single modalities
- CWRA beats the best single modality (UniMol) by **19%** at EF@1%
- CWRA shows **90%** improvement over random baseline at EF@10%
- On full dataset, CWRA recovers **75 actives** (20.5%) in top 10%, **168 actives** (45.9%) in top 30%

## Usage

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `data/labeled_raw_modalities.csv` | Input CSV with modalities + SMILES + source |
| `--outer_splits` | `5` | Number of outer CV folds |
| `--outer_repeats` | `3` | Number of outer CV repeats |
| `--inner_splits` | `3` | Number of inner CV folds |
| `--seed` | `42` | Random seed |
| `--risk_beta` | `0.3` | Risk aversion parameter (mean - beta*std) |
| `--focus` | `early` | Optimization focus: 'early', 'balanced', 'standard', 'comprehensive' |
| `--grid_mode` | `optimal` | Grid size: 'narrow', 'optimal', 'default', 'wide', 'extended', 'production' |
| `--output_prefix` | `results` | Prefix for output files |
| `--top_n` | `25` | Number of top/bottom structures to extract |
| `--n_jobs` | `-1` | Parallel jobs (-1 for all cores) |
| `--apply_weights` | `False` | Apply predefined optimal weights (skip CV) |

### Input Format

The input CSV requires:
- `smiles`: SMILES strings
- `source`: Source identifier (e.g., 'initial_370' for actives)
- Modality columns: `graphdta_kd`, `graphdta_ki`, `graphdta_ic50`, `mltle_pKd`, `vina_score`, `boltz_affinity`, `boltz_confidence`, `unimol_similarity`, `tankbind_affinity`, `drugban_affinity`, `moltrans_affinity`

### Output Files

- `{prefix}_table5_weights.csv`: Modality weights and individual performance
- `{prefix}_table6_performance.csv`: Performance comparison across methods
- `{prefix}_full_ranking.csv`: Complete ranking of all compounds
- `{prefix}_top{top_n}_G.csv`: Top generated compounds by CWRA rank
- `{prefix}_bottom{top_n}_G.csv`: Bottom generated compounds by CWRA rank
- `{prefix}_hyperparameters.csv`: Selected hyperparameters
- `{prefix}_latex_tables.tex`: LaTeX formatted tables for manuscript

## Metrics

- **EF@k%**: Enrichment Factor at k% of database
- **Hits@k**: Number of actives in top k compounds
- **BEDROC**: Boltzmann-Enhanced Discrimination of ROC
- **Mean Rank**: Average rank of active compounds

## API Reference

### CWRA Class

```python
CWRA(
    modalities: list,       # List of modality column names
    focus: str,             # 'early', 'balanced', 'standard'
    aggregation: str,       # 'weighted', 'rrf', 'power'
    n_repeats: int,         # CV repeats
    n_splits: int,          # CV folds
    seed: int               # Random seed
)
```

Methods:
- `fit_predict(df)`: Run CWRA analysis
- `summary()`: Return performance summary

### Utility Functions

- `compute_scaffold(smiles)`: Compute Murcko scaffold
- `bedroc_from_x(x, alpha, A, N)`: Compute BEDROC score
- `reciprocal_rank_fusion(ranks, k=60.0)`: RRF aggregation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Authors

- **Abylay Salimzhanov** — [ORCID](https://orcid.org/0000-0001-6630-585X)
- **Ferdinand Molnár** — [ORCID](https://orcid.org/0000-0001-9008-4233)
- **Siamac Fazli** — [ORCID](https://orcid.org/0000-0003-3397-0647)

## Citation

```bibtex
@software{salimzhanov2025cwra,
  title={CWRA: Calibrated Weighted Rank Aggregation for VDR Virtual Screening},
  author={Salimzhanov, Abylay and Moln{\'a}r, Ferdinand and Fazli, Siamac},
  year={2025},
  url={https://github.com/Salimzhanov/cwra-vdr},
  version={1.1.0}
}
```

## License

MIT License — see [LICENSE](LICENSE).
