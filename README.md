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
# Quick test
cwra --csv data/labeled_raw_modalities.csv --outer_repeats 1 --outer_splits 3 --focus early

# Full run
cwra --csv data/labeled_raw_modalities.csv --outer_repeats 5 --outer_splits 10 --focus early --output_prefix results
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

| Modality | Description | Source | Direction |
|----------|-------------|--------|-----------|
| GraphDTA-Kd | Graph neural network predicting dissociation constants from molecular graphs and protein sequences | [GitHub](https://github.com/thinng/GraphDTA) | Lower ↓ |
| GraphDTA-Ki | Graph neural network predicting inhibition constants | [GitHub](https://github.com/thinng/GraphDTA) | Lower ↓ |
| GraphDTA-IC50 | Graph neural network predicting half-maximal inhibitory concentrations | [GitHub](https://github.com/thinng/GraphDTA) | Lower ↓ |
| MLT-LE pKd | Multi-task residual neural network for binding affinity prediction across pKd, pKi, pIC50 tasks | [GitHub](https://github.com/VeaLi/MLT-LE) | Higher ↑ |
| AutoDock Vina | Physics-based docking scoring function | [AutoDock Vina](https://vina.scripps.edu/) | Lower ↓ |
| Boltz-2 affinity | Foundation model for biomolecular structure and binding affinity prediction | [GitHub](https://github.com/jwohlwend/boltz) | Lower ↓ |
| Boltz-2 confidence | Binding likelihood score from Boltz-2 | [GitHub](https://github.com/jwohlwend/boltz) | Higher ↑ |
| Uni-Mol similarity | 3D molecular representation learning framework; similarity to reference actives | [GitHub](https://github.com/deepmodeling/Uni-Mol) | Higher ↑ |
| TankBind affinity | Trigonometry-aware neural network for binding structure and affinity prediction | [GitHub](https://github.com/luwei0917/TankBind) | Lower ↓ |
| DrugBAN affinity | Bilinear attention network learning pairwise interactions from 2D molecular graphs and protein sequences | [GitHub](https://github.com/peizhenbai/DrugBAN) | Lower ↓ |
| MolTrans affinity | Transformer using frequent consecutive subsequence mining for drug-target interaction prediction | [GitHub](https://github.com/kexinhuang12345/MolTrans) | Lower ↓ |

## Performance

Results from scaffold-grouped nested CV on initial_365 actives. Values are mean ± std across 5-fold CV repeats.


### Method Comparison

| Category | Method | EF@10% | Hits@10% | Hits@20% | Hits@30% |
|----------|--------|--------|----------|----------|----------|
| Single | GraphDTA-Kd ↓ | 2.05 ± 0.91 | 65.8 ± 29.2 | 12.8 ± 3.7 | 17.2 ± 6.0 |
| Single | GraphDTA-Ki ↓ | 2.17 ± 0.89 | 69.7 ± 28.6 | 12.9 ± 4.9 | 18.2 ± 6.9 |
| Single | GraphDTA-IC50 ↓ | 2.13 ± 0.89 | 68.4 ± 28.6 | 12.3 ± 5.6 | 18.2 ± 7.5 |
| Single | MLT-LE pKd ↑ | 1.63 ± 0.96 | 52.3 ± 30.8 | 10.5 ± 3.8 | 16.2 ± 5.9 |
| Single | AutoDock Vina ↓ | 0.60 ± 0.38 | 19.3 ± 12.2 | 5.9 ± 4.0 | 8.9 ± 5.7 |
| Single | Boltz-2 affinity ↓ | 1.68 ± 0.71 | 53.9 ± 22.8 | 12.7 ± 12.1 | 16.9 ± 14.6 |
| Single | Boltz-2 confidence ↑ | 1.73 ± 0.63 | 55.5 ± 20.2 | 13.5 ± 6.2 | 19.1 ± 10.5 |
| Single | Uni-Mol similarity ↑ | 2.18 ± 1.87 | 70.0 ± 60.0 | 14.0 ± 8.9 | 17.6 ± 9.3 |
| Single | TankBind affinity ↓ | 0.73 ± 0.79 | 23.4 ± 25.4 | 8.2 ± 5.5 | 11.9 ± 6.1 |
| Single | DrugBAN ↓ | 1.04 ± 0.65 | 33.3 ± 20.8 | 7.9 ± 8.0 | 11.6 ± 10.7 |
| Single | MolTrans ↓ | 1.54 ± 0.60 | 49.4 ± 19.3 | 8.3 ± 3.0 | 13.2 ± 5.1 |
| Fusion | Equal-weight | 2.07 ± 0.75 | 67.7 ± 24.5 | 14.5 ± 6.3 | 18.9 ± 8.6 |
| Fusion | **CWRA-early** | **2.41 ± 0.89** | **78.9 ± 29.1** | 14.8 ± 5.7 | **19.2 ± 7.2** |
| Baseline | Random | 1.40 ± 0.59 | 44.9 ± 18.9 | 9.7 ± 5.0 | 14.0 ± 7.1 |

CWRA-early outperforms equal-weight fusion and all individual modalities on EF@10% and Hits@10%.

## Usage

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `labeled_raw_modalities.csv` | Input CSV with modalities + SMILES + source |
| `--outer_splits` | `10` | Number of outer CV folds |
| `--outer_repeats` | `5` | Number of outer CV repeats |
| `--seed` | `42` | Random seed |
| `--risk_beta` | `0.5` | Risk aversion parameter (mean - beta*std) |
| `--focus` | `early` | Optimization focus: 'early', 'balanced', 'standard' |
| `--aggregation` | `weighted` | Aggregation: 'weighted', 'rrf', 'power' |
| `--output_prefix` | `cwrad` | Prefix for output files |
| `--top_n` | `25` | Number of top/bottom structures to extract |

### Input Format

The input CSV requires:
- `smiles`: SMILES strings
- `source`: Source identifier (e.g., 'initial_370' for actives)
- Modality columns: `graphdta_kd`, `graphdta_ki`, `graphdta_ic50`, `mltle_pKd`, `vina_score`, `boltz_affinity`, `boltz_confidence`, `unimol_similarity`, `tankbind_affinity`, `drugban_affinity`, `moltrans_affinity`

### Output Files

- `{prefix}_table5_weights.csv`: Modality weights and individual performance
- `{prefix}_table6_performance.csv`: Performance comparison
- `{prefix}_full_ranking.csv`: Complete ranking
- `{prefix}_top{top_n}_G.csv`: Top generated compounds
- `{prefix}_bottom{top_n}_G.csv`: Bottom generated compounds

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
