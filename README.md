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

Hit recovery on full ranking of 1,602 compounds (365 initial_370 actives). EF@k% measures enrichment factor at top k% of the ranked database. Hits@k reports the number of actives found in the top k compounds.

### Method Comparison

| Category | Method | EF@10% | Hits@10% | Hits@20% | Hits@30% |
|----------|--------|--------|----------|----------|----------|
| Single | GraphDTA-Kd  | 1.40 | 51 | 100 | 138 |
| Single | GraphDTA-Ki  | 1.54 | 56 | 96 | 138 |
| Single | GraphDTA-IC50  | 1.48 | 54 | 100 | 136 |
| Single | MLT-LE pKd  | 1.26 | 46 | 79 | 126 |
| Single | AutoDock Vina  | 1.23 | 45 | 96 | 146 |
| Single | Boltz-2 affinity  | 1.45 | 53 | 98 | 135 |
| Single | Boltz-2 confidence  | 1.48 | 54 | 93 | 149 |
| Single | Uni-Mol similarity  | 1.56 | 57 | 104 | 145 |
| Single | TankBind affinity  | 0.88 | 32 | 68 | 118 |
| Single | DrugBAN  | 1.29 | 47 | 90 | 125 |
| Single | MolTrans  | 1.07 | 39 | 69 | 107 |
| Fusion | Equal-weight | 1.81 | 66 | 115 | 159 |
| Fusion | **CWRA** | **2.06** | **75** | **122** | **167** |
| Baseline | *Expected at random* | 1.00 | 36.5 | 72.9 | 109.4 |

**Key Results:**
- CWRA achieves **EF@10% = 2.06**, outperforming all single modalities and equal-weight fusion
- CWRA recovers **75 actives** (21%) in top 10%, **122 actives** (33%) in top 20%, **167 actives** (46%) in top 30%
- Best single modality (Uni-Mol similarity) achieves only 57 hits at 10% vs CWRA's 75 hits (+32%)

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
| `--focus` | `early` | Optimization focus: 'early', 'balanced', 'standard' |
| `--grid_mode` | `wide` | Grid size: 'narrow', 'default', 'wide', 'extended' |
| `--output_prefix` | `results` | Prefix for output files |
| `--top_n` | `25` | Number of top/bottom structures to extract |
| `--n_jobs` | `-1` | Parallel jobs (-1 for all cores) |

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
