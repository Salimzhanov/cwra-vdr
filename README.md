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
- joblib>=1.0.0

## Quick Start

### Command Line

```bash
# Recommended: Use fixed optimal weights (fast, production-ready)
cwra --csv data/labeled_raw_modalities.csv --fixed_weights --output_prefix results

# Quick CV test
cwra --csv data/labeled_raw_modalities.csv --outer_repeats 1 --outer_splits 3 --focus early

# Full CV run (slower, for hyperparameter tuning)
cwra --csv data/labeled_raw_modalities.csv --outer_repeats 5 --outer_splits 10 --focus early --output_prefix results
```

### Python API

```python
import subprocess
import sys

# Run CWRA from Python using subprocess
result = subprocess.run([
    sys.executable, "-m", "cwra",
    "--csv", "data/labeled_raw_modalities.csv",
    "--fixed_weights",
    "--output_prefix", "results"
], capture_output=True, text=True)
print(result.stdout)
```

Or import and use the core functions directly:

```python
import pandas as pd
import numpy as np
from cwra import (
    murcko_smiles, 
    bedroc, 
    compute_weights,
    calc_modality_metrics,
    eval_at_cutoffs,
    MODALITIES,
    OPTIMAL_FIXED_WEIGHTS
)

# Load your data
df = pd.read_csv('your_data.csv')

# Use optimal fixed weights for scoring
mod_cols = [m[0] for m in MODALITIES]
weights = np.array([OPTIMAL_FIXED_WEIGHTS[col] for col in mod_cols])
weights = weights / weights.sum()

# ... compute ranks and aggregate with weights
```

## Project Structure

```
cwra-vdr/
├── cwra/
│   ├── __init__.py
│   └── cwra.py
├── docs/
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE/
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

Results from fixed optimal weights evaluation on 366 actives (initial_370 + calcitriol). 

### Method Comparison

| Category | Method | EF@1% | EF@5% | EF@10% | Hits@10% | Hits@20% |
|----------|--------|-------|-------|--------|----------|----------|
| Single | Uni-Mol similarity ↑ | 1.80 | 2.00 | 1.55 | 57.00 | 105.00 |
| Single | Boltz-2 confidence ↑ | 1.03 | 1.30 | 1.52 | 56.00 | 94.00 |
| Single | Boltz-2 affinity ↓ | 0.77 | 1.40 | 1.44 | 53.00 | 99.00 |
| Single | MolTrans ↓ | 1.29 | 1.30 | 1.11 | 41.00 | 70.00 |
| Single | DrugBAN ↓ | 0.77 | 1.13 | 1.03 | 38.00 | 66.00 |
| Single | MLT-LE pKd ↑ | 1.03 | 1.13 | 0.92 | 34.00 | 68.00 |
| Single | GraphDTA-IC50 ↑ | 0.77 | 1.08 | 0.87 | 32.00 | 51.00 |
| Single | GraphDTA-Ki ↑ | 0.77 | 0.54 | 0.71 | 26.00 | 56.00 |
| Single | GraphDTA-Kd ↑ | 0.51 | 0.43 | 0.68 | 25.00 | 61.00 |
| Single | TankBind ↓ | 0.51 | 0.43 | 0.54 | 20.00 | 58.00 |
| Single | AutoDock Vina ↓ | 0.00 | 0.49 | 0.49 | 18.00 | 49.00 |
| Fusion | Equal-weight | 0.77 | 1.19 | 1.03 | 38.00 | 63.00 |
| Fusion | **CWRA-early** | **2.06** | **1.84** | **1.63** | **60.00** | **102.00** |
| Baseline | Random | 1.00 | 1.00 | 1.00 | 36.78 | 73.34 |

**Key Results:**
- CWRA outperforms Equal-weight fusion by **168%** at EF@1% and **58%** at EF@10%
- CWRA outperforms the best individual modality (Uni-Mol similarity) by **14%** at EF@1% and **5%** at EF@10%
- CWRA consistently outperforms random baseline across all metrics

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
| `--grid_mode` | `optimal` | Grid size: 'narrow', 'optimal', 'default', 'wide', 'extended' |
| `--output_prefix` | `results` | Prefix for output files |
| `--top_n` | `25` | Number of top/bottom structures to extract |
| `--n_jobs` | `-1` | Parallel jobs (-1 for all cores) |
| `--fixed_weights` | `False` | Use optimal fixed weights instead of CV-learned weights |

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

### Core Functions

```python
from cwra import (
    murcko_smiles,      # Compute Murcko scaffold from SMILES
    bedroc,             # Compute BEDROC score
    shrink_factors,     # Compute shrinkage factors for correlation
    eval_at_cutoffs,    # Evaluate EF and hits at cutoffs
    compute_weights,    # Compute modality weights
    calc_modality_metrics,  # Calculate EF, rank score, BEDROC
    balanced_group_kfold,   # Create balanced group k-fold splits
    MODALITIES,         # List of modality definitions
    CUTOFF_PCTS,        # Cutoff percentages [1, 5, 10, 20, 30]
    OPTIMAL_FIXED_WEIGHTS,  # Pre-computed optimal weights
)
```

### Function Signatures

```python
def murcko_smiles(smiles: str) -> Optional[str]:
    """Compute Murcko scaffold SMILES from input SMILES."""

def bedroc(x: np.ndarray, alpha: float, A: int, N: int) -> float:
    """Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).
    
    Args:
        x: Normalized ranks of actives (0 = best, 1 = worst)
        alpha: Exponential decay parameter
        A: Number of actives
        N: Total number of compounds
    
    Returns:
        BEDROC score in [0, 1]
    """

def eval_at_cutoffs(score: np.ndarray, active_mask: np.ndarray, 
                    cutoffs: Dict[str, int], N: int) -> Dict:
    """Evaluate EF and hits at multiple cutoffs."""
```

### Command Line Interface

The primary interface is through the command line:

```bash
python -m cwra --help
```

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
