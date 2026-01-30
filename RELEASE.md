# Release Notes

## v1.3.0 (January 2026)

### Highlights
- **CWRA achieves EF@1% = 24.96** on 16,059 compounds (366 actives)
- Calcitriol (reference VDR ligand) ranked #29 (top 0.2%)
- Complete Boltz-2 structure prediction pipeline for top candidates
- Consolidated codebase with `cwra.py` as single entry point

### Key Results
| Metric | Value |
|--------|-------|
| EF@1% | 24.96 |
| EF@5% | 13.57 |
| EF@10% | 9.68 |
| Active Recovery @30% | 100% |

### Data
- 16,059 total compounds
- 366 known actives (initial_370 + calcitriol)
- 11 modalities (docking, deep learning, similarity)
- 31 Boltz-2 predicted structures for G-group compounds

## Version 1.2.0 (2026-01-30)

### Summary
This release adds a complete structure prediction pipeline for generating 3D protein-ligand complex structures of CWRA-ranked compounds using both AutoDock Vina docking and Boltz-2 AI structure prediction.

### New Features
- **Structure Prediction Pipeline**
  - AutoDock Vina docking for top/bottom CWRA-ranked compounds
  - Boltz-2 AI structure prediction with GPU acceleration
  - G-group analysis (G1/G2/G3 consensus groups)
  - 31 VDR-ligand complex structures generated

- **Scripts Added**
  - `scripts/generate_g_group_pdbs.py`: Generate docked structures
  - `scripts/run_boltz2_predictions.py`: Run Boltz-2 predictions

### Files Updated
- README.md: Added structure prediction documentation
- CHANGELOG.md: Updated with v1.2.0 changes
- pyproject.toml: Version bump to 1.2.0
- .gitignore: Added project-specific ignores

### Requirements
- Python 3.8+
- AutoDock Vina (optional, for docking)
- Boltz-2 v2.2.1+ (optional, for AI structure prediction)
- NVIDIA GPU recommended for Boltz-2

### Results
- 31 Boltz-2 predicted structures in `results/cwra_final/boltz2_predictions/`
- Docked structures in `results/cwra_final/g_group_pdbs_docked/`
- Complete rankings in `results/cwra_final/cwra_final_rankings.csv`
