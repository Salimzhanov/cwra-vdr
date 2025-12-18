# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-18

### Added
- Initial release of CWRA (Calibrated Weighted Rank Aggregation) framework
- Support for 9 different molecular docking and binding affinity modalities:
  - GraphDTA Kd/Ki/IC50 predictions
  - MLT-LE pKd predictions
  - AutoDock Vina docking scores
  - Boltz-2 affinity and confidence scores
  - Uni-Mol similarity scores
  - TankBind affinity predictions
- Nested cross-validation with hyperparameter optimization
- Multiple aggregation methods: weighted ranks, RRF, power transformation
- Comprehensive performance evaluation with EF@k%, BEDROC, and hit rates
- Scaffold-based cross-validation to prevent data leakage
- Command-line interface with extensive configuration options
- Example scripts and documentation

### Features
- **Multi-modal fusion**: Combines predictions from multiple computational methods
- **Optimized weighting**: Learns optimal modality weights through nested CV
- **Stability-focused**: Reduces overfitting with robust objective functions
- **Reproducible**: Deterministic results with fixed random seeds
- **Extensible**: Easy to add new modalities or scoring functions

### Technical Details
- Python 3.8+ compatibility
- Dependencies: numpy, pandas, scipy, scikit-learn, rdkit
- Comprehensive test suite
- MIT License

## [0.1.0] - 2025-01-01

### Added
- Initial prototype implementation
- Basic weighted rank aggregation
- Support for core modalities (GraphDTA, Vina, Boltz)
- Simple cross-validation framework