"""
cwra -- Calibrated Weighted Rank Aggregation for VDR virtual screening.

Re-exports public symbols from the top-level cwra.py script so that
``from cwra import CWRAConfig`` and ``python -m cwra`` both work.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__version__ = "1.3.0"
__author__ = "Abylay Salimzhanov, Ferdinand Molnár, Siamac Fazli"
__email__ = ""


def _load_cwra_module():
    """Load the top-level ``cwra.py`` script as a private module.

    Python resolves ``import cwra`` to this *package* (``cwra/``), so the
    sibling ``cwra.py`` file is not directly importable.  We use
    :mod:`importlib` to load it under a private name.
    """
    _name = "_cwra_script"
    if _name in sys.modules:
        return sys.modules[_name]

    cwra_path = Path(__file__).resolve().parent.parent / "cwra.py"
    if not cwra_path.exists():
        raise ModuleNotFoundError(
            f"Top-level cwra.py not found at {cwra_path}.  "
            "Make sure you are running from the repository root or "
            "installed the package in editable mode (pip install -e .)."
        )

    spec = importlib.util.spec_from_file_location(_name, str(cwra_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec from {cwra_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[_name] = mod
    spec.loader.exec_module(mod)
    return mod


_cwra = _load_cwra_module()

# Re-export public symbols
main = _cwra.main
CWRAConfig = _cwra.CWRAConfig
CWRATrainTest = _cwra.CWRATrainTest
normalize_modalities = _cwra.normalize_modalities
OPTIMIZERS = _cwra.OPTIMIZERS
CUTOFFS = _cwra.CUTOFFS
OBJECTIVE_PRESETS = _cwra.OBJECTIVE_PRESETS
compute_bedroc = _cwra.compute_bedroc
compute_ef = _cwra.compute_ef
evaluate_weights = _cwra.evaluate_weights
make_cv_folds = _cwra.make_cv_folds
run_cross_validation = _cwra.run_cross_validation

__all__ = [
    "main",
    "CWRAConfig",
    "CWRATrainTest",
    "normalize_modalities",
    "OPTIMIZERS",
    "CUTOFFS",
    "OBJECTIVE_PRESETS",
    "compute_bedroc",
    "compute_ef",
    "evaluate_weights",
    "make_cv_folds",
    "run_cross_validation",
]