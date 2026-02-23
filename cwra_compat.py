#!/usr/bin/env python3
"""Compatibility exports for CWRA symbols across file/package layouts.

This project currently has both:
- a top-level script module: ``cwra.py``
- a package directory: ``cwra/``

Import resolution can pick the package first, which may not expose the
CV symbols needed by PU pipeline steps. This module loads symbols from
the top-level ``cwra.py`` file explicitly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_cwra_file_module():
    module_name = "_cwra_file_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    cwra_path = Path(__file__).with_name("cwra.py")
    if not cwra_path.exists():
        raise ModuleNotFoundError(
            f"Expected CWRA module file not found: {cwra_path}"
        )

    spec = importlib.util.spec_from_file_location(module_name, str(cwra_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {cwra_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_cwra_mod = _load_cwra_file_module()

CWRAConfig = _cwra_mod.CWRAConfig
normalize_modalities = _cwra_mod.normalize_modalities
OPTIMIZERS = _cwra_mod.OPTIMIZERS

