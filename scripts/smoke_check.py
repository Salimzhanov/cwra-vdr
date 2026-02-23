#!/usr/bin/env python3
"""Minimal environment and CLI smoke checks for CWRA-VDR."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "rdkit",
    "matplotlib",
    "PIL",
]

ENTRYPOINTS = [
    [sys.executable, "cwra.py", "--help"],
    [sys.executable, "pu_conformal.py", "--help"],
    [sys.executable, "create_cwra_tables.py", "--help"],
    [sys.executable, "make_mol_panel.py", "--help"],
    [sys.executable, "run_cwra.py", "--help"],
]


def _check_exists(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required path: {path}")


def _check_imports() -> None:
    for mod in REQUIRED_IMPORTS:
        importlib.import_module(mod)


def _run_help(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{msg}")


def main() -> int:
    _check_exists("data/composed_modalities_with_rdkit.csv")
    _check_imports()
    for cmd in ENTRYPOINTS:
        _run_help(cmd)
    print("SMOKE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
