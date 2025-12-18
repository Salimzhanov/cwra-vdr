"""
CWRA: Calibrated Weighted Rank Aggregation for VDR Virtual Screening

A robust machine learning framework for combining multiple molecular docking
and binding affinity prediction modalities to improve virtual screening
performance for Vitamin D Receptor (VDR) ligands.

Authors:
- Abylay Salimzhanov (First Author)
- Ferdinand Molnár (Second Author)
- Siamac Fazli (Corresponding Author)
"""

__version__ = "1.0.0"
__author__ = "Abylay Salimzhanov, Ferdinand Molnár, Siamac Fazli"
__email__ = ""

from .cwra import main

__all__ = ["main"]