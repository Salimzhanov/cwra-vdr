#!/usr/bin/env python3
"""
CWRA Example: Basic Usage

This script demonstrates how to use the CWRA framework for VDR virtual screening
with a simple configuration optimized for quick testing.
"""

import os
import sys

# Add the parent directory to the path to import cwra
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwra import main

if __name__ == "__main__":
    # Example: Run CWRA with reduced parameters for quick testing
    import sys
    sys.argv = [
        'cwra.py',
        '--csv', 'data/labeled_raw_modalities.csv',
        '--outer_repeats', '1',
        '--outer_splits', '3',
        '--focus', 'early',
        '--seed', '42',
        '--output_prefix', 'example_results'
    ]

    main()