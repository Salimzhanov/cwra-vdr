#!/usr/bin/env python3
"""
CWRA Example: Advanced Configuration

This script demonstrates advanced usage of the CWRA framework with
custom hyperparameters and different aggregation methods.
"""

import os
import sys

# Add the parent directory to the path to import cwra
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwra import main

if __name__ == "__main__":
    # Example: Run CWRA with full cross-validation for production use
    import sys
    sys.argv = [
        'cwra.py',
        '--csv', 'data/labeled_raw_modalities.csv',
        '--outer_repeats', '5',
        '--outer_splits', '10',
        '--focus', 'early',
        '--aggregation', 'weighted',
        '--seed', '42',
        '--output_prefix', 'production_results'
    ]

    main()