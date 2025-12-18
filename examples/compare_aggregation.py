#!/usr/bin/env python3
"""
CWRA Example: Compare Aggregation Methods

This script demonstrates how to compare different aggregation methods
(RRF, Power transformation, and Weighted ranks) using the CWRA framework.
"""

import os
import sys
import subprocess

def run_cwra(aggregation_method, output_prefix):
    """Run CWRA with specified aggregation method."""
    cmd = [
        sys.executable, 'cwra.py',
        '--csv', 'data/labeled_raw_modalities.csv',
        '--outer_repeats', '2',
        '--outer_splits', '5',
        '--focus', 'early',
        '--aggregation', aggregation_method,
        '--seed', '42',
        '--output_prefix', output_prefix
    ]

    print(f"Running CWRA with {aggregation_method} aggregation...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if result.returncode == 0:
        print(f"✓ {aggregation_method} completed successfully")
        return True
    else:
        print(f"✗ {aggregation_method} failed: {result.stderr}")
        return False

if __name__ == "__main__":
    methods = ['weighted', 'rrf', 'power']

    for method in methods:
        success = run_cwra(method, f'comparison_{method}')
        if not success:
            sys.exit(1)

    print("\nAll aggregation methods completed. Compare the results in the output files.")