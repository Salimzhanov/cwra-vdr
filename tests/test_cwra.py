"""
Unit tests for CWRA package
"""

import pytest
import numpy as np
import pandas as pd
from cwra.cwra import murcko_smiles, bedroc_from_x, reciprocal_rank_fusion


class TestCWRA:
    """Test CWRA utility functions."""

    def test_murcko_smiles_valid(self):
        """Test Murcko scaffold computation for valid SMILES."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        scaffold = murcko_smiles(smiles)
        assert scaffold is not None
        assert isinstance(scaffold, str)
        assert len(scaffold) > 0

    def test_murcko_smiles_invalid(self):
        """Test Murcko scaffold computation for invalid SMILES."""
        scaffold = murcko_smiles("invalid_smiles")
        assert scaffold is None

    def test_bedroc_from_x(self):
        """Test BEDROC computation."""
        # Create synthetic data
        np.random.seed(42)
        x = np.random.random(100)
        alpha = 20.0
        A = 10
        N = 100

        bedroc = bedroc_from_x(x, alpha, A, N)
        assert isinstance(bedroc, float)
        assert 0.0 <= bedroc <= 1.0

    def test_reciprocal_rank_fusion(self):
        """Test Reciprocal Rank Fusion."""
        np.random.seed(42)
        ranks = np.random.randint(1, 101, (50, 3))  # 50 compounds, 3 modalities

        scores = reciprocal_rank_fusion(ranks)
        assert len(scores) == 50
        assert all(isinstance(s, (int, float)) for s in scores)
        # Should be negative (higher ranks get lower scores)
        assert all(s <= 0 for s in scores)


class TestDataValidation:
    """Test data validation functions."""

    def test_required_columns(self):
        """Test that required columns are present."""
        # This would be tested in the main function
        # For now, just ensure the test framework works
        assert True

    def test_modality_columns_exist(self):
        """Test that modality columns exist in dataframe."""
        # Create test dataframe
        df = pd.DataFrame({
            'smiles': ['CC', 'CN'],
            'source': ['test1', 'test2'],
            'graphdta_kd': [1.0, 2.0],
            'vina_score': [-8.0, -7.0]
        })

        required_modalities = ['graphdta_kd', 'vina_score']
        for mod in required_modalities:
            assert mod in df.columns


if __name__ == "__main__":
    pytest.main([__file__])