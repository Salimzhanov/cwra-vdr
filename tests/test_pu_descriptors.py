import numpy as np
import pandas as pd

from pu_descriptors import extract_precomputed_descriptors


def test_descriptor_coercion_and_selection():
    df = pd.DataFrame(
        {
            "MW": ["300.5", "bad"],
            "cLogP": ["2.1", "3.3"],
            "unimol_similarity": [0.1, 0.2],
            "boltz_confidence": [0.9, 0.8],
        }
    )
    desc = extract_precomputed_descriptors(df)
    assert list(desc.columns) == ["MW", "cLogP"]
    assert np.isnan(desc.loc[1, "MW"])
    assert desc["cLogP"].dtype.kind in {"f", "i"}


def test_no_descriptors_found():
    df = pd.DataFrame({"x": [1, 2], "unimol_similarity": [0.1, 0.2]})
    desc = extract_precomputed_descriptors(df)
    assert desc.shape[1] == 0
    assert list(desc.index) == list(df.index)
