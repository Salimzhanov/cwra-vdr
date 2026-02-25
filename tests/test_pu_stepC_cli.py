import os
import tempfile

import numpy as np
import pandas as pd

from pu_stepC_conformal_pvalues import main


def test_stepC_cli_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "meta_scores.csv")
        out_dir = os.path.join(tmpdir, "out")
        df = pd.DataFrame(
            {
                "meta_score": [0.9, 0.1, 0.8, 0.2, 0.7],
                "pu_label": [1, 0, 1, 0, -1],
            }
        )
        df.to_csv(meta_path, index=False)

        rc = main(
            [
                "--meta-scores",
                meta_path,
                "--output",
                out_dir,
                "--calib-frac",
                "0.5",
                "--seed",
                "7",
                "--calib-set",
                "labeled",
            ]
        )
        assert rc == 0

        out_path = os.path.join(out_dir, "conformal_pvalues.csv")
        assert os.path.exists(out_path)

        out_df = pd.read_csv(out_path)
        assert "pval_unweighted" in out_df.columns
        assert np.all(out_df["pval_unweighted"] > 0)
        assert np.all(out_df["pval_unweighted"] <= 1)
