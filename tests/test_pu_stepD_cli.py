import os
import tempfile

import numpy as np
import pandas as pd

from pu_stepD_weighted_conformal import main


def _make_inputs(tmpdir):
    rng = np.random.RandomState(4)
    n = 20
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "source": ["initial_370"] * 5 + ["calcitriol"] * 5 + ["G1"] * 10,
            "graphdta_kd": rng.rand(n),
            "MW": rng.rand(n) * 300,
        }
    )
    input_csv = os.path.join(tmpdir, "input.csv")
    df.to_csv(input_csv, index=False)

    labels = pd.DataFrame(
        {
            "index": np.arange(n),
            "source": df["source"].values,
            "pu_label": [1] * 5 + [0] * 5 + [-1] * 10,
        }
    )
    labels_csv = os.path.join(tmpdir, "pu_labels.csv")
    labels.to_csv(labels_csv, index=False)

    meta_scores = pd.DataFrame(
        {
            "meta_score": rng.rand(n),
            "pu_label": labels["pu_label"].values,
            "source": df["source"].values,
        }
    )
    meta_csv = os.path.join(tmpdir, "meta_scores.csv")
    meta_scores.to_csv(meta_csv, index=False)

    return input_csv, labels_csv, meta_csv


def test_stepD_cli_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv, labels_csv, meta_csv = _make_inputs(tmpdir)
        out_dir = os.path.join(tmpdir, "out")

        rc = main(
            [
                "--input",
                input_csv,
                "--pu-labels",
                labels_csv,
                "--meta-scores",
                meta_csv,
                "--output",
                out_dir,
                "--seed",
                "7",
            ]
        )
        assert rc == 0

        out_path = os.path.join(out_dir, "weighted_pvalues.csv")
        assert os.path.exists(out_path)

        out_df = pd.read_csv(out_path)
        assert np.all(out_df["pval_weighted"] > 0)
        assert np.all(out_df["pval_weighted"] <= 1)
        assert np.all(np.isfinite(out_df["weight"]))
