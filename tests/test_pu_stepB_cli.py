import os
import tempfile

import numpy as np
import pandas as pd

from pu_stepB_train_meta import main


def _make_csv(path):
    rng = np.random.RandomState(3)
    n = 30
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "source": ["initial_370"] * 10 + ["calcitriol"] * 10 + ["other"] * 10,
            "smiles": ["CCO"] * n,
            "graphdta_kd": rng.rand(n),
            "MW": rng.rand(n) * 300,
        }
    )
    df.to_csv(path, index=False)


def _make_labels(path, n):
    labels = pd.DataFrame(
        {
            "index": np.arange(n),
            "source": ["initial_370"] * 10 + ["calcitriol"] * 10 + ["other"] * 10,
            "pu_label": [1] * 10 + [0] * 10 + [-1] * 10,
        }
    )
    labels.to_csv(path, index=False)


def test_stepB_cli_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "input.csv")
        labels_csv = os.path.join(tmpdir, "labels.csv")
        out_dir = os.path.join(tmpdir, "out")

        _make_csv(input_csv)
        _make_labels(labels_csv, 30)

        rc = main(
            [
                "--input",
                input_csv,
                "--pu-labels",
                labels_csv,
                "--output",
                out_dir,
                "--seed",
                "7",
                "--calib-frac",
                "0.5",
            ]
        )
        assert rc == 0

        assert os.path.exists(os.path.join(out_dir, "meta_model.joblib"))
        assert os.path.exists(os.path.join(out_dir, "scaler.joblib"))
        assert os.path.exists(os.path.join(out_dir, "feature_schema.json"))
        scores_path = os.path.join(out_dir, "meta_scores.csv")
        assert os.path.exists(scores_path)

        scores = pd.read_csv(scores_path)
        assert len(scores) == 30
        assert "meta_score" in scores.columns
        assert "pu_label" in scores.columns
        assert "source" in scores.columns
        assert "smiles" in scores.columns
