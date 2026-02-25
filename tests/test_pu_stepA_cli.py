import json
import os
import tempfile

import numpy as np
import pandas as pd

from pu_stepA_reliable_negatives import main


def _make_csv(path):
    rng = np.random.RandomState(1)
    n = 20
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "source": [
                "initial_370",
                "calcitriol",
                "initial_370",
                "newRef_137",
            ]
            + ["other"] * (n - 4),
            "smiles": ["CCO"] * n,
            "graphdta_kd": rng.rand(n),
            "boltz_confidence": rng.rand(n),
            "unimol_similarity": rng.rand(n),
        }
    )
    df.to_csv(path, index=False)


def test_stepA_cli_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "input.csv")
        out_dir = os.path.join(tmpdir, "out")
        _make_csv(input_csv)

        rc = main(
            [
                "--input",
                input_csv,
                "--output",
                out_dir,
                "--neg-pos-ratio",
                "10",
                "--bottom-q",
                "0.5",
                "--seed",
                "7",
                "--no-rdkit",
            ]
        )
        assert rc == 0

        labels_path = os.path.join(out_dir, "pu_labels.csv")
        report_path = os.path.join(out_dir, "pu_stepA_report.json")
        assert os.path.exists(labels_path)
        assert os.path.exists(report_path)

        labels = pd.read_csv(labels_path)
        assert "id" in labels.columns
        assert "pu_label" in labels.columns

        counts = labels["pu_label"].value_counts()
        n_pos = int(counts.get(1, 0))
        n_rn = int(counts.get(0, 0))
        assert n_rn <= 10 * n_pos

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["n_pos"] == n_pos
        assert report["n_rn"] == n_rn
        assert report["N"] == len(labels)
        used_keys = [k.lower() for k in report.get("used_modality_keys", [])]
        assert not any(("boltz" in k and "conf" in k) for k in used_keys)
        assert not any("unimol" in k for k in used_keys)
