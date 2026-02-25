import os
import tempfile

import numpy as np
import pandas as pd

from pu_stepE_select import main


def _make_inputs(tmpdir):
    df = pd.DataFrame(
        {
            "source": ["G1", "G1", "G2", "G2"],
            "smiles": ["CCO", "CCC", "CO", "CN"],
        }
    )
    input_csv = os.path.join(tmpdir, "input.csv")
    df.to_csv(input_csv, index=False)

    pvals = pd.DataFrame(
        {
            "meta_score": [0.9, 0.8, 0.2, 0.1],
            "pval_weighted": [0.01, 0.2, 0.05, 0.5],
            "pu_label": [-1, -1, -1, -1],
        }
    )
    pvals_csv = os.path.join(tmpdir, "pvals.csv")
    pvals.to_csv(pvals_csv, index=False)
    return input_csv, pvals_csv


def test_stepE_cli_outputs_bh():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv, pvals_csv = _make_inputs(tmpdir)
        out_dir = os.path.join(tmpdir, "out")

        rc = main(
            [
                "--input",
                input_csv,
                "--pvalues",
                pvals_csv,
                "--output",
                out_dir,
                "--mode",
                "bh",
                "--q",
                "0.2",
                "--select-over",
                "unlabeled",
            ]
        )
        assert rc == 0

        out_path = os.path.join(out_dir, "final_selected.csv")
        summary_path = os.path.join(out_dir, "selection_summary.json")
        assert os.path.exists(out_path)
        assert os.path.exists(summary_path)

        out_df = pd.read_csv(out_path)
        assert "qval" in out_df.columns
        assert "selected" in out_df.columns
        assert out_df["selected"].dtype == bool or out_df["selected"].isin([True, False]).all()


def test_stepE_cli_outputs_alpha():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv, pvals_csv = _make_inputs(tmpdir)
        out_dir = os.path.join(tmpdir, "out")

        rc = main(
            [
                "--input",
                input_csv,
                "--pvalues",
                pvals_csv,
                "--output",
                out_dir,
                "--mode",
                "alpha",
                "--alpha",
                "0.05",
                "--select-over",
                "unlabeled",
            ]
        )
        assert rc == 0

        out_df = pd.read_csv(os.path.join(out_dir, "final_selected.csv"))
        assert "selected" in out_df.columns
