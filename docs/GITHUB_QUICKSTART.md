# GitHub Quickstart (CWRA + PU-Conformal)

This is the fastest path for a new user to reproduce the main workflow.

## 1) Environment

Recommended: Python 3.10 in a fresh virtual environment.

```bash
git clone https://github.com/<your-org>/cwra-vdr.git
cd cwra-vdr
python -m pip install -U pip
python -m pip install -e .
```

Or with requirements:

```bash
python -m pip install -r requirements.txt
```

## 2) Run CWRA cross-validation

```bash
python cwra_cv_v2.py \
  -o results/bedroc \
  --norm minmax \
  --max-mw 600 --max-rotb 15 \
  --cv-folds 5 \
  --seed 42 \
  --bedroc --bedroc-alpha 80.0 \
  --extra-metrics --latex
```

Expected key outputs in `results/bedroc/`:
- `cwra_cv_mean_weights.csv`
- `cwra_cv_paper_table.csv`
- `cwra_cv_folds_*.csv`
- `cwra_cv_mean_rank_summary.csv`

## 3) Build LaTeX tables for paper/supplement

```bash
python create_cwra_tables.py --results-dir results/bedroc
```

Outputs:
- `fusion_performance_concise.tex`
- `fusion_performance_extended.tex`
- `fusion_weights_meanrank_table.tex`

## 4) Run PU-conformal pipeline with CWRA weights

```bash
python run_pu_conformal_pipeline.py \
  --input data/composed_modalities_with_rdkit.csv \
  --outdir results/bedroc \
  --score-source cwra \
  --cwra-weights-csv results/bedroc/cwra_cv_mean_weights.csv \
  --cwra-norm minmax \
  --max-mw 600 --max-rotb 15 \
  --calib-negatives unlabeled_random \
  --select-mode pval_cutoff --pval-cutoff 0.001 \
  --pval-type unweighted \
  --top-k 2000 \
  --seed 42
```

Final selection file:
- `results/bedroc/E/final_selected.csv`

## 5) Draw molecule panel

```bash
python make_mol_panel.py \
  --csv results/bedroc/E/final_selected.csv \
  --out results/bedroc/panel_10x5.pdf \
  --n 50 --per-row 5 --sort meta
```

## 6) Optional smoke check

```bash
python scripts/smoke_check.py
```

This validates imports and `--help` for key entrypoints.
