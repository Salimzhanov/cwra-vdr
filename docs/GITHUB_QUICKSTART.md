# GitHub Quickstart (CWRA + PU-Conformal)

This is the fastest path for a new user to reproduce the main workflow.

## 1) Environment

Recommended: Python 3.10 in a fresh virtual environment.

```bash
git clone https://github.com/Salimzhanov/cwra-vdr.git
cd cwra-vdr
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or as an editable package (enables the `cwra` console command):

```bash
pip install -e .
```

## 2) Run the full paper pipeline

```bash
python run_cwra.py
```

This runs all steps sequentially. Preview what will run with:

```bash
python run_cwra.py --dry-run
```

## 3) Individual steps

### 3a) CWRA cross-validation

```bash
python cwra.py \
  -i data/composed_modalities_with_rdkit.csv \
  -o results/final \
  --method fair --norm minmax --objective default \
  --bedroc --bedroc-alpha 100 --max-mw 700 \
  --train-frac 0.85 --cv-folds 5 --seed 42 \
  --de-workers -1 --include-newref-137-as-active \
  --fold-honest-unimol --unimol-embeddings data/unimol_embeddings.npz \
  --extra-metrics
```

Expected key outputs in `results/final/`:
- `cwra_cv_mean_weights.csv`
- `cwra_cv_paper_table.csv`
- `cwra_cv_folds_*.csv`
- `cwra_cv_mean_rank_summary.csv`

### 3b) Build LaTeX tables for paper/supplement

```bash
python create_cwra_tables.py --results-dir results/final
```

Outputs:
- `fusion_performance_concise.tex`
- `fusion_performance_extended.tex`
- `fusion_weights_meanrank_table.tex`

### 3c) PU-conformal pipeline with CWRA weights

```bash
python pu_conformal.py \
  --input data/composed_modalities_with_rdkit.csv \
  --outdir results/final/pipeline \
  --score-source cwra \
  --cwra-weights-csv results/final/cwra_cv_mean_weights.csv \
  --cwra-norm minmax \
  --include-newref-137-as-active --seed 42 \
  --max-mw 700 --max-rotb 15 --pval-type unweighted
```

Final selection file:
- `results/final/pipeline/E/final_selected.csv`

### 3d) Draw molecule panel

```bash
python make_mol_panel.py \
  --csv results/final/pipeline/E/final_selected.csv \
  --out results/final/panel_10x5.pdf \
  --n 50 --per-row 5 --cell 320 --sort meta --tau 0.001
```

## 4) Smoke check

```bash
python scripts/smoke_check.py
```

This validates that all required packages are importable and that the
main CLI entrypoints respond to `--help`.
