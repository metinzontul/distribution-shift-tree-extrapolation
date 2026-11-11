# When Does the Extrapolation Weakness of Gradient-Boosted Trees Matter?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24110546.svg)](https://doi.org/10.5281/zenodo.24110546)

Code and results for the paper

> U. Y. Caskurlu, H. Bakir, M. Zontul, S. Karateke. *When Does the Extrapolation Weakness of Gradient-Boosted Trees Matter? A Precondition Analysis of Distribution Shift in Tabular Regression.* Submitted to *Mathematics* (MDPI), 2026.

Gradient-boosted decision trees (GBDT) predict a constant beyond the range of their training data. The paper shows that this matters only when a shift pushes target-relevant features out of the training range (the *extrapolation demand*), and tests this on real and synthetic regression data:

- **Constructed feature-range shift** on 27 regression datasets (OpenML-CTR23 and three further datasets): CatBoost falls behind FT-Transformer as the test band moves beyond the training range, although it is the more accurate model without shift.
- **Natural temporal shift** on four TabReD datasets: the most target-relevant feature never leaves its training range, and no family gap is detected (TOST equivalence within ±0.05 nOOD).
- **Synthetic controls**: the gap appears when the target keeps rising beyond the training range and disappears when it flattens.
- **Linear extrapolators**: LightGBM with linear leaves extrapolates worse than CatBoost; ridge regression catches up only at high severity.

## Repository layout

```
first_set/            First set of experiments (15 seeds, 27 datasets, all model families)
  cwr/                data loading, shift construction, models, metrics
  experiments/        run scripts: wp3_pipeline.py (real data), wp2_confirmatory.py (synthetic),
                      run_hpo_ablation.py (tuning budget, RealMLP), run_walmart_oor.py (Walmart)
  results/
    wp3_analysis/     per-seed results (_long_covariate.csv, _long_temporal.csv), metadata
                      (_cell_meta.csv), summary tables, figure scripts, figures (pub/)
    wp3_revision/     tuning-budget ablation and Walmart analysis
  environment/        pinned environment (pip freeze) and platform notes
second_set/           Second set of experiments (3 seeds, 19 datasets): in-distribution
                      reference and LightGBM with linear leaves
  shiftbench/         reimplementation of the protocol as a Python package
  scripts/            run_covariate.py, run_temporal.py, run_synthetic.py, analyze.py, smoke_test.py
  results/            results and split diagnostics (CSV)
analysis/             scripts that reproduce the additional statistics reported in the paper
```

## Reproducing the reported statistics

All tables can be recomputed from the CSV files in this repository; no model needs to be retrained.

```bash
pip install numpy pandas scipy statsmodels
python analysis/first_set_stats.py    # Table 4 CIs, Holm, mixed model, ID comparison, TOST
python analysis/second_set_stats.py   # Table 3 (ID column), Table 5, normaliser checks
python analysis/supplementary_tables.py  # Supplementary Tables S1-S3 (writes analysis/supplementary/)
python first_set/results/wp3_analysis/make_pub_figures.py      # Figures 1, 3, 4, 5
python first_set/results/wp3_analysis/size_vs_gap_analysis.py  # Figure 2
```

## Rerunning the experiments

**Second set** (recommended entry point; Python ≥ 3.10):

```bash
cd second_set
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/run_covariate.py --ctr23 kin8nm --out runs/kin8nm \
    --models catboost lgbm_linear ridge --kinds extrap id --seeds 0-2
python scripts/analyze.py runs/kin8nm --out report.md
```

Datasets that are not in OpenML-CTR23 can be loaded by OpenML identifier (`--openml-id 225 296`) or from a local CSV (`--csv path.csv:target:name`).

**First set**: see `first_set/environment/environment.md`. Datasets are read from `first_set/datasets/` (not included); OpenML datasets are downloaded automatically, and the hourly bike-sharing data must be placed there manually. The raw per-cell prediction arrays (about 700 MB) are not included; the per-seed metrics derived from them are in `first_set/results/wp3_analysis/`.

## Metric

nOOD = RMSE on the test set divided by the standard deviation of the target in the training region. The gap is nOOD(CatBoost) − nOOD(FT-Transformer); positive values mean that CatBoost has the larger error. Per-dataset values are averaged over seeds before any test across datasets.

## Data sources

OpenML-CTR23 (suite 353), puma8NH (OpenML 225), Ailerons (OpenML 296), bike sharing (UCI; Fanaee-T and Gama, 2014), TabReD (Rubachev et al., 2025) and the Walmart dataset on Kaggle (`yasserh/walmart-dataset`). Please cite the original sources when using the data.

## Citation

See `CITATION.cff`. The code and results of version 1.0 are archived on Zenodo: https://doi.org/10.5281/zenodo.24110546.

## License

Code: MIT (see `LICENSE`). The datasets keep the licences of their original sources.
