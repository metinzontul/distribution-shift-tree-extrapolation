# Environment lock (reproducibility snapshot)

Pinned software environment for the experiments in this repository (first set of experiments). The complete pinned dependency list is in `environment.lock.txt` (the output of `pip freeze` in the `torch` conda environment, 89 packages).

## Platform

- OS: Windows 11
- Python: 3.11.15 (conda environment `torch`)
- PyTorch: 2.11.0+cu128 (CUDA 12.8)
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)
- Compute placement: the gradient-boosted models (CatBoost, XGBoost, LightGBM) and the MLP run on CPU; FT-Transformer and TabPFN run on the single GPU.

## Key package versions

| Package | Version |
| --- | --- |
| torch | 2.11.0+cu128 |
| catboost | 1.2.10 |
| xgboost | 3.2.0 |
| lightgbm | 4.7.0 |
| tabpfn | 8.2.0 |
| rtdl_revisiting_models (FT-Transformer) | 0.0.2 |
| scikit-learn | 1.9.0 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scipy | 1.17.1 |
| openml | 0.15.1 |

## Reproduce

1. Create a Python 3.11 environment and install the pinned dependencies:
   ```
   pip install -r first_set/environment/environment.lock.txt
   ```
   PyTorch is pinned to a CUDA 12.8 build; on a different platform install the matching PyTorch wheel from the official index, then install the remaining packages from the lockfile.
2. Run the pipeline from `first_set/`, for example:
   ```
   python experiments/wp3_pipeline.py --arm all --seeds 15 --hpo-budget 8
   ```
3. Every reported number averages 15 random seeds (0 through 14). The protocol is leakage-free: scaling, imputation, hyperparameter tuning, and early stopping are fit on the training portion only, and the shifted test partition is never touched during model fitting or selection.
