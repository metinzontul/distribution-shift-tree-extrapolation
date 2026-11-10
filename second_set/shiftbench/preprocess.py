"""Train-only preprocessing: median imputation + standardisation for numeric
columns, most-frequent imputation + one-hot encoding for categoricals.
Re-fitted on the training rows of every split, never on test rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     min_frequency=None)),
        ]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def fit_transform_split(X: pd.DataFrame, num_cols, cat_cols, train_idx, test_idx):
    """Return float32 design matrices (train, test) and the fitted transformer."""
    pre = make_preprocessor(num_cols, cat_cols)
    if cat_cols:  # pandas StringDtype + pd.NA breaks SimpleImputer: use object + np.nan
        X = X.copy()
        for c in cat_cols:
            col = X[c].astype(object)
            X[c] = col.where(pd.notna(col), np.nan)
    Xtr = pre.fit_transform(X.iloc[train_idx])
    Xte = pre.transform(X.iloc[test_idx])
    return np.asarray(Xtr, dtype=np.float32), np.asarray(Xte, dtype=np.float32), pre
