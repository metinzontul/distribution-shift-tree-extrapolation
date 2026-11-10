"""Dataset containers and loaders.

Three sources are supported:
  * OpenML-CTR23 suite (needs the `openml` package and internet access);
  * any local CSV / parquet file (covariate arm or temporal arm);
  * temporal datasets with an author-defined split, given either as a
    `split` column ("train"/"test") or as a time column plus a boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CTR23_SUITE_ID = 353  # OpenML-CTR23 benchmark suite


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    num_cols: list = field(default_factory=list)
    cat_cols: list = field(default_factory=list)
    # temporal arm only: boolean masks for the author-defined split
    train_mask: Optional[np.ndarray] = None
    test_mask: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.y)

    def describe(self) -> dict:
        return {
            "dataset": self.name,
            "n_rows": self.n,
            "n_num": len(self.num_cols),
            "n_cat": len(self.cat_cols),
            "missing_frac": float(self.X.isna().mean().mean()),
        }


def _infer_columns(X: pd.DataFrame, max_cat_unique: int = 0):
    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
            if max_cat_unique and X[c].nunique() <= max_cat_unique:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def from_frame(name: str, df: pd.DataFrame, target: str, drop: Optional[list] = None) -> Dataset:
    df = df.copy()
    df = df[~df[target].isna()].reset_index(drop=True)
    y = df[target].to_numpy(dtype=float)
    X = df.drop(columns=[target] + list(drop or []))
    num_cols, cat_cols = _infer_columns(X)
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return Dataset(name=name, X=X, y=y, num_cols=num_cols, cat_cols=cat_cols)


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:  # e.g. CSVs saved by Excel on a Turkish Windows
        return pd.read_csv(path, encoding="cp1254")


def load_local(path: str | Path, target: str, name: Optional[str] = None,
               drop: Optional[list] = None) -> Dataset:
    """Covariate-arm dataset from a local file."""
    path = Path(path)
    return from_frame(name or path.stem, load_table(path), target, drop)


def load_temporal(path: str | Path, target: str, name: Optional[str] = None,
                  split_col: Optional[str] = "split", time_col: Optional[str] = None,
                  boundary=None, drop: Optional[list] = None) -> Dataset:
    """Temporal-arm dataset with an author-defined split.

    Either `split_col` holds "train"/"test" labels (e.g. TabReD exports), or
    `time_col` + `boundary` define train = time < boundary, test = time >= boundary.
    Rows labelled "val" are folded into train (models carve their own
    validation slices from training data).
    """
    path = Path(path)
    df = load_table(path)
    if split_col and split_col in df.columns:
        lab = df[split_col].astype(str).str.lower()
        train = lab.isin(["train", "val", "validation"]).to_numpy()
        test = (lab == "test").to_numpy()
        extra_drop = [split_col]
    elif time_col is not None and boundary is not None:
        t = pd.to_datetime(df[time_col], errors="ignore")
        b = pd.to_datetime(boundary) if not np.issubdtype(np.asarray(t).dtype, np.number) else boundary
        train = (t < b).to_numpy()
        test = ~train
        extra_drop = [time_col]
    else:
        raise ValueError("Provide split_col present in the file, or time_col and boundary.")
    keep = ~df[target].isna().to_numpy()
    df, train, test = df[keep].reset_index(drop=True), train[keep], test[keep]
    ds = from_frame(name or path.stem, df, target, (drop or []) + extra_drop)
    ds.train_mask, ds.test_mask = train, test
    return ds


def load_openml_id(data_id: int, name: Optional[str] = None, target: Optional[str] = None) -> Dataset:
    """Any OpenML dataset by id; uses its default target and drops its
    ignore-attributes (e.g. casual/registered in Bike_Sharing_Demand, id 42712)."""
    import openml  # optional dependency

    d = openml.datasets.get_dataset(data_id, download_data=True)
    target = target or d.default_target_attribute
    X, y, cat_ind, attr = d.get_data(target=target, dataset_format="dataframe",
                                     include_ignore_attribute=False)
    ignore = d.ignore_attribute or []
    if isinstance(ignore, str):
        ignore = [ignore]
    keep_cols = [a for a in attr if a not in ignore]
    cat_cols = [a for a, c in zip(attr, cat_ind) if c and a in keep_cols]
    X = X[keep_cols].reset_index(drop=True)
    keep = ~pd.isna(y).to_numpy()
    X, y = X[keep].reset_index(drop=True), np.asarray(y, dtype=float)[keep]
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    num_cols = [a for a in keep_cols if a not in cat_cols]
    return Dataset(name=name or d.name, X=X, y=y, num_cols=num_cols, cat_cols=cat_cols)


def load_openml_ctr23(names: Optional[list] = None, cache_dir: Optional[str] = None) -> list[Dataset]:
    """Load (a subset of) the OpenML-CTR23 regression suite."""
    import openml  # optional dependency

    if cache_dir:
        openml.config.set_root_cache_directory(cache_dir)
    suite = openml.study.get_suite(CTR23_SUITE_ID)
    out = []
    for tid in suite.tasks:
        task = openml.tasks.get_task(tid, download_splits=False)
        d = task.get_dataset()
        if names and d.name not in names:
            continue
        X, y, cat_ind, attr = d.get_data(target=task.target_name, dataset_format="dataframe")
        X = X.reset_index(drop=True)
        keep = ~pd.isna(y).to_numpy()
        X, y = X[keep].reset_index(drop=True), np.asarray(y, dtype=float)[keep]
        cat_cols = [a for a, c in zip(attr, cat_ind) if c]
        num_cols = [a for a in attr if a not in cat_cols]
        for c in cat_cols:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        out.append(Dataset(name=d.name, X=X, y=y, num_cols=num_cols, cat_cols=cat_cols))
    return out
