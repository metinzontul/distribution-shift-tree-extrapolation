"""Dataset loaders. Local 4 (housing, air_quality, bike_hour, airlines) + OpenML-CTR23 suite.

Returns RAW numeric DataFrames (one-hot, imputed). Scaling/splitting happen downstream so they
can be redone per fold without leakage (see experts.py / killcheck.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import DATASETS_DIR


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    target: str

    @property
    def n(self):
        return len(self.y)

    @property
    def d(self):
        return self.X.shape[1]

    def __repr__(self):
        return f"Dataset({self.name!r}, n={self.n}, d={self.d}, target={self.target!r})"


def _finalize(X, y, name, target, impute=True):
    """Coerce to numeric, drop non-finite y rows, return a RAW numeric Dataset.

    `impute` (default True) is the historical behaviour: NaN/inf are filled with the WHOLE-dataset
    column median at load time, before any split. That is fine for WP2 and every NaN-free dataset,
    but for the leakage-free WP3 covariate arm it would leak the test-region marginal into the
    train-region fill. Pass `impute=False` to PRESERVE NaN (inf still mapped to NaN) so the WP3
    covariate pipeline can impute with a fit-train-only median inside the fold (wp3_pipeline.run_cell).
    For a NaN-free dataset impute=False and impute=True yield byte-identical X (fill is a no-op)."""
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.float64)
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if impute:
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna() & np.isfinite(y)
    X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
    return Dataset(name, X.astype(np.float64), y.astype(np.float64), target)


def load_housing(impute=True):
    df = pd.read_csv(DATASETS_DIR / "housing.csv")
    y = df["median_house_value"]
    X = pd.get_dummies(df.drop(columns=["median_house_value"]), drop_first=True)
    return _finalize(X, y, "housing", "median_house_value", impute=impute)


def load_bike_hour(impute=True):
    df = pd.read_csv(DATASETS_DIR / "bike+sharing+dataset" / "hour.csv")
    y = df["cnt"]
    X = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])
    X = pd.get_dummies(X, columns=["season", "mnth", "weekday", "weathersit", "hr"], drop_first=True)
    return _finalize(X, y, "bike_hour", "cnt", impute=impute)


def load_air_quality(impute=True):
    df = pd.read_csv(DATASETS_DIR / "AirQualityUCI.csv", sep=";", decimal=",", na_values=[-200])
    df = df.dropna(axis=1, how="all")
    if "Time" in df.columns:
        df["Time"] = df["Time"].astype(str).str.replace(".", ":", regex=False)
    df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                    errors="coerce", dayfirst=True)
    df["Hour"] = df["Datetime"].dt.hour
    df["Weekday"] = df["Datetime"].dt.weekday
    df["Month"] = df["Datetime"].dt.month
    df["IsWeekend"] = df["Weekday"].isin([5, 6]).astype(int)
    target = "NOx(GT)"
    df = df[df[target].notna()].copy()
    X = df.drop(columns=[c for c in ["Date", "Time", "Datetime", target] if c in df.columns])
    y = df[target]
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    if {"T", "RH"}.issubset(X.columns):
        X["T_RH"] = X["T"] * X["RH"]
    if {"CO(GT)", "NO2(GT)"}.issubset(X.columns):
        X["CO_NO2_ratio"] = X["CO(GT)"] / (X["NO2(GT)"] + 1e-3)
    if "Hour" in X.columns:
        X["Hour_sin"] = np.sin(2 * np.pi * X["Hour"] / 24)
        X["Hour_cos"] = np.cos(2 * np.pi * X["Hour"] / 24)
    return _finalize(X, y, "air_quality", target, impute=impute)


def load_airlines(top_cat=25, max_rows=None, seed=42, impute=True):
    df = pd.read_csv(DATASETS_DIR / "airlines_train_regression.csv")
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    y = df["DepDelay"]
    dest_col = "Dest" if "Dest" in df.columns else "Des"  # this CSV uses truncated 'Des'

    def hhmm(col):
        col = col.astype(int)
        return (col // 100) % 24, col % 100

    dh, dm = hhmm(df["CRSDepTime"])
    ah, am = hhmm(df["CRSArrTime"])
    X = df.drop(columns=["DepDelay", "CRSDepTime", "CRSArrTime"])
    X["CRSDepHour"], X["CRSDepMin"], X["CRSArrHour"], X["CRSArrMin"] = dh, dm, ah, am
    # cap high-cardinality categoricals (airports/carriers) to top-K + OTHER to bound feature count
    for c in ["UniqueCarrier", "Origin", dest_col]:
        top = X[c].value_counts().nlargest(top_cat).index
        X[c] = X[c].where(X[c].isin(top), other="OTHER")
    X = pd.get_dummies(X, columns=["UniqueCarrier", "Origin", dest_col], drop_first=True)
    return _finalize(X, y, "airlines", "DepDelay", impute=impute)


def load_sgemm(max_rows=None, seed=42, impute=True):
    """SGEMM GPU kernel performance (OpenML 43144, ~241k rows). Target = log mean runtime.
    Big AND high-signal (LightGBM R2 ~ 0.985), so an accuracy win here is meaningful."""
    import openml
    d = openml.datasets.get_dataset(43144)
    X, _, _, _ = d.get_data()
    run_cols = [c for c in X.columns if c.lower().startswith("run")]
    yt = np.log(X[run_cols].astype(float).mean(axis=1).values)
    Xf = X.drop(columns=run_cols)
    if max_rows and len(Xf) > max_rows:
        idx = np.random.default_rng(seed).choice(len(Xf), max_rows, replace=False)
        Xf, yt = Xf.iloc[idx], yt[idx]
    return _finalize(Xf, pd.Series(yt), "sgemm", "log_mean_runtime", impute=impute)


def load_openml_did(did, max_rows=None, seed=42, impute=True):
    """Generic OpenML dataset loader by dataset id, using its default regression target.

    `impute=False` preserves NaN for the leakage-free WP3 covariate arm (see _finalize)."""
    import openml
    d = openml.datasets.get_dataset(did)
    tgt = d.default_target_attribute
    X, y, _, _ = d.get_data(target=tgt)
    yarr = np.asarray(y, dtype=float)
    X = pd.get_dummies(X, drop_first=True).reset_index(drop=True)
    if max_rows and len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, yarr = X.iloc[idx].reset_index(drop=True), yarr[idx]
    return _finalize(X, pd.Series(yarr), f"openml_{did}", str(tgt), impute=impute)


LOCAL = {"housing": load_housing, "air_quality": load_air_quality,
         "bike_hour": load_bike_hour, "airlines": load_airlines, "sgemm": load_sgemm}
DEFAULT_DATASETS = ["housing", "air_quality", "bike_hour"]


def load(name):
    if name not in LOCAL:
        raise KeyError(f"unknown local dataset {name!r}; known: {list(LOCAL)}")
    return LOCAL[name]()


# ---------- OpenML-CTR23 (full-run / journal backbone; not used in the pilot kill-check) ----------
CTR23_SUITE_ID = 353  # OpenML-CTR23 benchmark suite


def ctr23_task_ids():
    import openml
    return list(openml.study.get_suite(CTR23_SUITE_ID).tasks)


def load_openml_task(task_id, max_rows=None, seed=42):
    """Load one OpenML task as a Dataset (one-hot + impute). Optional subsampling for cost control."""
    import openml
    task = openml.tasks.get_task(task_id, download_splits=False)
    ds = task.get_dataset()
    X, y, cat, names = ds.get_data(target=task.target_name)
    X = pd.get_dummies(X, drop_first=True)
    if max_rows and len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    return _finalize(X, pd.Series(np.asarray(y, float)), f"ctr23_{task_id}", str(task.target_name))
