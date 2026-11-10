"""WP3 real-data registry + loaders + inclusion criteria (design protocol §2, §9 minors).

Sources (all reported, nulls included -- G2):
  * local     : housing, air_quality, bike_hour, sgemm (cwr.data loaders; CSV/OpenML).
  * ctr23     : the 35-dataset OpenML-CTR23 regression suite (suite 353). did table cached below so
                the registry is deterministic offline; `ctr23_from_suite()` refreshes it online.
  * gretton   : gretton2008-overlapping sets NOT already in CTR23 -> puma8NH (225), Ailerons (296).
                abalone / kin8nm / california_housing are ALSO in CTR23 -> DEDUPED (dropped here).
  * tabred    : 5 real temporal regression sets (A5 PRIMARY natural shift): sberbank-housing,
                cooking-time, weather, maps-routing, delivery-eta. `load_tabred()` returns a compact
                leakage-safe Dataset + the authored `default` temporal split. `tabred_registry()` is
                kept SEPARATE from registry() -- TabReD drives the temporal arm only, not covariate.

Inclusion (design §9 minor): n>=500, >=3 continuous features (each >=20 distinct values), sigma_Y>0.
Inapplicable datasets are REPORTED (check_inclusion), never silently dropped.

Loaders return the cwr.data.Dataset (RAW numeric X DataFrame, y Series); scaling/splitting happen
downstream per fold (leakage-free). Large sets accept a max_rows train-safe subsample for cost.
"""
from __future__ import annotations

import numpy as np

from . import DATASETS_DIR
from .data import LOCAL as _LOCAL_LOADERS
from .data import _finalize, load_openml_did
from .wp3_shift import continuous_columns

# --------------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------------

# OpenML-CTR23 (suite 353): (did, name, n_instances) -- cached from suite metadata 2026-08-05.
CTR23_TABLE = [
    (44962, "forest_fires", 517), (44967, "student_performance_por", 649),
    (44960, "energy_efficiency", 768), (44994, "cars", 804),
    (44970, "QSAR_fish_toxicity", 908), (44959, "concrete_compressive_strength", 1030),
    (44965, "geographical_origin_of_music", 1059), (44966, "solar_flare", 1066),
    (44987, "socmob", 1156), (41021, "Moneyball", 1232), (44957, "airfoil_self_noise", 1503),
    (44972, "red_wine", 1599), (44958, "auction_verification", 2043), (45402, "space_ga", 3107),
    (44956, "abalone", 4177), (44971, "white_wine", 4898), (44978, "cpu_activity", 8192),
    (44981, "pumadyn32nh", 8192), (44980, "kin8nm", 8192), (44973, "grid_stability", 10000),
    (44990, "brazilian_houses", 10692), (44969, "naval_propulsion_plant", 11934),
    (44983, "miami_housing", 13932), (45012, "fifa", 19178), (44977, "california_housing", 20640),
    (44964, "superconductivity", 21263), (44989, "kings_county", 21613),
    (44993, "health_insurance", 22272), (44992, "fps_benchmark", 24624),
    (44984, "cps88wages", 28155), (44963, "physiochemical_protein", 45730),
    (44976, "sarcos", 48933), (44979, "diamonds", 53940), (44974, "video_transcoding", 68784),
    (44975, "wave_energy", 72000),
]

# gretton2008-overlapping additions NOT already in CTR23 (deduped). abalone/kin8nm/california_housing
# are dropped (present in CTR23_TABLE). Local 'housing' is also CA-housing-family -> flagged, kept.
GRETTON_TABLE = [(225, "puma8NH", 8192), (296, "Ailerons", 13750)]
GRETTON_DEDUPED_OUT = ["abalone (->ctr23 44956)", "kin8nm (->ctr23 44980)",
                       "california_housing (->ctr23 44977; local 'housing' also CA-housing family)"]

LOCAL_NAMES = ["housing", "air_quality", "bike_hour", "sgemm"]

# Datasets with a usable natural temporal / row-order axis (A5). CSV row order is chronological for
# air_quality and bike_hour; TabReD is the real temporal source (see tabred_status()).
ROW_ORDER_TEMPORAL = {"air_quality", "bike_hour"}
TEMPORAL_HINTS = ("year", "yr", "date", "time", "timestamp", "day", "month", "week", "season")


def ctr23_from_suite():
    """Refresh the CTR23 (did, name, n) table from OpenML live (needs network)."""
    import openml
    suite = openml.study.get_suite(353)
    dl = openml.datasets.list_datasets(data_id=list(suite.data), output_format="dataframe")
    return [(int(r.did), str(r["name"]), int(r.NumberOfInstances)) for _, r in dl.iterrows()]


def registry():
    """Full ordered registry of (name, source, id_or_none, n_hint) across all four sources."""
    reg = [(n, "local", None, None) for n in LOCAL_NAMES]
    reg += [(nm, "ctr23", did, n) for (did, nm, n) in CTR23_TABLE]
    reg += [(nm, "gretton", did, n) for (did, nm, n) in GRETTON_TABLE]
    return reg


# --------------------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------------------


def load_one(name, source, did=None, max_rows=None, seed=42, impute=True):
    """Dispatch a single dataset to its loader. Raises on failure; caller guards + reports.

    `impute=False` (WP3 covariate arm) preserves NaN so the pipeline can impute with a fit-train-only
    median inside the fold (leakage-safe). Only the OpenML (ctr23/gretton) path honours it; local
    loaders keep their own load-time handling, and TabReD already imputes train-only in load_tabred."""
    if source == "local":
        ds = _LOCAL_LOADERS[name](impute=impute)   # local loaders honour impute (train-only WP3 path)
        if max_rows and ds.n > max_rows:
            idx = np.random.default_rng(seed).choice(ds.n, max_rows, replace=False)
            from .data import Dataset
            ds = Dataset(ds.name, ds.X.iloc[idx].reset_index(drop=True),
                         ds.y.iloc[idx].reset_index(drop=True), ds.target)
        return ds
    if source in ("ctr23", "gretton"):
        ds = load_openml_did(did, max_rows=max_rows, seed=seed, impute=impute)
        ds.name = name          # human-readable name over the openml_<did> default
        return ds
    if source == "tabred":
        return load_tabred(name, pool_cap=max_rows, seed=seed)
    raise KeyError(f"unknown source {source!r}")


def load_walmart(impute=False):
    """WP3-REVISION Plan B primary (): Walmart Store Sales (Kaggle 'Walmart Recruiting - Store
    Sales Forecasting'), KEYLESS local layout. NOT on OpenML keyless (searched 2026-10-12 -> absent;
    closest rossmann_store_sales lacks the CPI/Fuel_Price/Unemployment macro-exogenous features), so
    the data must be dropped under datasets/walmart/ as either:
      (a) the 3 raw Kaggle files  train.csv + features.csv + stores.csv  (merged here), or
      (b) a single merged  walmart.csv  with a Date column + Weekly_Sales target.
    Raises FileNotFoundError (clear message) when absent -> the runner reports 'data pending'.

    Target = Weekly_Sales. Store/Dept/Type -> one-hot (drop_first); IsHoliday -> int; the continuous
    macro/exogenous features (Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, Size) stay
    numeric (these are the natural-OOR candidates). Date is NOT a feature; it is returned as the
    chronological order key on `ds.order_values` for the real-calendar temporal split."""
    import pandas as pd
    wdir = DATASETS_DIR / "walmart"
    merged = DATASETS_DIR / "walmart.csv"
    if (wdir / "train.csv").exists():
        tr = pd.read_csv(wdir / "train.csv")
        feats = pd.read_csv(wdir / "features.csv") if (wdir / "features.csv").exists() else None
        stores = pd.read_csv(wdir / "stores.csv") if (wdir / "stores.csv").exists() else None
        df = tr
        if feats is not None:
            on = [c for c in ("Store", "Date", "IsHoliday") if c in tr.columns and c in feats.columns]
            df = df.merge(feats, on=on, how="left", suffixes=("", "_f"))
        if stores is not None:
            df = df.merge(stores, on="Store", how="left")
    elif merged.exists():
        df = pd.read_csv(merged)
    elif (wdir / "walmart.csv").exists():
        df = pd.read_csv(wdir / "walmart.csv")
    else:
        raise FileNotFoundError(
            f"Walmart Store Sales not found under {wdir} or {merged}. It is NOT keyless on OpenML; "
            "drop the Kaggle files (train.csv+features.csv+stores.csv) or a merged walmart.csv there.")
    if "Weekly_Sales" not in df.columns:
        raise KeyError(f"Walmart: no Weekly_Sales target column (have {list(df.columns)[:12]}...)")
    df = df[df["Weekly_Sales"].notna()].copy()
    # Robust date parse: the single-file Walmart.csv ships DD-MM-YYYY, which the month-first default
    # misreads (days 13-31 -> NaT, chronology scrambled). Parse both ways and keep whichever yields
    # fewer NaT (day-first vs month-first), so either calendar convention loads correctly.
    _mf = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    _df = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    order = _df if int(_df.isna().sum()) < int(_mf.isna().sum()) else _mf
    df = df.loc[order.notna()].copy(); order = order[order.notna()]
    srt = np.argsort(order.values.astype("datetime64[ns]").astype("int64"), kind="stable")
    df = df.iloc[srt].reset_index(drop=True)
    order_values = order.values.astype("datetime64[ns]").astype("int64")[srt].astype(float)
    y = df["Weekly_Sales"]
    drop = [c for c in ("Weekly_Sales", "Date") if c in df.columns]
    X = df.drop(columns=drop)
    cat_cols = [c for c in ("Store", "Dept", "Type") if c in X.columns]
    if "IsHoliday" in X.columns:
        X["IsHoliday"] = X["IsHoliday"].astype(float)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    ds = _finalize(X, y, "walmart", "Weekly_Sales", impute=impute)
    ds.order_values = order_values[: ds.n]      # aligned after _finalize's y-NaN drop (none expected)
    ds.source = "walmart"
    return ds


def temporal_order(ds):
    """Return a temporal/order key (1-D array) for `ds` or None. air_quality/bike_hour use the
    chronological CSV row order; otherwise a numeric column whose name hints at time (>=20 unique)
    is used. TabReD is the authoritative temporal source (this is a best-effort fallback)."""
    if ds.name in ROW_ORDER_TEMPORAL:
        return np.arange(ds.n, dtype=float)
    for c in ds.X.columns:
        lc = str(c).lower()
        if any(h in lc for h in TEMPORAL_HINTS) and int(ds.X[c].nunique()) >= 20:
            return ds.X[c].values.astype(float)
    return None


# --------------------------------------------------------------------------------------------------
# Inclusion criteria
# --------------------------------------------------------------------------------------------------


# G2: computational-infeasibility exclusions (REPORTED, not silent; WP3 §11 build-time, 2026-08-16).
# fps_benchmark: very high feature count -> FT-Transformer capped-batch fit ~40min -> ~97min/cell
# (14x the next slowest dataset; 59% of all compute alone). Excluded for tractability, reported in paper.
COMPUTE_EXCLUDE = {"fps_benchmark": "FT-Transformer computationally infeasible (high feature count -> ~97min/cell)"}

# Covariate-ARM roster exclusions (2026-08-27, REPORTED). air_quality: predictive features (pollutant
# sensors) are 30-90% NaN -> no leakage-safe covariate-range split axis (stays a TEMPORAL proxy).
# housing: dedup of ctr23 california_housing (same survey: identical n=20640, target, income-split
# spearman/slope_ratio; local adds ocean_proximity dummies + raw total_bedrooms NaN). The analysis
# roster authority is results/wp3_analysis/wp3_analyze.py:COVARIATE_ARM_EXCLUDE. Final covariate N=27.
COVARIATE_ARM_EXCLUDE = {
    "air_quality": "predictive features 30-90% NaN -> no leakage-safe covariate axis (temporal proxy kept)",
    "housing": "dedup: same survey as ctr23 california_housing (identical n/target/income-split stats)",
}


def check_inclusion(ds, min_n=500, min_continuous=3, min_unique=20):
    """design §9 minor gate. Returns a dict with the decision + reasons (reported, not silent)."""
    cont = continuous_columns(ds.X, min_unique=min_unique)
    sigma_y = float(np.std(np.asarray(ds.y, float)))
    n_ok = ds.n >= min_n
    cont_ok = len(cont) >= min_continuous
    sig_ok = sigma_y > 0
    compute_ok = ds.name not in COMPUTE_EXCLUDE
    reasons = []
    if not n_ok:
        reasons.append(f"n={ds.n}<{min_n}")
    if not cont_ok:
        reasons.append(f"continuous>=({min_unique}u)={len(cont)}<{min_continuous}")
    if not sig_ok:
        reasons.append("sigma_Y=0")
    if not compute_ok:
        reasons.append(f"compute-infeasible: {COMPUTE_EXCLUDE[ds.name]}")
    return {"name": ds.name, "n": ds.n, "d": ds.d, "n_continuous": len(cont), "sigma_y": sigma_y,
            "included": bool(n_ok and cont_ok and sig_ok and compute_ok), "reasons": reasons,
            "continuous_cols": cont}


# --------------------------------------------------------------------------------------------------
# TabReD: 5 real temporal regression sets = A5-PRIMARY natural-shift evidence
# --------------------------------------------------------------------------------------------------
#
# On-disk layout (per set, VERIFIED 2026-08-05):
#   x_num.npy  (N x F float32)  main numeric features        -> USED (feature matrix)
#   x_bin.npy  (N x k float32)  binary features, OPTIONAL    -> USED (concatenated numerically)
#   x_cat.npy  (N x k int64)    categorical codes, OPTIONAL  -> NOT a feature (dropped)
#   x_meta.npy (N x m int64)    metadata; col 0 = TIMESTAMP  -> NOT a feature (this is the time axis)
#   y.npy      (N float32)      regression target
#   splits/default/{train,val,test}.npy = CONTIGUOUS chronological index blocks (train early, val
#     mid, test future) = the authored TIME split we use (random-*/sliding-window-* exist but are NOT
#     used; `default` is the temporal one). x_num carries ~3-8% NaN -> TRAIN-only median imputation.

from .data import Dataset  # noqa: E402  (compact TabReD Dataset construction)

TABRED_DIR = DATASETS_DIR / "tabred"
# (name, n_hint, n_features_hint) -- n_features = x_num + x_bin cols (VERIFIED on disk). maps-routing
# has 984 features (> TabPFN's ~500 ceiling) so TabPFN is dropped there; GBDT/NN stay (see
# wp3_models.TABPFN_MAX_FEATURES). All 5 sets are >10k rows (regime driven by the per-cell fit cap).
TABRED_REGISTRY = [
    ("sberbank-housing", 28321, 382), ("cooking-time", 319986, 189),
    ("weather", 423795, 103), ("delivery-eta", 416451, 222),
    # maps-routing (340981, 984) EXCLUDED 2026-08-19: 984 feat -> FT-Transformer capped-batch
    # computationally infeasible (~fps_benchmark-scale, hours/cell). Reported (G2). 4/5 temporal sets stand.
]
TABRED_SETS = [n for (n, _, _) in TABRED_REGISTRY]


def tabred_registry():
    """The 5 TabReD temporal-primary sets as (name, 'tabred', None, n_hint). Kept SEPARATE from the
    covariate registry() (CTR23 + gretton + local): TabReD drives the A5 temporal arm ONLY."""
    return [(n, "tabred", None, n_hint) for (n, n_hint, _) in TABRED_REGISTRY]


def _tabred_dir(set_name):
    return TABRED_DIR / set_name


def _maybe_extract_tabred(set_name):
    """Idempotent extract: if `<set>/` is missing but `<set>.tabred` (a gzip tar) is present, unpack
    it in place. No-op when the set directory already exists. Never raises past a clear message."""
    d = _tabred_dir(set_name)
    if (d / "x_num.npy").exists():
        return d
    arch = TABRED_DIR / f"{set_name}.tabred"
    if arch.exists():
        import tarfile
        d.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(arch, "r:gz") as t:      # .tabred = gzip tar (magic 1f 8b)
                t.extractall(d)
        except tarfile.ReadError as e:
            raise RuntimeError(f"{arch} is not a gzip tar ({e}); expected TabReD archive") from e
        # some archives nest a single top-level dir; hoist the npys up if so
        if not (d / "x_num.npy").exists():
            for sub in d.iterdir():
                if sub.is_dir() and (sub / "x_num.npy").exists():
                    for f in sub.iterdir():
                        f.rename(d / f.name)
                    break
    if not (d / "x_num.npy").exists():
        raise FileNotFoundError(f"TabReD set {set_name!r} not found under {TABRED_DIR} "
                                "(neither extracted dir nor <set>.tabred archive)")
    return d


def tabred_split_indices(set_name):
    """Read the authored `default` (temporal) split -> {'train','val','test'} int index arrays."""
    d = _maybe_extract_tabred(set_name)
    sp = d / "splits" / "default"
    return {k: np.load(sp / f"{k}.npy").astype(np.int64) for k in ("train", "val", "test")}


def tabred_meta(set_name):
    """Cheap mmap-based metadata for one TabReD set (NO full load): shapes, feature count, split
    sizes, TabPFN feasibility. Used by the effective-N projection and status without paying the
    x_num read cost (maps-routing x_num alone is 1.3 GB)."""
    from .wp3_models import TABPFN_MAX_FEATURES
    d = _maybe_extract_tabred(set_name)
    xnum = np.load(d / "x_num.npy", mmap_mode="r")
    n, f_num = int(xnum.shape[0]), int(xnum.shape[1])
    f_bin = int(np.load(d / "x_bin.npy", mmap_mode="r").shape[1]) if (d / "x_bin.npy").exists() else 0
    n_features = f_num + f_bin
    sp = tabred_split_indices(set_name)
    return {"name": set_name, "n": n, "n_features": n_features, "n_num": f_num, "n_bin": f_bin,
            "n_train": int(len(sp["train"])), "n_val": int(len(sp["val"])),
            "n_test": int(len(sp["test"])), "tabpfn_feasible": bool(n_features <= TABPFN_MAX_FEATURES)}


def _cap_ordered(idx, cap, rng):
    """Seeded subsample of `idx` to at most `cap`, then SORTED ascending so the natural TEMPORAL
    order inside the (chronological) block is preserved (design pin: 'temporal-sıra korunarak')."""
    idx = np.asarray(idx, np.int64)
    if cap and len(idx) > cap:
        idx = rng.choice(idx, int(cap), replace=False)
    return np.sort(idx)


def load_tabred(set_name, pool_cap=None, seed=42):
    """Load one TabReD set as a COMPACT, leakage-safe cwr Dataset for the A5 temporal arm.

    Feature matrix X = x_num (+ x_bin if present), numeric only. x_cat and x_meta are NOT features
    (x_meta col 0 is the timestamp/time axis). The authored `default` temporal split (train early /
    val mid / test future) is exposed on the returned object.

    Memory-safe by design: rather than materialising the full arrays (maps-routing x_num is 1.3 GB),
    each split block is optionally seeded-subsampled to `pool_cap` rows (temporal order preserved) and
    ONLY those rows are read from the memory-mapped npy. The returned Dataset is the concatenation
    train-block + val-block + test-block (in chronological order); `ds.splits` gives LOCAL index
    arrays into that compact matrix. Attached attributes:
      ds.source        = 'tabred'
      ds.splits        = {'train','val','test'}  local int indices (train < val < test)
      ds.sigma_ref     = FULL train-block Y-std (train-only, seed-independent) = sigma_Y^ref (temporal)
      ds.n_features    = feature count (drives TabPFN feasibility)
      ds.n_full        = true dataset N (before any subsample)

    Imputation: x_num carries ~3-8% NaN. NaNs (and any +/-inf) are filled with the compact TRAIN
    block's per-column median (TRAIN-only -> leakage-safe); an all-NaN train column falls back to 0.
    """
    d = _maybe_extract_tabred(set_name)
    sp_full = tabred_split_indices(set_name)
    rng = np.random.default_rng(seed)
    # per-period seeded subsample (temporal order preserved); global indices into the on-disk arrays
    g_train = _cap_ordered(sp_full["train"], pool_cap, rng)
    g_val = _cap_ordered(sp_full["val"], pool_cap, rng)
    g_test = _cap_ordered(sp_full["test"], pool_cap, rng)
    g_all = np.concatenate([g_train, g_val, g_test])          # compact row order = chronological

    xnum = np.load(d / "x_num.npy", mmap_mode="r")
    cols_num = np.asarray(xnum[g_all], np.float64)            # fancy-index reads ONLY selected rows
    has_bin = (d / "x_bin.npy").exists()
    if has_bin:
        xbin = np.load(d / "x_bin.npy", mmap_mode="r")
        cols_bin = np.asarray(xbin[g_all], np.float64)
        Xc = np.concatenate([cols_num, cols_bin], axis=1)
    else:
        Xc = cols_num
    Xc[~np.isfinite(Xc)] = np.nan                             # unify inf -> NaN for imputation

    n_train = len(g_train)
    # TRAIN-only median imputation (leakage-safe): medians from the compact train block only
    train_block = Xc[:n_train]
    with np.errstate(all="ignore"):
        med = np.nanmedian(train_block, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)                # all-NaN train column -> 0
    nan_mask = np.isnan(Xc)
    if nan_mask.any():
        Xc[nan_mask] = np.take(med, np.where(nan_mask)[1])

    y_full = np.load(d / "y.npy")                             # small; full target for a stable ref
    yc = np.asarray(y_full[g_all], np.float64)
    # sigma_Y^ref (temporal) = FULL train-period Y-std, TRAIN-only, seed-independent, dataset-fixed
    sigma_ref = float(np.std(np.asarray(y_full[sp_full["train"]], np.float64)))
    if not (sigma_ref > 0):
        sigma_ref = float(np.std(yc)) or 1.0

    f_num = cols_num.shape[1]
    names = [f"num_{i}" for i in range(f_num)]
    if has_bin:
        names += [f"bin_{i}" for i in range(Xc.shape[1] - f_num)]
    import pandas as pd
    ds = Dataset(set_name, pd.DataFrame(Xc, columns=names), pd.Series(yc), "y")
    n0, n1 = n_train, n_train + len(g_val)
    ds.splits = {"train": np.arange(0, n0), "val": np.arange(n0, n1),
                 "test": np.arange(n1, len(g_all))}
    ds.source = "tabred"
    ds.sigma_ref = sigma_ref
    ds.n_features = Xc.shape[1]
    ds.n_full = int(xnum.shape[0])
    return ds


def tabred_status():
    """Report availability + shape/feasibility of the 5 TabReD temporal sets (mmap metadata, no full
    load). A5-PRIMARY: these drive the temporal arm; air_quality/bike_hour row-order are the SECONDARY
    fallback. Gated download (https://github.com/yandex-research/tabred) -> reports, never raises."""
    present, meta, missing = [], {}, []
    for s in TABRED_SETS:
        try:
            meta[s] = tabred_meta(s)
            present.append(s)
        except Exception as e:  # noqa: BLE001 (status must never raise)
            missing.append(s)
            meta[s] = {"error": repr(e)[:120]}
    return {"dir": str(TABRED_DIR), "available": present, "missing": missing, "meta": meta,
            "instructions": ("TabReD is gated: clone yandex-research/tabred, request the data, and "
                             f"place each set under {TABRED_DIR}/<set>/ (or leave <set>.tabred to "
                             "auto-extract). default split = A5-PRIMARY temporal; missing -> temporal "
                             "arm falls back to air_quality/bike_hour row-order (SECONDARY).")}
