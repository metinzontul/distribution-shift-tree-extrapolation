"""Resumable experiment runner. One CSV row per (split, model, seed)."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset
from .diagnostics import shift_report, top_feature_by_train_correlation
from .models import RunConfig, SkipModel, fit_with_search, rmse
from .preprocess import fit_transform_split
from .splits import Split

KEY = ["dataset", "kind", "severity", "model", "seed"]


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=KEY)
    df["severity"] = df["severity"].fillna(-1)
    return set(map(tuple, df[KEY].astype(str).to_numpy()))


def _append(path: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if not path.exists():
        df.to_csv(path, index=False)
        return
    cols = list(pd.read_csv(path, nrows=0).columns)
    if set(df.columns) <= set(cols):
        df.reindex(columns=cols).to_csv(path, mode="a", header=False, index=False)
    else:  # new columns appeared (e.g. an error note): rewrite with the union
        old = pd.read_csv(path)
        pd.concat([old, df], ignore_index=True).to_csv(path, index=False)


def temporal_split(ds: Dataset) -> Split:
    if ds.train_mask is None:
        raise ValueError(f"{ds.name} has no author-defined split")
    tr, te = np.where(ds.train_mask)[0], np.where(ds.test_mask)[0]
    f, r = top_feature_by_train_correlation(ds.X, ds.y, ds.num_cols, tr)
    return Split(ds.name, "temporal", np.nan, f, None, tr, te, meta={"abs_rho": r})


def run_splits(ds: Dataset, splits: list[Split], models: list[str], seeds: list[int],
               cfg: RunConfig, out_dir: str | Path, save_predictions: bool = True,
               verbose: bool = True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path, diag_path = out_dir / "results.csv", out_dir / "diagnostics.csv"
    done = _done_keys(res_path)
    diag_done = set()
    if diag_path.exists():
        d = pd.read_csv(diag_path, usecols=["dataset", "kind", "severity"])
        diag_done = set(map(tuple, d.fillna(-1).astype(str).to_numpy()))

    for sp in splits:
        sev_key = str(-1.0 if np.isnan(sp.severity) else float(sp.severity))
        y = sp.y if sp.y is not None else ds.y
        todo = [(m, s) for m in models for s in seeds
                if (ds.name, sp.kind, sev_key, m, str(s)) not in done]
        dkey = (ds.name, sp.kind, sev_key)
        if not todo and dkey in diag_done:
            continue
        Xtr, Xte, _ = fit_transform_split(ds.X, ds.num_cols, ds.cat_cols, sp.train_idx, sp.test_idx)
        ytr, yte = y[sp.train_idx], y[sp.test_idx]
        sigma_train = float(np.std(ytr))          # manuscript eq. sigma_Y^ref
        sigma_full = float(np.std(y))             # alternative normaliser
        if dkey not in diag_done:
            key_feat = sp.feature or sp.meta.get("extrap_feature")
            rep = shift_report(ds.X, y, ds.num_cols, sp.train_idx, sp.test_idx, key_feat)
            _append(diag_path, [{"dataset": ds.name, "kind": sp.kind, "severity": sp.severity,
                                 "feature": sp.feature, "direction": sp.direction,
                                 "n_train": len(ytr), "n_test": len(yte),
                                 "sigma_train": sigma_train, "sigma_full": sigma_full,
                                 **{k: v for k, v in sp.meta.items() if np.isscalar(v)}, **rep}])
        if save_predictions:
            pdir = out_dir / "predictions" / ds.name
            pdir.mkdir(parents=True, exist_ok=True)
            np.save(pdir / f"{sp.kind}_{sev_key}_test_idx.npy", sp.test_idx)
            np.save(pdir / f"{sp.kind}_{sev_key}_y.npy", yte)
        for m, s in todo:
            row = {"dataset": ds.name, "kind": sp.kind, "severity": sp.severity, "model": m,
                   "seed": s, "feature": sp.feature, "direction": sp.direction,
                   "n_train": len(ytr), "n_test": len(yte),
                   "sigma_train": sigma_train, "sigma_full": sigma_full,
                   "status": None, "rmse": np.nan, "nood": np.nan, "nood_full": np.nan,
                   "val_rmse": np.nan, "params": "", "note": ""}
            t0 = time.time()
            try:
                model, params, info = fit_with_search(m, Xtr, ytr, s, cfg)
                pred = np.asarray(model.predict(Xte), dtype=float).ravel()
                e = rmse(pred, yte)
                row.update(status="ok", rmse=e, nood=e / sigma_train if sigma_train > 0 else np.nan,
                           nood_full=e / sigma_full if sigma_full > 0 else np.nan,
                           val_rmse=info["val_rmse"], params=json.dumps(params, default=str))
                if save_predictions:
                    np.save(pdir / f"{sp.kind}_{sev_key}_{m}_{s}.npy", pred.astype(np.float32))
            except SkipModel as ex:
                row.update(status="skipped", note=str(ex))
            except Exception as ex:  # keep the sweep alive, log the failure
                row.update(status="error", note=f"{type(ex).__name__}: {ex}")
                if os.environ.get("SHIFTBENCH_DEBUG"):
                    traceback.print_exc()
            row["seconds"] = time.time() - t0
            _append(res_path, [row])
            if verbose:
                print(f"[{ds.name}] {sp.kind:<18} sev={sp.severity!s:<5} {m:<16} seed={s:<2} "
                      f"{row['status']:<7} nOOD={row.get('nood', float('nan')):.3f} "
                      f"({row['seconds']:.1f}s)", flush=True)
