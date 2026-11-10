#!/usr/bin/env python
"""Covariate arm: constructed feature-range splits on real datasets.

Examples
--------
# paper protocol + all revision controls on the full CTR23 suite
python scripts/run_covariate.py --ctr23 --out results/covariate \
    --kinds extrap saturation saturation_robust saturation_matched plateau reverse id

# local files (path:target[:name]); repeatable
python scripts/run_covariate.py --csv data/diamonds.csv:price --out results/cov_local

# leakage sensitivity: choose the shift feature on a discarded 30 % holdout
python scripts/run_covariate.py --ctr23 --selection holdout --out results/cov_holdout --kinds extrap saturation
"""
import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shiftbench.data import load_local, load_openml_ctr23  # noqa: E402

try:  # newer data.py
    from shiftbench.data import load_openml_id  # noqa: E402
except ImportError:  # older data.py: self-contained fallback
    import numpy as np
    import pandas as pd
    from shiftbench.data import Dataset

    def load_openml_id(data_id, name=None, target=None):
        import openml
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
from shiftbench.models import DEFAULT_MODELS, RunConfig  # noqa: E402
from shiftbench.runner import run_splits  # noqa: E402
from shiftbench.splits import SEVERITIES, build_splits  # noqa: E402


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctr23", nargs="*", default=None, help="load OpenML-CTR23 (optionally only these names)")
    ap.add_argument("--csv", action="append", default=[], help="path:target[:name]")
    ap.add_argument("--openml-id", nargs="*", default=[],
                    help="OpenML dataset ids, optionally id:name (e.g. 225:puma8NH 42712:bike_hour)")
    ap.add_argument("--exclude", nargs="*", default=["fps_benchmark"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--kinds", nargs="*", default=["extrap", "saturation"])
    ap.add_argument("--severities", nargs="*", type=float, default=list(SEVERITIES))
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--train-frac", type=float, default=0.30)
    ap.add_argument("--band", type=float, default=0.20)
    ap.add_argument("--selection", choices=["full", "holdout"], default="full")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tabpfn-max-rows", type=int, default=10_000)
    ap.add_argument("--tabpfn-policy", choices=["skip", "subsample"], default="skip")
    ap.add_argument("--no-predictions", action="store_true")
    a = ap.parse_args()

    datasets = []
    if a.ctr23 is not None:
        datasets += load_openml_ctr23(a.ctr23 or None)
    for spec in a.openml_id:
        did, _, nm = spec.partition(":")
        datasets.append(load_openml_id(int(did), nm or None))
    for spec in a.csv:
        parts = spec.split(":")
        datasets.append(load_local(parts[0], parts[1], parts[2] if len(parts) > 2 else None))
    datasets = [d for d in datasets if d.name not in set(a.exclude)]
    if not datasets:
        ap.error("no datasets: use --ctr23, --openml-id and/or --csv")

    cfg = RunConfig(n_trials=a.n_trials, n_threads=a.threads, device=a.device,
                    tabpfn_max_rows=a.tabpfn_max_rows, tabpfn_policy=a.tabpfn_policy)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
    for ds in datasets:
        try:
            splits = build_splits(ds, kinds=a.kinds, severities=a.severities, train_frac=a.train_frac,
                                  band=a.band, selection=a.selection)
        except ValueError as ex:
            print(f"skip {ds.name}: {ex}")
            continue
        run_splits(ds, splits, a.models, parse_seeds(a.seeds), cfg, out,
                   save_predictions=not a.no_predictions)


if __name__ == "__main__":
    main()
