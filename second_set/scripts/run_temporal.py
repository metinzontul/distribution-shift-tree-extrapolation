#!/usr/bin/env python
"""Natural temporal arm (author-defined splits), plus the crossed design:
the constructed extrapolation split applied to the SAME temporal datasets.

Manifest (JSON list), one entry per dataset:
  [{"name": "weather", "path": "data/tabred/weather.parquet", "target": "y",
    "split_col": "split", "tier": "primary"},
   {"name": "walmart", "path": "data/walmart.csv", "target": "Weekly_Sales",
    "time_col": "Date", "boundary": "2012-01-01", "tier": "secondary"}]

Examples
--------
python scripts/run_temporal.py --manifest temporal.json --out results/temporal
python scripts/run_temporal.py --manifest temporal.json --out results/temporal --crossed --max-rows 50000
"""
import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shiftbench.data import load_temporal  # noqa: E402
from shiftbench.models import DEFAULT_MODELS, RunConfig  # noqa: E402
from shiftbench.runner import run_splits, temporal_split  # noqa: E402
from shiftbench.splits import SEVERITIES, build_splits  # noqa: E402
from run_covariate import parse_seeds  # noqa: E402


def subsample(ds, max_rows, seed=0):
    """Identical row subsample for ALL models (fixes the TabPFN-only subsampling)."""
    if max_rows is None or ds.n <= max_rows:
        return ds
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ds.n, max_rows, replace=False))
    ds.X, ds.y = ds.X.iloc[idx].reset_index(drop=True), ds.y[idx]
    if ds.train_mask is not None:
        ds.train_mask, ds.test_mask = ds.train_mask[idx], ds.test_mask[idx]
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="subsample every dataset to this many rows, for every model alike")
    ap.add_argument("--tabpfn-max-rows", type=int, default=10_000)
    ap.add_argument("--tabpfn-policy", choices=["skip", "subsample"], default="skip")
    ap.add_argument("--crossed", action="store_true",
                    help="also run constructed extrap/saturation splits on these datasets")
    ap.add_argument("--crossed-kinds", nargs="*", default=["extrap", "saturation_matched", "id"])
    a = ap.parse_args()

    cfg = RunConfig(n_trials=a.n_trials, n_threads=a.threads, device=a.device,
                    tabpfn_max_rows=a.tabpfn_max_rows, tabpfn_policy=a.tabpfn_policy)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
    for e in json.loads(Path(a.manifest).read_text(encoding="utf-8")):
        ds = load_temporal(e["path"], e["target"], e.get("name"), e.get("split_col", "split"),
                           e.get("time_col"), e.get("boundary"), e.get("drop"))
        ds = subsample(ds, a.max_rows)
        splits = [temporal_split(ds)]
        splits[0].meta["tier"] = e.get("tier", "primary")
        if a.crossed:
            splits += build_splits(ds, kinds=a.crossed_kinds, severities=SEVERITIES)
        run_splits(ds, splits, a.models, parse_seeds(a.seeds), cfg, out)


if __name__ == "__main__":
    main()
