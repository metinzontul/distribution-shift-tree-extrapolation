#!/usr/bin/env python
"""Synthetic phase with an explicit generator (see shiftbench/synthetic.py).

Each (geometry, shape, replicate) is one synthetic *dataset*; analysis treats
replicates, not seeds x cells, as the unit (fixes the pseudo-replication).

python scripts/run_synthetic.py --out results/synthetic --replicates 20
"""
import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shiftbench.models import RunConfig  # noqa: E402
from shiftbench.runner import run_splits  # noqa: E402
from shiftbench.synthetic import SYN_SEVERITIES, make_synthetic, synthetic_splits  # noqa: E402
from run_covariate import parse_seeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--geometries", nargs="*", default=["axis", "oblique"])
    ap.add_argument("--shapes", nargs="*", default=["rising", "plateau"])
    ap.add_argument("--models", nargs="*", default=["catboost", "ft_transformer", "lgbm_linear", "ridge"])
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    cfg = RunConfig(n_trials=a.n_trials, n_threads=a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
    for g in a.geometries:
        for sh in a.shapes:
            for r in range(a.replicates):
                ds, z0 = make_synthetic(a.n, a.d, g, sh, a.noise, seed=r)
                run_splits(ds, synthetic_splits(ds, z0, SYN_SEVERITIES), a.models,
                           parse_seeds(a.seeds), cfg, out, save_predictions=False)


if __name__ == "__main__":
    main()
