#!/usr/bin/env python
"""Fast end-to-end check on small generated data (no internet needed).
Runs every split kind, every default model, the temporal arm with TOST, the
synthetic generator, and the report. Takes a few minutes on CPU."""
import shutil
import warnings
warnings.filterwarnings("ignore")
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shiftbench.analysis import full_report, load_results  # noqa: E402
from shiftbench.data import from_frame  # noqa: E402
from shiftbench.models import RunConfig  # noqa: E402
from shiftbench.runner import run_splits, temporal_split  # noqa: E402
from shiftbench.splits import ALL_KINDS, build_splits  # noqa: E402
from shiftbench.synthetic import make_synthetic, synthetic_splits  # noqa: E402


def toy_frame(seed, n=1200, with_cat=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 2 * X[:, 0] + np.sin(2 * X[:, 1]) + 0.5 * np.tanh(3 * X[:, 2]) + rng.normal(0, .3, n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    if with_cat:
        df["city"] = rng.choice(["a", "b", "c"], n)
        y = y + df["city"].map({"a": 0, "b": .5, "c": -.5}).to_numpy()
        df.loc[rng.random(n) < .02, "f3"] = np.nan
    df["y"] = y
    return df


def main():
    out = Path("/tmp/shiftbench_smoke")
    shutil.rmtree(out, ignore_errors=True)
    cfg = RunConfig(n_trials=2, n_threads=2, ft_max_epochs=6, ft_min_epochs=2, ft_patience=2,
                    mlp_max_iter=50, gbdt_trees={"catboost": 100, "xgboost": 100,
                                                 "lightgbm": 100, "lgbm_linear": 100})
    models = ["catboost", "xgboost", "lightgbm", "lgbm_linear", "ft_transformer", "mlp", "ridge", "constant"]
    seeds = [0, 1]

    # covariate arm: 4 toy datasets, every split kind
    for i in range(4):
        ds = from_frame(f"toy{i}", toy_frame(i, with_cat=i % 2 == 0), "y")
        splits = build_splits(ds, kinds=ALL_KINDS, severities=(0, 50))
        run_splits(ds, splits, models, seeds, cfg, out / "cov", verbose=False)
    # holdout-selection variant on one dataset
    ds = from_frame("toy_holdout", toy_frame(9), "y")
    run_splits(ds, build_splits(ds, kinds=["extrap"], severities=(0, 50), selection="holdout"),
               ["catboost", "ridge"], seeds, cfg, out / "cov_holdout", verbose=False)

    # temporal arm: split column
    for i in range(3):
        df = toy_frame(10 + i)
        df["split"] = np.where(np.arange(len(df)) < 900, "train", "test")
        ds = from_frame(f"temp{i}", df, "y", drop=["split"])
        ds.train_mask, ds.test_mask = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
        run_splits(ds, [temporal_split(ds)], models, [0, 1, 2], cfg, out / "temporal", verbose=False)

    # synthetic
    for shape in ("rising", "plateau"):
        for r in range(2):
            ds, z0 = make_synthetic(n=1500, d=6, shape=shape, seed=r)
            run_splits(ds, synthetic_splits(ds, z0, (0.0, 1.0)), ["catboost", "ft_transformer"],
                       [0], cfg, out / "syn", save_predictions=False, verbose=False)

    # resumability: rerun must add nothing
    n_before = len(pd.read_csv(out / "cov" / "results.csv"))
    ds = from_frame("toy0", toy_frame(0), "y")
    run_splits(ds, build_splits(ds, kinds=ALL_KINDS, severities=(0, 50)), models, seeds, cfg,
               out / "cov", verbose=False)
    assert len(pd.read_csv(out / "cov" / "results.csv")) == n_before, "resume failed"

    res = pd.concat([pd.read_csv(out / d / "results.csv") for d in ["cov", "temporal", "syn"]])
    print(res["status"].value_counts().to_string())
    assert (res["status"] == "ok").all(), res[res.status != "ok"][["model", "note"]]

    df = pd.concat([load_results(out / d / "results.csv") for d in ["cov", "temporal", "syn"]])
    diag = pd.read_csv(out / "cov" / "diagnostics.csv")
    rep = full_report(df, diag, [("catboost", "ft_transformer"), ("catboost", "ridge"),
                                 ("lgbm_linear", "ft_transformer")])
    (out / "report.md").write_text(rep, encoding="utf-8")
    print(rep[:3000])
    print("\nSMOKE TEST PASSED ->", out)


if __name__ == "__main__":
    main()
