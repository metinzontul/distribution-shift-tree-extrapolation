#!/usr/bin/env python
"""Build the statistics report from one or more result folders.

python scripts/analyze.py results/covariate results/temporal --out report.md \
    --pairs catboost:ft_transformer catboost:ridge lgbm_linear:ft_transformer catboost:tabpfn
"""
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shiftbench.analysis import full_report, load_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--pairs", nargs="*", default=["catboost:ft_transformer", "catboost:ridge",
                                                   "lgbm_linear:ft_transformer"])
    ap.add_argument("--max-severity", type=float, default=50.0)
    ap.add_argument("--margin", type=float, default=0.05)
    a = ap.parse_args()
    df = pd.concat([load_results(Path(d) / "results.csv") for d in a.dirs], ignore_index=True)
    diags = [Path(d) / "diagnostics.csv" for d in a.dirs if (Path(d) / "diagnostics.csv").exists()]
    diag = pd.concat([pd.read_csv(p) for p in diags], ignore_index=True) if diags else None
    pairs = [tuple(p.split(":")) for p in a.pairs]
    pairs = [p for p in pairs if set(p) <= set(df.model)]
    text = full_report(df, diag, pairs, a.max_severity, a.margin)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
