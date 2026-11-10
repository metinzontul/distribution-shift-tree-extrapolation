"""WP3-REVISION Experiment A: HPO budget-sensitivity ablation (internal review).

DESCRIPTIVE / exploratory ONLY (NOT a mechanism proof, NOT in the primary pre-specified FWER
family, G8). Question: does a much larger HPO budget {8, 50, 100} close the CatBoost-FT extrapolation
gap? Claim limit (MANDATORY): "more budget does not erase the gap" (if it persists) + the caveat
"HPO is selected on ID-val, not OOD; persistence is also consistent with ID->OOD non-transfer."
NEVER "we refuted the budget artifact."

Reuses wp3_pipeline.run_cell VERBATIM (same leakage body: fixed lower-30% train, ID carved from
train, ONE fit-train StandardScaler, train-only HPO val split, test windows DISJOINT from train,
sigma_Y^ref = dataset-fixed train-only lower-30% std, nOOD = OOD-RMSE/sigma_ref, full severity sweep
{0,15,30,50}). The ONLY changes vs the frozen covariate arm:
  * tuner = continuous-space random search (cwr.wp3_hpo_continuous.tune_all_continuous) instead of
    the frozen discrete-8-grid tune_all -- SAME random-search algorithm, only the sampler + #trials
    change (DA-A3: no algorithm x budget confound; frozen metric defs untouched);
  * optional EXTRA untuned RealMLP (holzmuller2024 strong default; reviewer's "fragile NN" target).

Budgets are run SEPARATELY (one --budget per invocation) so each gets its own --out dir, and every
budget (incl. 8) is drawn fresh from the SAME continuous space. Paired across budgets by seed.

Full run (one per budget):
  conda run -n torch python experiments/run_hpo_ablation.py --budget 8   --seeds 3
  conda run -n torch python experiments/run_hpo_ablation.py --budget 50  --seeds 3
  conda run -n torch python experiments/run_hpo_ablation.py --budget 100 --seeds 3
Resume-safe: an existing <out>/<dataset>__extrapolation__seed<k>.npz is skipped.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

from cwr import RESULTS_DIR  # noqa: E402
from cwr import wp3_data as D  # noqa: E402
from cwr import wp3_shift as S  # noqa: E402
from cwr.wp3_hpo_continuous import realmlp_factory, tune_all_continuous  # noqa: E402

import wp3_pipeline as P  # noqa: E402  (same experiments/ dir)

# The 4 PINNED datasets (pre-specified, objective rule).
# sev50 extrapolation gap = median_over_seeds(nOOD_catboost - nOOD_ft_transformer) on frozen results/wp3:
#   diamonds +1.575 (highest) · wave_energy +0.312 (upper-median) · Moneyball +0.010 (null-like) ·
#   video_transcoding -1.982 (most-negative). All reported.
DEFAULT_PINS = ["diamonds", "wave_energy", "Moneyball", "video_transcoding"]


def main():
    ap = argparse.ArgumentParser(description="WP3-revision HPO budget-sensitivity ablation (Plan A).")
    ap.add_argument("--budget", type=int, required=True, help="HPO trial budget (8 / 50 / 100)")
    ap.add_argument("--datasets", default="", help="comma list (default: the 4 pins)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--id-frac", type=float, default=0.25)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--max-fit-train", type=int, default=5000)
    ap.add_argument("--tabpfn-max-train", type=int, default=3000)
    ap.add_argument("--with-realmlp", dest="with_realmlp", action="store_true", default=True)
    ap.add_argument("--no-realmlp", dest="with_realmlp", action="store_false")
    ap.add_argument("--realmlp-variant", default="td_s", choices=["td", "td_s"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="default: results/wp3_revision/hpo_b<budget>")
    ap.add_argument("--smoke-one", action="store_true",
                    help="run ONLY the first dataset + seed0 (timing smoke, no full run)")
    args = ap.parse_args()

    out_dir = args.out or str(RESULTS_DIR / "wp3_revision" / f"hpo_b{args.budget}")
    os.makedirs(out_dir, exist_ok=True)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()] or list(DEFAULT_PINS)
    reg = {n: (n, src, did) for (n, src, did, _) in D.registry()}
    pins = [reg[n] for n in names if n in reg]
    missing = [n for n in names if n not in reg]
    if missing:
        print(f"  WARNING: not in registry -> skipped: {missing}")
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    extra = {f"realmlp_{args.realmlp_variant}": realmlp_factory(args.realmlp_variant)} if args.with_realmlp else None

    if args.smoke_one:
        pins = pins[:1]
        seeds = seeds[:1]

    print(f"HPO-ABLATION budget={args.budget}  datasets={[p[0] for p in pins]}  seeds={seeds}  "
          f"realmlp={args.realmlp_variant if args.with_realmlp else 'OFF'}  device={args.device}")
    print(f"  tuner=CONTINUOUS random-search (space fixed, only #trials varies)  out={out_dir}")
    print("=" * 100)

    # design protocol marker in the out dir (budget + pins + tuner identity)
    with open(os.path.join(out_dir, "_ablation_meta.json"), "w") as f:
        json.dump({"budget": args.budget, "datasets": [p[0] for p in pins], "seeds": seeds,
                   "tuner": "continuous_random_search", "realmlp": args.realmlp_variant if args.with_realmlp else None,
                   "max_fit_train": args.max_fit_train, "max_rows": args.max_rows,
                   "id_frac": args.id_frac, "arm": "extrapolation"}, f, indent=2)

    t_all, per_cell, n_skip = time.time(), [], 0
    for (name, source, did) in pins:
        try:
            ds = D.load_one(name, source, did=did, max_rows=args.max_rows, impute=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} LOAD-FAIL {repr(e)[:80]}"); continue
        inc = D.check_inclusion(ds)
        if not inc["included"]:
            print(f"  {name} EXCLUDED {inc['reasons']}"); continue
        spec = S.build_extrapolation_split(ds.X, ds.y, candidates=inc["continuous_cols"])
        if not spec:
            print(f"  {name} no extrapolation split"); continue
        spec["source"] = source
        for s in seeds:
            tag = f"{name}__extrapolation__seed{s}"
            if os.path.exists(os.path.join(out_dir, f"{tag}.npz")):
                n_skip += 1; continue
            r = P.run_cell(ds, spec, s, out_dir, args.device, args.budget, args.id_frac,
                           args.max_fit_train, args.tabpfn_max_train,
                           tune_fn=tune_all_continuous, extra_model_factories=extra)
            if r.get("skipped"):
                print(f"  [{tag}] SKIP {r['reason']}")
            else:
                per_cell.append(r["meta"]["cell_seconds"])

    total = time.time() - t_all
    print("=" * 100)
    if per_cell:
        print(f"DONE budget={args.budget}: {len(per_cell)} cells ({n_skip} resumed) in {total/60:.1f} min. "
              f"avg {np.mean(per_cell):.1f}s/cell (min {np.min(per_cell):.1f}, max {np.max(per_cell):.1f}).")
    else:
        print(f"no new cells (all {n_skip} resumed).")
    print(f"raw arrays -> {out_dir}")


if __name__ == "__main__":
    main()
