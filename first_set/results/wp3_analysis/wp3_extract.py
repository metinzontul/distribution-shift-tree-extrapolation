"""WP3 analysis stage 1 -- EXTRACT: raw npz (results/wp3/) -> long-format CSVs + cell-meta CSV.

Reconstructed + persisted 2026-08-26 (the original extract lived only in a scratchpad; see
the internal analysis notes). Byte-for-byte revalidated against the frozen
_long_covariate.csv / _long_temporal.csv / _cell_meta.csv before use.

nOOD is recomputed from the RAW arrays: nOOD = RMSE(y_true, y_pred) / sigma_Y^ref (sigma_Y^ref is the
dataset-fixed, train-only reference scale stored in each cell's meta). No epsilon (matches frozen).

Iteration order = sorted(glob('*.npz')); per cell, models in meta order, inner severity [0,15,30,50]
-- this reproduces the frozen row order so a diff isolates only the changed (Moneyball) rows.

Usage:  conda run -n torch python wp3_extract.py [--wp3 <dir>] [--out <dir>]
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

SEVS = [0, 15, 30, 50]


def _rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def extract(wp3_dir):
    cov_rows, tmp_rows, meta_rows = [], [], []
    for f in sorted(glob.glob(os.path.join(wp3_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "meta" not in z.files:
            continue
        m = json.loads(str(z["meta"]))
        arm = m["arm"]
        ds, seed = m["dataset"], m["seed"]
        sig = float(m["sigma_y_ref"])
        models = m["models"]
        has_tabpfn = "tabpfn" in models
        if arm == "temporal":
            ood_true = z["ood_true"]; id_true = z["y_id_true"]
            for mm in models:
                nood = _rmse(ood_true, z[f"{mm}__ood_pred"]) / sig
                nood_id = _rmse(id_true, z[f"{mm}__y_id_pred"]) / sig
                tmp_rows.append([ds, seed, mm, nood, nood_id, sig,
                                 m.get("temporal_source"), m.get("tabpfn_feasible"), m.get("n_ood")])
            meta_rows.append([ds, arm, seed, m["n"], m["d"], m["regime"], m.get("feature"),
                              m.get("spearman"), m.get("slope_ratio"), m.get("stratum"), sig,
                              m.get("source"), m.get("temporal_source"), m.get("tabpfn_feasible"),
                              m.get("n_fit"), m.get("n_id"), m.get("n_ood"), has_tabpfn])
        else:
            id_true = z["y_id_true"]
            slope = m.get("slope_ratio"); regime = m["regime"]
            for mm in models:
                nood_id = _rmse(id_true, z[f"{mm}__y_id_pred"]) / sig
                for S in SEVS:
                    tk = f"ood_sev{S}_true"
                    pk = f"{mm}__ood_sev{S}_pred"
                    if tk not in z.files or pk not in z.files:
                        continue                       # empty window -> no row (matches frozen)
                    nood = _rmse(z[tk], z[pk]) / sig
                    cov_rows.append([ds, arm, seed, S, mm, nood, nood_id, sig, slope, regime])
            meta_rows.append([ds, arm, seed, m["n"], m["d"], regime, m.get("feature"),
                              m.get("spearman"), slope, m.get("stratum"), sig,
                              m.get("source"), m.get("temporal_source"), m.get("tabpfn_feasible"),
                              m.get("n_fit"), m.get("n_id"), m.get("n_ood"), has_tabpfn])
    cov = pd.DataFrame(cov_rows, columns=["dataset", "arm", "seed", "severity", "model", "nood",
                                          "nood_id", "sigma_y_ref", "slope_ratio", "regime"])
    tmp = pd.DataFrame(tmp_rows, columns=["dataset", "seed", "model", "nood", "nood_id",
                                          "sigma_y_ref", "temporal_source", "tabpfn_feasible", "n_ood"])
    meta = pd.DataFrame(meta_rows, columns=["dataset", "arm", "seed", "n", "d", "regime", "feature",
                                            "spearman", "slope_ratio", "stratum_provisional",
                                            "sigma_y_ref", "source", "temporal_source",
                                            "tabpfn_feasible", "n_fit", "n_id", "n_ood", "has_tabpfn"])
    return cov, tmp, meta


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--wp3", default=os.path.join(here, "..", "wp3"))
    ap.add_argument("--out", default=here)
    args = ap.parse_args()
    cov, tmp, meta = extract(args.wp3)
    cov.to_csv(os.path.join(args.out, "_long_covariate.csv"), index=False)
    tmp.to_csv(os.path.join(args.out, "_long_temporal.csv"), index=False)
    meta.to_csv(os.path.join(args.out, "_cell_meta.csv"), index=False)
    print(f"extract: covariate rows={len(cov)}  temporal rows={len(tmp)}  cells={len(meta)}")
    print(f"  datasets(cov)={cov.dataset.nunique()}  temporal-sets={tmp.dataset.nunique()}")


if __name__ == "__main__":
    main()
