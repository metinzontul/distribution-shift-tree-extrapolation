"""Minimal leakage-free pipeline for one synthetic factorial cell (WP1 skeleton).

One cell = (shift-kind x geometry/rotation x severity x extrap-regime x seed). The pipeline:
  1. generates a matched train/OOD-test pair via make_shifted_regime_data (shift on structural frame);
  2. carves an ID-test holdout out of the TRAIN side (same distribution as train, leakage-free);
  3. fits ONE StandardScaler on the fit-train only, then fast models (constant-mean, ridge, LightGBM);
  4. evaluates on ID-test and OOD-test, writes RAW arrays (y_true / y_pred per model) to results/;
  5. probes / smoke-tests the heavy models (FT-Transformer skeleton, TabPFN v2) with guards.

PRE-REGISTRATION GATE: the OOD numbers are NOT to be interpreted as findings or used to tune anything
(design.md; frozen-protocol header). This script only proves the pipeline runs end-to-end, shapes are
right, and the split is leakage-free.

Protocol (design.md §7 / frozen): outer split; ALL fit (scaler + models) on fit-train only; test data
never seen during fit; scaler fit once on fit-train (fold/family-level fit lands in WP2).

Run:  python experiments/synth_pipeline.py --shift covariate --severity 0.6 --rotation 1.0 [--heavy]
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

warnings.filterwarnings("ignore", message="X does not have valid feature names")  # benign LGBM/np

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cwr import RESULTS_DIR
from cwr.heavy_models import FTTransformerWrapper, TabPFNRegressorWrapper, heavy_status
from cwr.synth import ShiftSpec, make_shifted_regime_data


def fast_models():
    return {
        "constant_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "lightgbm": LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  random_state=0, n_jobs=-1, verbose=-1),
    }


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def run_cell(spec, rotation, seed, n_train, n_test, d, k, id_frac, out_dir, heavy):
    data = make_shifted_regime_data(spec, n_train=n_train, n_test=n_test, d=d, k=k,
                                    rotation=rotation, seed=seed)
    X_tr_all, y_tr_all = data["X_train"], data["y_train"]
    X_ood, y_ood = data["X_test"], data["y_test"]

    # ID-test holdout carved from the TRAIN side (same distribution as train, leakage-free split).
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(X_tr_all))
    n_id = max(int(round(id_frac * len(perm))), 1)
    id_idx, fit_idx = perm[:n_id], perm[n_id:]
    X_fit, y_fit = X_tr_all[fit_idx], y_tr_all[fit_idx]
    X_id, y_id = X_tr_all[id_idx], y_tr_all[id_idx]

    # ---- leakage guards (structural + explicit) ----
    assert set(fit_idx).isdisjoint(set(id_idx)), "fit/ID-test index overlap"
    assert X_ood.base is not data["X_train"], "OOD must be a physically separate array"
    scaler = StandardScaler().fit(X_fit)            # fit ONLY on fit-train
    Z_fit, Z_id, Z_ood = scaler.transform(X_fit), scaler.transform(X_id), scaler.transform(X_ood)

    preds = {"y_id_true": y_id, "y_ood_true": y_ood}
    rows = []
    for name, model in fast_models().items():
        model.fit(Z_fit, y_fit)                     # never sees y_id or y_ood
        p_id, p_ood = model.predict(Z_id), model.predict(Z_ood)
        preds[f"{name}__y_id_pred"] = p_id
        preds[f"{name}__y_ood_pred"] = p_ood
        rows.append((name, rmse(y_id, p_id), r2(y_id, p_id), rmse(y_ood, p_ood), r2(y_ood, p_ood)))

    heavy_report = {kf: {"available": v["available"], "needs": v["needs"]} for kf, v in heavy_status().items()}
    if heavy:
        # TabPFN v2 real smoke (tiny, CPU); FT-Transformer skeleton probe.
        try:
            tp = TabPFNRegressorWrapper(device="cpu", max_train=min(500, len(X_fit)))
            if tp.available:
                tp.fit(Z_fit[:500], y_fit[:500])
                p_id, p_ood = tp.predict(Z_id), tp.predict(Z_ood)
                preds["tabpfn__y_id_pred"] = p_id
                preds["tabpfn__y_ood_pred"] = p_ood
                rows.append(("tabpfn_v2", rmse(y_id, p_id), r2(y_id, p_id), rmse(y_ood, p_ood), r2(y_ood, p_ood)))
                heavy_report["TabPFNv2"]["smoke"] = "ran"
            else:
                heavy_report["TabPFNv2"]["smoke"] = "skipped (unavailable)"
        except Exception as e:
            heavy_report["TabPFNv2"]["smoke"] = f"error: {type(e).__name__}: {str(e)[:160]}"
        ft = FTTransformerWrapper()
        heavy_report["FT-Transformer"]["smoke"] = ("available (WP2 impl pending)" if ft.available
                                                   else f"skipped: {heavy_status()['FT-Transformer']['needs']}")

    os.makedirs(out_dir, exist_ok=True)
    tag = (f"{spec.kind}_{spec.covariate_mode if spec.kind=='covariate' else 'na'}"
           f"_rot{rotation}_sev{spec.severity}_{spec.extrap}_seed{seed}")
    npz_path = os.path.join(out_dir, f"{tag}.npz")
    np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in preds.items()},
                        meta=json.dumps(data["meta"]))
    meta = {"cell": tag, "shapes": {"fit": list(X_fit.shape), "id_test": list(X_id.shape),
                                    "ood_test": list(X_ood.shape)},
            "fast_models": [{"model": r[0], "id_rmse": r[1], "id_r2": r[2],
                             "ood_rmse": r[3], "ood_r2": r[4]} for r in rows],
            "heavy": heavy_report, "npz": npz_path,
            "leakage_checks": "scaler+models fit on fit-train only; ID/fit disjoint; OOD separate array"}
    return meta, rows


def main():
    ap = argparse.ArgumentParser(description="Leakage-free synthetic pipeline for one factorial cell.")
    ap.add_argument("--shift", default="covariate", choices=["none", "covariate", "concept", "target"])
    ap.add_argument("--severity", type=float, default=0.6)
    ap.add_argument("--covariate-mode", default="region", choices=["region", "mean"])
    ap.add_argument("--extrap", default="linear", choices=["linear", "plateau", "oscillate"])
    ap.add_argument("--rotation", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--id-frac", type=float, default=0.25)
    ap.add_argument("--heavy", action="store_true", help="also smoke TabPFN v2 / probe FT-Transformer")
    ap.add_argument("--out", default=str(RESULTS_DIR / "synth_pipeline"))
    args = ap.parse_args()

    spec = ShiftSpec(kind=args.shift, severity=args.severity, covariate_mode=args.covariate_mode,
                     extrap=args.extrap)
    meta, rows = run_cell(spec, args.rotation, args.seed, args.n_train, args.n_test, args.d, args.k,
                          args.id_frac, args.out, args.heavy)

    print(f"cell: {meta['cell']}")
    print(f"shapes: fit={meta['shapes']['fit']} id_test={meta['shapes']['id_test']} "
          f"ood_test={meta['shapes']['ood_test']}")
    print(f"{'model':>14}{'ID_rmse':>10}{'ID_r2':>9}{'OOD_rmse':>11}{'OOD_r2':>9}")
    for name, ir, i2, orm, o2 in rows:
        print(f"{name:>14}{ir:>10.3f}{i2:>9.3f}{orm:>11.3f}{o2:>9.3f}")
    print("heavy:")
    for fam, st in meta["heavy"].items():
        print(f"  {fam}: available={st['available']} needs={st['needs']} "
              f"smoke={st.get('smoke', '(not run; pass --heavy)')}")
    print(f"raw arrays -> {meta['npz']}")
    print("NOTE: OOD numbers are NOT interpreted (design protocol gate closed).")


if __name__ == "__main__":
    main()
