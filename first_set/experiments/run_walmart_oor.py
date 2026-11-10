"""WP3-REVISION Experiment B: natural out-of-range (OOR) + drift on a real temporal shift. (internal review). DESCRIPTIVE / EXPLORATORY ONLY. Claim limits (MANDATORY):
  (a) per-feature OOR fraction = descriptive, low-risk;
  (b) within-test OOR-stratified cat-FT differential = EXPLORATORY/CONFOUNDED (DA-B1 the paper's own
      mechanism is leverage-specific; DA-B2 natural OOR co-occurs with concept/target drift + tail
      sparsity). NEVER "the mechanism is confirmed on natural data."
2x2 interpretation (pre-specified): (OOR+/gap+)=consistent[stratification required];
(OOR+/gap~0)=weak practical importance; (OOR~0/gap+)=claim refined to "not triggered unless shift
pushes a feature OOR"; (OOR~0/gap~0)=consistent with the existing null.

Primary dataset = Walmart Store Sales (CPI/Fuel_Price/Unemployment inflation-driven OOR). Walmart is
NOT keyless on OpenML (searched 2026-10-12); drop the Kaggle files under datasets/walmart/ (see
cwr.wp3_data.load_walmart) and run --dataset walmart. Keyless secondaries runnable NOW:
  --dataset air_quality  (De Vito UCI AirQualityUCI, local; exogenous sensor drift, chronological),
  --dataset beijing_pm25 (if loadable from OpenML; else reported skipped).

Leakage discipline MIRRORS wp3_pipeline.run_temporal_cell VERBATIM: real-calendar time split (earliest
~70% train, latest ~30% test, ID-val carved from the train tail, all strictly before the test period);
per-period seeded subsample with temporal order preserved; train-period-only NaN median imputation;
ONE fit-train StandardScaler; train-only HPO (frozen discrete 8-grid tune_all); sigma_Y^ref = full
train-period Y-std (train-only, dataset-fixed); nOOD = OOD-RMSE/sigma_ref. OOR/drift is computed on the
FULL (un-subsampled) train/test blocks (marginal, seed-independent) and written once to oor_drift.json;
the per-seed npz additionally stores the target-feature OOR membership of the PREDICTED OOD rows so the
OOR-stratified differential can be formed downstream (data-stats-analyst).
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from cwr import RESULTS_DIR  # noqa: E402
from cwr import wp3_data as D  # noqa: E402
from cwr import wp3_shift as S  # noqa: E402
from cwr import wp3_oor as O  # noqa: E402
from cwr.wp3_metrics import nood  # noqa: E402
from cwr.wp3_models import (TABPFN_MAX_FEATURES, TABPFN_REGIME_MAX_N, build_wp3_models,
                            tune_all)  # noqa: E402

MIN_FIT_TRAIN = 200


def regime_of(n):
    return "tabpfn" if n <= TABPFN_REGIME_MAX_N else "large"


def _cap_ordered(idx, cap, rng):
    idx = np.asarray(idx, np.int64)
    if cap and len(idx) > cap:
        idx = rng.choice(idx, int(cap), replace=False)
    return np.sort(idx)


def get_dataset_and_order(name):
    """Return (ds, order_values, feature_names). order_values = chronological key (float)."""
    if name == "walmart":
        ds = D.load_walmart(impute=False)
        return ds, np.asarray(ds.order_values, float), list(ds.X.columns)
    if name == "air_quality":
        ds = D.load_one("air_quality", "local", impute=False)
        return ds, np.arange(ds.n, dtype=float), list(ds.X.columns)   # chronological CSV row order
    if name == "beijing_pm25":
        import openml
        # UCI Beijing PM2.5 mirror on OpenML (did resolved at runtime by name search).
        dl = openml.datasets.list_datasets(output_format="dataframe")
        hit = dl[dl["name"].str.contains("beijing|pm2.?5", case=False, regex=True, na=False)]
        if not len(hit):
            raise FileNotFoundError("Beijing PM2.5 not found on OpenML")
        did = int(hit.sort_values("NumberOfInstances", ascending=False).iloc[0]["did"])
        ds = D.load_openml_did(did, impute=False)
        ds.name = "beijing_pm25"
        return ds, np.arange(ds.n, dtype=float), list(ds.X.columns)
    raise KeyError(name)


def compute_oor_drift(ds, order_values, train_frac, val_frac):
    """Split-level OOR + drift on the FULL (un-subsampled) train/test blocks. Deterministic; written
    once per dataset to oor_drift.json. train = pre-test period (fit+ID-val); test = future period."""
    sp = S.temporal_split(ds, order_values=order_values, train_frac=train_frac, val_frac=val_frac)
    X = ds.X.values.astype(float); y = np.asarray(ds.y, float); names = list(ds.X.columns)
    tr = np.where(sp["train_mask"])[0]          # full train period (fit + ID-val)
    te = np.where(sp["test_mask"])[0]           # full future test period
    Xtr, Xte = X[tr], X[te]
    # train-only NaN median fill for the drift computations (leakage-safe, same rule as the cell)
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    Xtrf = np.where(np.isnan(Xtr), med, Xtr); Xtef = np.where(np.isnan(Xte), med, Xte)

    frac, lo, hi = O.oor_fraction_per_feature(Xtrf, Xtef)
    tname, tidx, trho = O.target_relevant_feature(Xtrf, y[tr], names)
    psi = O.psi_per_feature(Xtrf, Xtef)
    lab = O.label_shift(y[tr], y[te])
    sc = StandardScaler().fit(Xtrf)
    cp = O.concept_probe_ratio(sc.transform(Xtrf), y[tr], sc.transform(Xtef), y[te])
    maha = O.mahalanobis_oor_fraction(Xtrf, Xtef)

    order = np.argsort(-frac)                   # features by descending OOR fraction
    per_feat = [{"feature": names[j], "oor_fraction": float(frac[j]),
                 "train_min": float(lo[j]), "train_max": float(hi[j]), "psi": float(psi[j])}
                for j in order]
    return {
        "dataset": ds.name, "n_full": int(ds.n), "n_features": int(X.shape[1]),
        "n_train_period": int(len(tr)), "n_test_period": int(len(te)),
        "train_frac": train_frac, "val_frac": val_frac,
        "target_relevant_feature": tname, "target_relevant_idx": tidx,
        "target_relevant_abs_spearman": trho,
        "target_feature_oor_fraction": float(frac[tidx]) if tidx is not None else None,
        "oor_fraction_summary": {
            "max": float(np.nanmax(frac)), "mean": float(np.nanmean(frac)),
            "median": float(np.nanmedian(frac)),
            "n_features_oor_gt_0": int(np.nansum(frac > 0)),
            "n_features_oor_gt_0.01": int(np.nansum(frac > 0.01)),
            "argmax_feature": names[int(np.nanargmax(frac))]},
        "psi_summary": {"max": float(np.nanmax(psi)), "mean": float(np.nanmean(psi)),
                        "argmax_feature": names[int(np.nanargmax(psi))],
                        "n_features_psi_gt_0.25": int(np.nansum(psi > 0.25))},
        "label_shift": lab, "concept_probe": cp, "multivariate_oor": maha,
        "per_feature": per_feat,
    }


def run_cell(ds, order_values, seed, out_dir, device, hpo_budget, max_fit_train, tabpfn_max_train,
             train_frac, val_frac, target_idx, oor_idx=None, verbose=True):
    """One leakage-safe temporal cell (mirrors run_temporal_cell) + OOR membership of OOD rows."""
    t_cell = time.time()
    sp = S.temporal_split(ds, order_values=order_values, train_frac=train_frac, val_frac=val_frac)
    if sp is None:
        return {"skipped": True, "reason": "no temporal axis"}
    y = np.asarray(ds.y, float); X = ds.X.values.astype(float)
    train_idx0 = np.where(sp["train_mask"] & ~sp["idval_mask"])[0]
    id_idx0 = np.where(sp["idval_mask"])[0]
    test_idx0 = np.where(sp["test_mask"])[0]
    if len(train_idx0) < MIN_FIT_TRAIN or len(test_idx0) == 0:
        return {"skipped": True, "reason": f"n_fit={len(train_idx0)}<{MIN_FIT_TRAIN} or empty test"}
    rng = np.random.default_rng(1000 + seed)
    train_idx = _cap_ordered(train_idx0, max_fit_train, rng)
    id_idx = _cap_ordered(id_idx0, max_fit_train, rng)
    test_idx = _cap_ordered(test_idx0, max_fit_train, rng)
    assert max(train_idx.max(), id_idx.max()) < test_idx.min(), "future leakage"

    # train-period-only range for the target feature (OOR membership of the predicted OOD rows)
    tf_lo = float(np.nanmin(X[train_idx0, target_idx])); tf_hi = float(np.nanmax(X[train_idx0, target_idx]))
    tf_vals_ood = X[test_idx, target_idx]
    oor_mask = (tf_vals_ood < tf_lo) | (tf_vals_ood > tf_hi)

    # SECONDARY stratification axis = the MAX-OOR feature (e.g. CPI for walmart). Natural range-
    # extrapolation often lands on a low-leverage macro feature, not the target-relevant one (whose
    # OOR can be ~0). Storing this axis lets the EXPLORATORY within-test differential be formed where
    # OOR actually occurs. Train-period-only range (leakage-safe). EXPLORATORY/CONFOUNDED (DA-B1/B2).
    if oor_idx is not None and int(oor_idx) != int(target_idx):
        of_lo = float(np.nanmin(X[train_idx0, oor_idx])); of_hi = float(np.nanmax(X[train_idx0, oor_idx]))
        of_vals_ood = X[test_idx, oor_idx]
        oormax_mask = (of_vals_ood < of_lo) | (of_vals_ood > of_hi)
    else:
        of_lo = of_hi = None; of_vals_ood = None; oormax_mask = None

    if np.isnan(X).any():
        med = np.nanmedian(X[train_idx], axis=0); med = np.where(np.isfinite(med), med, 0.0)
        X = X.copy(); nm = np.isnan(X); X[nm] = np.take(med, np.where(nm)[1])

    nfeat = X.shape[1]
    regime = regime_of(ds.n) if nfeat <= TABPFN_MAX_FEATURES else "large"
    X_fit, y_fit = X[train_idx], y[train_idx]
    scaler = StandardScaler().fit(X_fit)
    Z_fit, Z_id, Z_ood = scaler.transform(X_fit), scaler.transform(X[id_idx]), scaler.transform(X[test_idx])
    sigma_ref = float(np.std(y[train_idx0])) or float(np.std(y))

    tuned, curves = ({}, {})
    if hpo_budget and hpo_budget > 0:
        tuned, curves = tune_all(Z_fit, y_fit, hpo_budget, device=device, random_state=seed)
    models = build_wp3_models(regime, device=device, tabpfn_max_train=tabpfn_max_train,
                              random_state=seed, tuned=tuned, n_features=nfeat)
    preds = {"y_id_true": y[id_idx], "ood_true": y[test_idx],
             "ood_target_feature": tf_vals_ood, "ood_oor_mask": oor_mask.astype(float)}
    if oormax_mask is not None:
        preds["ood_oormax_feature"] = of_vals_ood
        preds["ood_oormax_mask"] = oormax_mask.astype(float)
    nood_row, nood_oor, nood_ir, timings = {}, {}, {}, {}
    nood_oormax, nood_ir_oormax = {}, {}
    y_ood = y[test_idx]
    for name, model in models.items():
        t0 = time.time()
        model.fit(Z_fit, y_fit)
        timings[name] = round(time.time() - t0, 3)
        preds[f"{name}__y_id_pred"] = model.predict(Z_id)
        p = model.predict(Z_ood)
        preds[f"{name}__ood_pred"] = p
        nood_row[name] = nood(y_ood, p, sigma_ref)
        if oor_mask.any():
            nood_oor[name] = nood(y_ood[oor_mask], p[oor_mask], sigma_ref)
        if (~oor_mask).any():
            nood_ir[name] = nood(y_ood[~oor_mask], p[~oor_mask], sigma_ref)
        if oormax_mask is not None and oormax_mask.any():
            nood_oormax[name] = nood(y_ood[oormax_mask], p[oormax_mask], sigma_ref)
        if oormax_mask is not None and (~oormax_mask).any():
            nood_ir_oormax[name] = nood(y_ood[~oormax_mask], p[~oormax_mask], sigma_ref)

    meta = {"dataset": ds.name, "arm": "temporal_oor", "seed": seed, "n": int(ds.n), "d": int(ds.d),
            "n_features": nfeat, "regime": regime, "sigma_y_ref": sigma_ref,
            "target_feature_idx": int(target_idx), "target_feature_train_lo": tf_lo,
            "target_feature_train_hi": tf_hi, "n_ood": int(len(test_idx)),
            "n_ood_oor": int(oor_mask.sum()), "n_ood_in_range": int((~oor_mask).sum()),
            "oor_fraction_predicted": float(oor_mask.mean()),
            "oor_feature_idx": (int(oor_idx) if oor_idx is not None else None),
            "oor_feature_train_lo": of_lo, "oor_feature_train_hi": of_hi,
            "n_ood_oormax": (int(oormax_mask.sum()) if oormax_mask is not None else None),
            "oormax_fraction_predicted": (float(oormax_mask.mean()) if oormax_mask is not None else None),
            "n_fit": int(len(train_idx)), "n_id": int(len(id_idx)),
            "train_frac": train_frac, "val_frac": val_frac, "hpo_budget": int(hpo_budget or 0),
            "models": list(models.keys()), "tuned_configs": tuned, "timings": timings,
            "nood": nood_row, "nood_oor_subset": nood_oor, "nood_in_range_subset": nood_ir,
            "nood_oormax_subset": nood_oormax, "nood_in_range_oormax_subset": nood_ir_oormax,
            "cell_seconds": round(time.time() - t_cell, 3),
            "leakage_checks": "mirrors run_temporal_cell: fit/ID strictly before test; train-only "
                              "scaler/HPO/sigma_ref; order-preserved subsample; target-feature OOR "
                              "range train-period-only"}
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{ds.name}__temporal_oor__seed{seed}"
    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), meta=json.dumps(meta, default=float),
                        **{k: np.asarray(v, float) for k, v in preds.items()})
    if verbose:
        g = nood_row.get("catboost", np.nan) - nood_row.get("ft_transformer", np.nan)
        g_oor = (nood_oor.get("catboost", np.nan) - nood_oor.get("ft_transformer", np.nan)) if nood_oor else np.nan
        g_ir = (nood_ir.get("catboost", np.nan) - nood_ir.get("ft_transformer", np.nan)) if nood_ir else np.nan
        gm = (nood_oormax.get("catboost", np.nan) - nood_oormax.get("ft_transformer", np.nan)) if nood_oormax else np.nan
        gm_ir = (nood_ir_oormax.get("catboost", np.nan) - nood_ir_oormax.get("ft_transformer", np.nan)) if nood_ir_oormax else np.nan
        om = oormax_mask.mean() if oormax_mask is not None else np.nan
        print(f"[{tag}] {meta['cell_seconds']:5.1f}s regime={regime} n_fit={len(train_idx)} "
              f"OODoor(tgt)={oor_mask.mean():.3f} cat-FT gap={g:+.3f} (tgtOOR={g_oor:+.3f} IR={g_ir:+.3f}) "
              f"| maxOORfeat={om:.3f} gap(OOR={gm:+.3f} IR={gm_ir:+.3f})")
    return {"skipped": False, "meta": meta, "tag": tag}


def main():
    ap = argparse.ArgumentParser(description="WP3-revision Experiment B (natural OOR + drift).")
    ap.add_argument("--dataset", default="walmart",
                    choices=["walmart", "air_quality", "beijing_pm25"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--hpo-budget", type=int, default=8)
    ap.add_argument("--max-fit-train", type=int, default=5000)
    ap.add_argument("--tabpfn-max-train", type=int, default=3000)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_dir = args.out or str(RESULTS_DIR / "wp3_revision" / (args.dataset if args.dataset != "walmart" else "walmart"))
    os.makedirs(out_dir, exist_ok=True)
    try:
        ds, order, names = get_dataset_and_order(args.dataset)
    except Exception as e:  # noqa: BLE001
        print(f"DATASET {args.dataset} UNAVAILABLE: {repr(e)[:200]}")
        print("-> report (data pending). No run performed.")
        return
    inc = D.check_inclusion(ds)
    print(f"{args.dataset}: n={ds.n} d={ds.d} included={inc['included']} reasons={inc['reasons']}")

    drift = compute_oor_drift(ds, order, args.train_frac, args.val_frac)
    with open(os.path.join(out_dir, "oor_drift.json"), "w") as f:
        json.dump(drift, f, indent=2, default=float)
    print(f"  target-feature={drift['target_relevant_feature']} "
          f"(|rho|={drift['target_relevant_abs_spearman']:.3f}) "
          f"OOR-frac={drift['target_feature_oor_fraction']:.4f}")
    print(f"  OOR dist: max={drift['oor_fraction_summary']['max']:.3f} "
          f"(@{drift['oor_fraction_summary']['argmax_feature']}) "
          f"#feat OOR>0.01={drift['oor_fraction_summary']['n_features_oor_gt_0.01']}  "
          f"PSI_max={drift['psi_summary']['max']:.3f} (@{drift['psi_summary']['argmax_feature']})  "
          f"label-shift={drift['label_shift']['mean_shift_in_sigma']:+.2f}sigma "
          f"KS={drift['label_shift']['ks_stat']:.3f}  "
          f"concept-ratio={drift['concept_probe']['ridge_rmse_ratio_test_over_train']:.2f}  "
          f"maha-OOR={drift['multivariate_oor']['maha_oor_fraction']:.3f}")
    tidx = drift["target_relevant_idx"]
    if tidx is None:
        print("  no target-relevant feature -> cannot stratify; stop."); return
    # Secondary stratification axis = the feature where natural OOR actually occurs (max-OOR feature,
    # e.g. CPI). None if nothing is OOR. The within-test differential on THIS axis is the exploratory
    # test of "does the gap concentrate where a (low-leverage) feature leaves its training range?".
    oname = drift["oor_fraction_summary"].get("argmax_feature")
    oidx = (names.index(oname) if (oname in names and drift["oor_fraction_summary"]["max"] > 0) else None)
    print(f"  max-OOR axis = {oname} (idx={oidx}, OOR-frac={drift['oor_fraction_summary']['max']:.3f})"
          f" -> secondary within-test stratification [EXPLORATORY]")

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    print("=" * 100)
    for s in seeds:
        tag = f"{ds.name}__temporal_oor__seed{s}"
        if os.path.exists(os.path.join(out_dir, f"{tag}.npz")):
            print(f"  [{tag}] resumed"); continue
        run_cell(ds, order, s, out_dir, args.device, args.hpo_budget, args.max_fit_train,
                 args.tabpfn_max_train, args.train_frac, args.val_frac, tidx, oidx)
    print(f"raw arrays + oor_drift.json -> {out_dir}")


if __name__ == "__main__":
    main()
