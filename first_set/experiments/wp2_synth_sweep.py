"""WP2 synthetic core model-sweep runner (shift-robustness design protocol).

Factorial (design.md §3/§5.1, frozen Lock-in 1-8):
  shift-type {covariate(region, primary), concept, target}
    x geometry/rotation {0, 0.5, 1}
    x severity sweep
    x target-extrapolation {linear, plateau, oscillate}  (crossed with covariate; linear-only for
                                                           concept/target, where no OOD region exists)
    x seed
Per cell (one seed realization = one leakage-free dataset):
  1. make_shifted_regime_data(return_worlds) -> matched (train, OOD-test) on the structural frame;
  2. carve an ID-test out of the TRAIN side (same distribution, leakage-free);
  3. fit ONE StandardScaler on fit-train ONLY (per realization; test never seen);
  4. train the Lock-in 1 representatives + baselines: CatBoost, FT-Transformer (GPU),
     TabPFN v2 (GPU), constant-mean, ridge; ORACLE = true noise-free signal (diagnostic);
  5. write RAW arrays (y_true / y_pred per model, ID + OOD) + cell meta + obliqueness to
     results/wp2_synth/<condition>__seed<k>.npz; append a timing row to _timing.jsonl.

PRE-REGISTRATION: the OOD/test gate is OPEN (methodology GO). Hypotheses/metric/thresholds are
FROZEN (frozen-protocol-shift-robustness.md); results do NOT change them (no p-hacking).

Env: conda env `torch` (GPU). Run every python call via `conda run -n torch python ...`.
  Pilot:  python experiments/wp2_synth_sweep.py --mode pilot
  Full :  python experiments/wp2_synth_sweep.py --mode full --seeds 30
Portable: --device cpu falls back cleanly; --out redirects results (lab-machine offload).
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
from sklearn.preprocessing import StandardScaler

from cwr import RESULTS_DIR
from cwr.synth import ShiftSpec, make_shifted_regime_data
from cwr.wp2_metrics import norm_rmse, obliqueness_score
from cwr.wp2_models import (BASELINE_MODELS, FAMILY_MODELS, CatBoostGBDT, ConstantMean,
                            FTTransformerNN, RidgeLinear, TabPFNv2, tune_catboost, tune_ft)

# --------------------------------------------------------------------------------------------------
# Factorial condition builders
# --------------------------------------------------------------------------------------------------

ROTATIONS = (0.0, 0.5, 1.0)
FULL_SEVERITIES = (0.25, 0.5, 0.75, 1.0)
PILOT_SEVERITIES = (0.5, 1.0)


def _cond(shift, rotation, severity, extrap):
    return {"shift": shift, "rotation": rotation, "severity": severity, "extrap": extrap}


def full_conditions(severities=FULL_SEVERITIES, rotations=ROTATIONS):
    """Full factorial. Extrap crosses covariate only (Kanıt C lives there); concept/target use linear
    (no out-of-support region -> extrap is inert, wp1-shift-generator note)."""
    conds = []
    for rot in rotations:
        for sev in severities:
            for ex in ("linear", "plateau", "oscillate"):
                conds.append(_cond("covariate", rot, sev, ex))
            conds.append(_cond("concept", rot, sev, "linear"))
            conds.append(_cond("target", rot, sev, "linear"))
    return conds


def pilot_conditions():
    """Representative slice (task WP2 §3): H1 covariate x geometry, H2 concept x geometry, Kanıt C
    linear&plateau target-regime; plus a target-shift sanity pair."""
    conds = []
    for rot in ROTATIONS:                      # H1: covariate x geometry (linear)
        for sev in PILOT_SEVERITIES:
            conds.append(_cond("covariate", rot, sev, "linear"))
    for rot in ROTATIONS:                      # Kanıt C reverse control: covariate x plateau
        for sev in PILOT_SEVERITIES:
            conds.append(_cond("covariate", rot, sev, "plateau"))
    for rot in ROTATIONS:                      # H2: concept x geometry
        for sev in PILOT_SEVERITIES:
            conds.append(_cond("concept", rot, sev, "linear"))
    for rot in (0.0, 1.0):                     # target-shift sanity
        conds.append(_cond("target", rot, 1.0, "linear"))
    return conds


def cond_tag(c):
    return (f"{c['shift']}_rot{c['rotation']}_sev{c['severity']}_{c['extrap']}"
            .replace(".", "p"))


# --------------------------------------------------------------------------------------------------
# One leakage-free cell
# --------------------------------------------------------------------------------------------------


def run_cell(cond, seed, n_train, n_test, d, k, id_frac, device, hpo_budget, out_dir,
             tabpfn_max_train, verbose=True):
    t_cell = time.time()
    spec = ShiftSpec(kind=cond["shift"], severity=cond["severity"],
                     covariate_mode="region", extrap=cond["extrap"])
    data = make_shifted_regime_data(spec, n_train=n_train, n_test=n_test, d=d, k=k,
                                    rotation=cond["rotation"], seed=seed, return_worlds=True)
    X_tr_all, y_tr_all = data["X_train"], data["y_train"]
    X_ood, y_ood = data["X_test"], data["y_test"]
    reg_tr, reg_ood = data["regime_train"], data["regime_test"]
    w_tr, w_te = data["world_train"], data["world_test"]
    tgt, _ = data["_target_signal"], data["_regime_of"]

    # --- ID-test carved from the TRAIN side (same distribution, leakage-free) ---
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(X_tr_all))
    n_id = max(int(round(id_frac * len(perm))), 1)
    id_idx, fit_idx = perm[:n_id], perm[n_id:]
    X_fit, y_fit = X_tr_all[fit_idx], y_tr_all[fit_idx]
    X_id, y_id = X_tr_all[id_idx], y_tr_all[id_idx]

    # --- leakage guards ---
    assert set(fit_idx.tolist()).isdisjoint(set(id_idx.tolist())), "fit/ID index overlap"
    assert not np.shares_memory(X_ood, X_tr_all), "OOD must be a physically separate array"

    # --- oracle: true noise-free signal (diagnostic upper bound, Lock-in 10) ---
    R = w_tr.R
    S_tr_all = X_tr_all @ R.T
    sig_tr_all = tgt(S_tr_all, w_tr, reg_tr)             # train-world signal at train points
    S_ood = X_ood @ R.T
    sig_ood = tgt(S_ood, w_te, reg_ood)                 # test-world signal at OOD points
    oracle_id = sig_tr_all[id_idx]
    oracle_ood = sig_ood

    # --- per-realization scaler, fit on fit-train ONLY (design.md §7) ---
    scaler = StandardScaler().fit(X_fit)
    Z_fit, Z_id, Z_ood = scaler.transform(X_fit), scaler.transform(X_id), scaler.transform(X_ood)
    y_fit_std = float(np.std(y_fit))

    # --- optional HPO (equal #config across the two tunable families; TabPFN excluded) ---
    cat_params = None
    ft_kwargs = {}
    if hpo_budget and hpo_budget > 0:
        cat_params = tune_catboost(Z_fit, y_fit, hpo_budget, random_state=seed)
        ft_kwargs = tune_ft(Z_fit, y_fit, hpo_budget, device=device, random_state=seed)

    models = {
        "constant_mean": ConstantMean(),
        "ridge": RidgeLinear(alpha=1.0),
        "catboost": CatBoostGBDT(random_state=seed, params=cat_params),
        "ft_transformer": FTTransformerNN(device=device, random_state=seed, **ft_kwargs),
        "tabpfn": TabPFNv2(device=device, max_train=tabpfn_max_train, random_state=seed),
    }

    preds = {"y_id_true": y_id, "y_ood_true": y_ood,
             "oracle__y_id_pred": oracle_id, "oracle__y_ood_pred": oracle_ood}
    timings = {}
    rows = []
    for name, model in models.items():
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Z_fit, y_fit)
            p_id, p_ood = model.predict(Z_id), model.predict(Z_ood)
        timings[name] = round(time.time() - t0, 3)
        preds[f"{name}__y_id_pred"] = p_id
        preds[f"{name}__y_ood_pred"] = p_ood
        rows.append((name, norm_rmse(y_id, p_id, y_fit_std), norm_rmse(y_ood, p_ood, y_fit_std)))
    rows.append(("oracle", norm_rmse(y_id, oracle_id, y_fit_std),
                 norm_rmse(y_ood, oracle_ood, y_fit_std)))

    obliq = obliqueness_score(X_fit, y_fit)

    meta = {**cond, "seed": seed, "n_train": n_train, "n_test": n_test, "d": d, "k": k,
            "id_frac": id_frac, "n_fit": len(fit_idx), "n_id": len(id_idx), "n_ood": len(X_ood),
            "y_fit_std": y_fit_std, "obliqueness": float(obliq),
            "hpo_budget": int(hpo_budget or 0), "device": device,
            "cat_params": cat_params, "ft_kwargs": ft_kwargs,
            "shift_severity_effective": data["meta"].get("severity"),
            "noise_scale": data["meta"].get("noise_scale"),
            "timings": timings, "cell_seconds": round(time.time() - t_cell, 3),
            "leakage_checks": "scaler+models fit on fit-train only; ID/fit disjoint; OOD separate array"}

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cond_tag(cond)}__seed{seed}"
    npz_path = os.path.join(out_dir, f"{tag}.npz")
    np.savez_compressed(npz_path, meta=json.dumps(meta),
                        **{kk: np.asarray(vv, float) for kk, vv in preds.items()})

    with open(os.path.join(out_dir, "_timing.jsonl"), "a") as f:
        f.write(json.dumps({"tag": tag, **{k: meta[k] for k in ("shift", "rotation", "severity",
                "extrap", "seed", "cell_seconds", "obliqueness")}, "timings": timings}) + "\n")

    if verbose:
        rr = "  ".join(f"{n}:{oo:.3f}" for n, _, oo in rows)
        print(f"[{tag}] {meta['cell_seconds']:5.1f}s  nOOD-RMSE  {rr}")
    return meta, rows, npz_path


# --------------------------------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="WP2 synthetic model-sweep (leakage-free, pre-specified).")
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds per condition")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (seeds = seed0..seed0+seeds-1)")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--id-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hpo-budget", type=int, default=0, help="0 = authoritative defaults (design §4.4)")
    ap.add_argument("--tabpfn-max-train", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="cap number of cells (smoke); 0 = no cap")
    ap.add_argument("--out", default=str(RESULTS_DIR / "wp2_synth"))
    args = ap.parse_args()

    conds = pilot_conditions() if args.mode == "pilot" else full_conditions()
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    cells = [(c, s) for c in conds for s in seeds]
    if args.limit:
        cells = cells[:args.limit]

    print(f"mode={args.mode}  conditions={len(conds)}  seeds={len(seeds)}  cells={len(cells)}  "
          f"device={args.device}  hpo_budget={args.hpo_budget}  out={args.out}")
    print(f"per-cell: n_train={args.n_train} n_test={args.n_test} d={args.d} k={args.k} "
          f"id_frac={args.id_frac}")
    print("=" * 100)

    t_all = time.time()
    per_cell = []
    for i, (c, s) in enumerate(cells):
        meta, _, _ = run_cell(c, s, args.n_train, args.n_test, args.d, args.k, args.id_frac,
                              args.device, args.hpo_budget, args.out, args.tabpfn_max_train)
        per_cell.append(meta["cell_seconds"])
        if (i + 1) % 10 == 0 or i + 1 == len(cells):
            done = i + 1
            avg = float(np.mean(per_cell))
            eta = avg * (len(cells) - done) / 60.0
            print(f"  ... {done}/{len(cells)} cells, avg {avg:.1f}s/cell, ETA {eta:.1f} min")

    total = time.time() - t_all
    avg = float(np.mean(per_cell)) if per_cell else float("nan")
    print("=" * 100)
    print(f"DONE {len(cells)} cells in {total/60:.1f} min. avg {avg:.2f}s/cell "
          f"(min {np.min(per_cell):.1f}, max {np.max(per_cell):.1f}).")
    print(f"raw arrays -> {args.out}  (timing log: {os.path.join(args.out, '_timing.jsonl')})")
    print("NOTE: hypotheses/metric/thresholds are FROZEN; results do not change them.")


if __name__ == "__main__":
    main()
