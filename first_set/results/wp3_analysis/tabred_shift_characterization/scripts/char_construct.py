"""Recompute the construct (covariate extrapolation-arm) OOR/PSI calibration over the LEAKAGE-SAFE
N=27 roster, with the CURRENT leakage-safe code (impute=False so partially-missing features are
ineligible as a split axis, matching _abs_spearman + the wp3_pipeline covariate arm).

Roster = extrapolation-arm datasets actually run (results/wp3/*__extrapolation__*.npz)
         minus the 4 TabReD sets minus COVARIATE_ARM_EXCLUDE {air_quality, housing}  -> 27.

Validation: reconstructed split feature + |spearman| are cross-checked against each dataset's
seed0 npz meta (the experiment ground truth).
"""
import os, sys, json, time
import numpy as np
from scipy.stats import spearmanr

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[4])
sys.path.insert(0, os.path.join(ROOT, "code"))
from cwr import wp3_data as D
from cwr import wp3_shift as S

WP3 = os.path.join(ROOT, "results", "wp3")
OUT = os.path.join(ROOT, "results", "wp3_analysis", "tabred_shift_characterization")
TABRED = {"cooking-time", "delivery-eta", "sberbank-housing", "weather"}

run = {f.split("__")[0] for f in os.listdir(WP3) if f.endswith(".npz") and "__extrapolation__" in f}
ROSTER = sorted(run - TABRED - set(D.COVARIATE_ARM_EXCLUDE))
assert len(ROSTER) == 27, f"expected 27, got {len(ROSTER)}: {ROSTER}"

# registry lookup: name -> (source, did)
REG = {name: (source, did) for (name, source, did, _n) in D.registry()}


def npz_meta(name):
    import glob
    fs = sorted(glob.glob(os.path.join(WP3, f"{name}__extrapolation__seed0.npz")))
    if not fs:
        return None
    z = np.load(fs[0], allow_pickle=True)
    return json.loads(str(z["meta"]))


def psi(train, test, bins=10):
    qs = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return np.nan
    qs = qs.astype(np.float64); qs[0], qs[-1] = -np.inf, np.inf
    tr_c, _ = np.histogram(train, qs); te_c, _ = np.histogram(test, qs)
    tr_p = np.clip(tr_c / max(tr_c.sum(), 1), 1e-6, None)
    te_p = np.clip(te_c / max(te_c.sum(), 1), 1e-6, None)
    return float(np.sum((te_p - tr_p) * np.log(te_p / tr_p)))


def oor_metrics(Xtr, Xte):
    lo_mm, hi_mm = Xtr.min(0), Xtr.max(0)
    lo_pp, hi_pp = np.percentile(Xtr, 1, 0), np.percentile(Xtr, 99, 0)
    def rates(lo, hi):
        oor = (Xte < lo[None, :]) | (Xte > hi[None, :])
        return oor.mean(0), float(oor.any(1).mean()), float(oor.sum(1).mean())
    pf_mm, any_mm, cnt_mm = rates(lo_mm, hi_mm)
    pf_pp, any_pp, cnt_pp = rates(lo_pp, hi_pp)
    return pf_mm, any_mm, cnt_mm, pf_pp, any_pp, cnt_pp


def one(name, source, did):
    ds = D.load_one(name, source, did=did, max_rows=20000, seed=42, impute=False)  # LEAKAGE-SAFE
    inc = D.check_inclusion(ds)
    if not inc["included"]:
        return {"set": name, "included": False, "reasons": inc["reasons"]}
    cont = inc["continuous_cols"]
    ex = S.build_extrapolation_split(ds.X, ds.y, candidates=cont)
    if ex is None:
        return {"set": name, "included": False, "reasons": ["no split"]}
    feat = ex["feature"]
    train_mask = ex["windows"][0]["train_mask"]
    w50 = next((w for w in ex["windows"] if w["severity"] == 50), None)
    if w50 is None:
        return {"set": name, "included": False, "reasons": ["no sev50"]}
    test_mask = w50["test_mask"] & ~train_mask
    if test_mask.sum() < 10 or train_mask.sum() < 10:
        return {"set": name, "included": False, "reasons": ["empty sev50/train"]}

    X = np.array(ds.X[cont].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    X.setflags(write=True)
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X[train_mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    m = np.isnan(X)
    if m.any():
        X[m] = np.take(med, np.where(m)[1])
    ptp = np.ptp(X[train_mask], axis=0); std_tr = X[train_mask].std(0)
    keep = (ptp > 1e-12) & (std_tr > 1e-12)
    Xk = X[:, keep]; cont_k = [c for c, k in zip(cont, keep) if k]
    Xtr, Xte = Xk[train_mask], Xk[test_mask]
    F = Xk.shape[1]
    feat_j = cont_k.index(feat) if feat in cont_k else None

    pf_mm, any_mm, cnt_mm, pf_pp, any_pp, cnt_pp = oor_metrics(Xtr, Xte)
    std_trk = Xtr.std(0); std_trk = np.where(std_trk > 1e-12, std_trk, 1.0)
    smd = np.abs(Xte.mean(0) - Xtr.mean(0)) / std_trk
    psis = np.array([psi(Xtr[:, j], Xte[:, j]) for j in range(F)])
    psis_f = psis[np.isfinite(psis)]

    return {
        "set": name, "source": source, "included": True, "n": int(ds.n), "F_used": int(F),
        "split_feature": str(feat), "abs_spearman": float(ex["spearman"]),
        "n_train": int(train_mask.sum()), "n_test_sev50": int(test_mask.sum()),
        "oor_minmax": {"mean_rate": float(pf_mm.mean()), "max_feat_rate": float(pf_mm.max()),
                       "frac_rows_any": any_mm, "mean_nfeat_oor": cnt_mm,
                       "frac_feats_over50": float((pf_mm > 0.5).mean())},
        "oor_p1p99": {"mean_rate": float(pf_pp.mean()), "max_feat_rate": float(pf_pp.max()),
                      "frac_rows_any": any_pp, "mean_nfeat_oor": cnt_pp},
        "split_feature_oor": {"oor_mm": float(pf_mm[feat_j]) if feat_j is not None else None,
                              "oor_pp": float(pf_pp[feat_j]) if feat_j is not None else None},
        "covariate_shift": {"smd_median": float(np.median(smd)), "smd_max": float(smd.max()),
                            "psi_median": float(np.median(psis_f)) if len(psis_f) else None,
                            "psi_max": float(psis_f.max()) if len(psis_f) else None,
                            "frac_psi_over025": float((psis_f > 0.25).mean()) if len(psis_f) else None},
    }


def main():
    t0 = time.time()
    out, valfail = {}, []
    for name in ROSTER:
        source, did = REG[name]
        try:
            r = one(name, source, did)
        except Exception as e:
            r = {"set": name, "included": False, "reasons": [f"ERR {repr(e)[:120]}"]}
        out[name] = r
        if r.get("included"):
            meta = npz_meta(name)
            fok = (meta is not None and meta["feature"] == r["split_feature"])
            sok = (meta is not None and abs(abs(meta["spearman"]) - r["abs_spearman"]) < 1e-3)
            tag = "OK" if (fok and sok) else "MISMATCH"
            if not (fok and sok):
                valfail.append((name, r["split_feature"], None if meta is None else meta["feature"],
                                r["abs_spearman"], None if meta is None else abs(meta["spearman"])))
            sf = r["split_feature_oor"]
            print(f"[{name:30s}] feat={r['split_feature']:24s} rho={r['abs_spearman']:.3f} "
                  f"splitOOR={sf['oor_mm']:.3f} maxF={r['oor_minmax']['max_feat_rate']:.3f} "
                  f"meanMM={r['oor_minmax']['mean_rate']:.4f} meanPP={r['oor_p1p99']['mean_rate']:.4f} "
                  f"psiMed={r['covariate_shift']['psi_median']:.3f} [{tag}]")
        else:
            print(f"[{name:30s}] EXCLUDED {r.get('reasons')}")

    inc = {k: v for k, v in out.items() if v.get("included")}
    smm = np.array([v["split_feature_oor"]["oor_mm"] for v in inc.values()])
    mxf = np.array([v["oor_minmax"]["max_feat_rate"] for v in inc.values()])
    mmm = np.array([v["oor_minmax"]["mean_rate"] for v in inc.values()])
    mpp = np.array([v["oor_p1p99"]["mean_rate"] for v in inc.values()])
    psi_med = np.array([v["covariate_shift"]["psi_median"] for v in inc.values()
                        if v["covariate_shift"]["psi_median"] is not None])
    n27 = int((smm >= 0.999).sum())
    print(f"\n=== CONSTRUCT sev50 over {len(inc)} datasets (LEAKAGE-SAFE, impute=False) ===")
    print(f"  split-feature OOR[min,max]: median={np.median(smm):.3f} mean={smm.mean():.3f} "
          f"min={smm.min():.3f} | #(==1.0)={n27}/{len(inc)}")
    print(f"  max-feature  OOR[min,max]:  median={np.median(mxf):.3f} mean={mxf.mean():.3f}")
    print(f"  mean-per-feature OOR[mm]:    median={np.median(mmm):.4f} mean={mmm.mean():.4f}")
    print(f"  mean-per-feature OOR[p1,p99]:median={np.median(mpp):.4f} mean={mpp.mean():.4f}")
    print(f"  median PSI:                  median={np.median(psi_med):.4f} mean={psi_med.mean():.4f}")
    if valfail:
        print("\n!!! VALIDATION MISMATCHES vs npz meta (name, recomputed_feat, npz_feat, rho, npz_rho):")
        for row in valfail:
            print("   ", row)
    else:
        print("\nVALIDATION: all 27 reconstructed splits match seed0 npz meta (feature + |spearman|).")

    with open(os.path.join(OUT, "construct_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWROTE", os.path.join(OUT, "construct_metrics.json"), f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
