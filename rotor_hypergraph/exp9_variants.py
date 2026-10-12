"""Experiment 9: estimator variants against the review's Path-1/Path-2 ideas.

The review proposes four refinements of the saturated symbolic-entropy rule:
  A. normalised-curvature maximum          (lpf_curv)
  B. saturating-exponential fit, tau*=tc   (lpf_fit)
  C. adaptive saturation fraction alpha    (lpf_adaptive)
  D. standard ordinal-pattern symbolisation(pe_sem)
Each is compared with the proposed lpf_sem, the original raw_se and the MI,
ACF and C-C baselines on the same protocol as Experiment 1, but estimate-only
(no Lyapunov exponent), which is the quantity the review asks about (the delay
accuracy and its stability).

Four blocks are computed:
  1a  chaos_accuracy   mean |tau - tau_ref|/tau_ref over the five systems
  1b  chaos_noise      relative delay deviation at 10/5/0 dB
  1c  rotor_tau        per-method delay mean/std over the 180 records, plus
                       a between-class/within-class stability ratio (Path-2
                       suggestion 5: "stable within state, separable between")
  1d  rotor_noise      mean absolute delay deviation at 10/5 dB
  2   downstream       separability of the hypergraph descriptor per delay
                       rule on the measured records (Path-2: does a more
                       stable delay change downstream separability?)

All reference delays are the median of the five baseline estimators on a long
clean series, exactly as in Experiment 1, so the numbers remain comparable.
Results are written to results/exp9_variants.json.
"""
import json
import time

import numpy as np

import chaos
import data
import psr
import stats
import topology
from config import RESULTS, SEED

# New variants plus the baselines.  lpf_sem_d2 (unnormalised second derivative)
# is kept for completeness; the review's proposal is the normalised lpf_curv.
METHODS = ["lpf_sem", "raw_se", "lpf_sem_d2", "lpf_curv", "lpf_fit",
           "lpf_adaptive", "pe_sem", "mi", "acf", "cc"]
BASELINES = ["lpf_sem", "raw_se", "mi", "acf", "cc"]   # for the consensus ref
NCH = 4000          # working length for chaotic series
NREF = 20000        # length used to define the consensus reference
SNRS = [10.0, 5.0, 0.0]
REPS = 3
from exp3_hypergraph import FEATURE_KEYS


def _consensus(x_long):
    """Reference delay: median of the five baseline estimators on a long series."""
    res = psr.estimate_many(x_long, BASELINES, with_dim=False)
    taus = [res[m][0] for m in BASELINES]
    return int(np.median(taus))


def chaos_accuracy():
    out = {}
    for nm in chaos.SYSTEMS:
        x_long, _, _ = chaos.build(nm, n=NREF)
        ref = _consensus(x_long)
        x = x_long[:NCH]
        clean = psr.estimate_many(x, METHODS, with_dim=False)
        rec = {"ref_tau": ref, "methods": {}}
        for m in METHODS:
            tau = clean[m][0]
            rec["methods"][m] = {
                "tau": tau,
                "tau_err": float(abs(tau - ref) / max(1, ref)),
            }
        out[nm] = rec
    return out


def chaos_noise():
    out = {}
    for nm in chaos.SYSTEMS:
        x_long, _, _ = chaos.build(nm, n=NREF)
        x = x_long[:NCH]
        rng = np.random.default_rng(SEED)
        clean = psr.estimate_many(x, METHODS, with_dim=False)
        rec = {"clean": {m: clean[m][0] for m in METHODS}, "snr": {}}
        for snr in SNRS:
            row = {}
            for m in METHODS:
                taus = []
                for _ in range(REPS):
                    xn = data.add_awgn(x, snr, rng)
                    taus.append(psr.estimate_many(xn, [m], with_dim=False)[m][0])
                med = float(np.median(taus))
                row[m] = {"tau": med,
                          "rel_err": float(abs(med - clean[m][0]) /
                                           max(1, clean[m][0])),
                          "tau_std": float(np.std(taus))}
            rec["snr"][str(int(snr))] = row
        out[nm] = rec
    return out


def rotor_tau():
    X, y = data.load_rotor180()
    taus = {m: [] for m in METHODS}
    for i in range(len(X)):
        r = psr.estimate_many(X[i], METHODS, with_dim=False)
        for m in METHODS:
            taus[m].append(r[m][0])
    out = {"per_method": {}, "per_class": {}}
    for m in METHODS:
        ta = np.array(taus[m], float)
        out["per_method"][m] = {
            "tau_mean": float(ta.mean()), "tau_std": float(ta.std()),
            "tau_median": float(np.median(ta)),
        }
        pc = {}
        within = []
        for c in range(4):
            idx = np.where(y == c)[0]
            v = ta[idx]
            pc[str(c)] = {"tau_mean": float(v.mean()),
                          "tau_std": float(v.std())}
            within.append(v.var())
        # Path-2 suggestion 5: between-class / within-class variance ratio.
        means = np.array([pc[str(c)]["tau_mean"] for c in range(4)])
        between = float(means.var())
        within_mean = float(np.mean(within))
        out["per_method"][m]["between_var"] = between
        out["per_method"][m]["within_var"] = within_mean
        out["per_method"][m]["stability_ratio"] = float(
            between / max(within_mean, 1e-12))
        out["per_class"][m] = pc
    return out


def rotor_noise(n_per_class=8, reps=REPS):
    X, y = data.load_rotor180()
    idx = np.concatenate([np.where(y == c)[0][:n_per_class]
                          for c in range(4)])
    rng = np.random.default_rng(SEED)
    clean = {m: [] for m in METHODS}
    for i in idx:
        r = psr.estimate_many(X[i], METHODS, with_dim=False)
        for m in METHODS:
            clean[m].append(r[m][0])
    out = {"clean": {m: {"tau_mean": float(np.mean(clean[m]))}
                     for m in METHODS}, "snr": {}}
    for snr in SNRS:
        row = {}
        for m in METHODS:
            devs = []
            for pos, i in enumerate(idx):
                taus = []
                for _ in range(reps):
                    xn = data.add_awgn(X[i], snr, rng)
                    taus.append(psr.estimate_many(xn, [m], with_dim=False)[m][0])
                devs.append(abs(float(np.median(taus)) - clean[m][pos]))
            row[m] = {"mean_abs_dev": float(np.mean(devs))}
        out["snr"][str(int(snr))] = row
    return out


def downstream(n_per_class=8):
    """Separability of the hypergraph descriptor for each delay rule."""
    X, y = data.load_rotor180()
    idx = np.concatenate([np.where(y == c)[0][:n_per_class]
                          for c in range(4)])
    yy = y[idx]
    out = {"n_per_class": n_per_class, "methods": {}}
    for m in METHODS:
        feats = []
        for i in idx:
            tau, dim, _ = psr.estimate_many(X[i], [m])[m]
            r = topology.signal_to_hypergraph(X[i], tau, dim)
            f = dict(r["features"])
            f["tau"] = tau
            feats.append(f)
        F = stats.to_matrix(feats, FEATURE_KEYS)
        s = stats.summarise(F, yy, FEATURE_KEYS)
        out["methods"][m] = {
            "nc_cv_acc": s["nc_cv_acc"], "fisher_mean": s["fisher_mean"],
            "silhouette": s["silhouette"],
            "overlap_mean": float(np.mean([f["overlap_ratio"]
                                           for f in feats])),
            "tau_mean": float(np.mean([f["tau"] for f in feats])),
            "tau_std": float(np.std([f["tau"] for f in feats])),
        }
        print(f"  downstream {m:12s} acc={out['methods'][m]['nc_cv_acc']:.3f}"
              f" fish={out['methods'][m]['fisher_mean']:.3f}"
              f" tau_std={out['methods'][m]['tau_std']:.2f}", flush=True)
    return out


def main():
    t0 = time.time()
    res = {}
    p = RESULTS / "exp9_variants.json"
    if p.exists():                       # resume partial run
        res = json.loads(p.read_text())
    print("1a chaos accuracy ...", flush=True)
    if "chaos_accuracy" not in res:
        res["chaos_accuracy"] = chaos_accuracy()
        (RESULTS / "exp9_chaos_acc.json").write_text(
            json.dumps(res["chaos_accuracy"], indent=1))
    print("1b chaos noise ...", flush=True)
    if "chaos_noise" not in res:
        res["chaos_noise"] = chaos_noise()
        (RESULTS / "exp9_chaos_noise.json").write_text(
            json.dumps(res["chaos_noise"], indent=1))
    print("1c rotor tau ...", flush=True)
    if "rotor_tau" not in res:
        res["rotor_tau"] = rotor_tau()
    print("1d rotor noise ...", flush=True)
    if "rotor_noise" not in res:
        res["rotor_noise"] = rotor_noise()
    print("2 downstream ...", flush=True)
    if "downstream" not in res:
        res["downstream"] = downstream()
    res["runtime_s"] = time.time() - t0
    p.write_text(json.dumps(res, indent=1))
    print("exp9 done in", round(res["runtime_s"], 1), "s -> results/exp9_variants.json")


if __name__ == "__main__":
    main()
