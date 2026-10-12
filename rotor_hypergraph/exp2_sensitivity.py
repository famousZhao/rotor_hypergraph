"""Experiment 2: parameter sensitivity.

Filter parameters (Butterworth order, cutoff, saturation fraction alpha) act
only on the symbolic-entropy curve, so they are swept on the delay estimate.
Community-construction parameters (edge-removal ratio, recurrence percentile)
are swept on the downstream topology and scored by separability.
"""
import json
import time
import numpy as np

import data
import psr
import stats
import topology
from exp3_hypergraph import FEATURE_KEYS, _subset
from config import E_FACTOR, RESULTS
from psr import lpf_sem_from_curve, symbolic_entropy_curve
from config import SYMBOLS_P, TAU_MAX


def filter_sweep(n_per_class=5):
    """Delay estimate vs filter order, cutoff and saturation fraction."""
    X, y = data.load_rotor180()
    idx = _subset(y, n_per_class)
    taus = np.arange(1, TAU_MAX + 1)
    curves = [symbolic_entropy_curve(X[i], taus) for i in idx]

    out = {"orders": {}, "cutoffs": {}, "alphas": {}}
    from scipy import signal as sg

    for order in [1, 2, 3, 4, 5]:
        vals = []
        for H in curves:
            b, a = sg.butter(order, 0.1, "low")
            Hf = sg.filtfilt(b, a, H)
            d1 = np.gradient(Hf)
            ip = int(np.argmax(d1))
            below = np.where(d1[ip:] <= 0.1 * d1[ip])[0]
            j = ip + int(below[0]) if len(below) else len(taus) - 1
            vals.append(taus[j])
        out["orders"][str(order)] = {"tau_mean": float(np.mean(vals)),
                                     "tau_std": float(np.std(vals)),
                                     "taus": [int(v) for v in vals]}
    for cut in [0.05, 0.1, 0.2, 0.3]:
        vals = []
        for H in curves:
            b, a = sg.butter(3, cut, "low")
            Hf = sg.filtfilt(b, a, H)
            d1 = np.gradient(Hf)
            ip = int(np.argmax(d1))
            below = np.where(d1[ip:] <= 0.1 * d1[ip])[0]
            j = ip + int(below[0]) if len(below) else len(taus) - 1
            vals.append(taus[j])
        out["cutoffs"][str(cut)] = {"tau_mean": float(np.mean(vals)),
                                    "tau_std": float(np.std(vals))}
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
        vals = [lpf_sem_from_curve(H, taus, alpha)["tau"] for H in curves]
        out["alphas"][str(alpha)] = {"tau_mean": float(np.mean(vals)),
                                     "tau_std": float(np.std(vals)),
                                     "taus": [int(v) for v in vals]}
    return out


def community_sweep(n_per_class=6):
    """Downstream separability vs edge-removal ratio and recurrence rate."""
    X, y = data.load_rotor180()
    idx = _subset(y, n_per_class)
    yy = y[idx]
    taus_dims = []
    for i in idx:
        tau, dim, _ = psr.estimate_many(X[i], ["lpf_sem"])["lpf_sem"]
        taus_dims.append((tau, dim))

    out = {"remove_ratio": {}, "recurrence_percentile": {}, "vmd_K": {}}

    # Below roughly 0.6 the sparsified graph is still dense enough that the
    # maximal-clique enumeration blows up exponentially, so the sweep covers
    # the tractable region only.
    for rr in [0.65, 0.70, 0.75, 0.80, 0.85]:
        feats = []
        for (i, (tau, dim)) in zip(idx, taus_dims):
            r = topology.signal_to_hypergraph(X[i], tau, dim,
                                              remove_ratio=rr)
            feats.append(r["features"])
        F = stats.to_matrix(feats, FEATURE_KEYS)
        s = stats.summarise(F, yy, FEATURE_KEYS)
        out["remove_ratio"][str(rr)] = {
            "nc_cv_acc": s["nc_cv_acc"], "fisher": s["fisher_mean"],
            "hedge_mean": float(np.mean([f["n_hyperedges"] for f in feats])),
            "overlap_mean": float(np.mean([f["overlap_ratio"] for f in feats])),
        }

    for pct in [1.0, 2.0, 3.0, 5.0, 8.0]:
        feats = []
        for (i, (tau, dim)) in zip(idx, taus_dims):
            r = topology.signal_to_hypergraph(X[i], tau, dim, percentile=pct)
            feats.append(r["features"])
        F = stats.to_matrix(feats, FEATURE_KEYS)
        s = stats.summarise(F, yy, FEATURE_KEYS)
        out["recurrence_percentile"][str(pct)] = {
            "nc_cv_acc": s["nc_cv_acc"], "fisher": s["fisher_mean"],
            "hedge_mean": float(np.mean([f["n_hyperedges"] for f in feats])),
        }

    import vmdutil
    for K in [2, 3, 4]:
        feats = []
        for i in idx:
            u, _ = vmdutil.decompose(X[i], K=K)
            d = {}
            for k in range(K):
                tau, dim, _ = psr.estimate_many(u[k], ["lpf_sem"])["lpf_sem"]
                f = topology.signal_to_hypergraph(u[k], tau, dim)["features"]
                for key in FEATURE_KEYS:
                    d[f"L{k}_{key}"] = f[key]
            feats.append(d)
        keys = list(feats[0].keys())
        F = stats.to_matrix(feats, keys)
        s = stats.summarise(F, yy, keys)
        out["vmd_K"][str(K)] = {"nc_cv_acc": s["nc_cv_acc"],
                                "fisher": s["fisher_mean"]}
    return out


def main():
    t0 = time.time()
    res = {}
    print("exp2 filter sweep ...", flush=True)
    res["filter"] = filter_sweep()
    print("exp2 community sweep ...", flush=True)
    res["community"] = community_sweep()
    res["runtime_s"] = time.time() - t0
    (RESULTS / "exp2.json").write_text(json.dumps(res, indent=1))
    print("exp2 done in", round(res["runtime_s"], 1), "s")


if __name__ == "__main__":
    main()
