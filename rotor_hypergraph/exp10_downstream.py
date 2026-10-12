"""Experiment 10: downstream separability verification for the estimator
variants, using the authoritative exp8-style protocol.

The exploratory n_per_class=8 single-pass numbers in exp9 are high-variance,
so the separability of the variant delays is re-measured with the protocol of
Experiment 8: n_per_class=12, averaged over several noise seeds at the clean
condition (and at 10 dB).  This decides whether the surprisingly high
separability of lpf_fit in the small sample survives a reliable measurement.
"""
import json
import time
import numpy as np

import data
import psr
import stats
import topology
from config import RESULTS, SEED
from exp3_hypergraph import FEATURE_KEYS
from exp8_propagation import _subset, _condition, _tau_dispersion

METHODS = ["lpf_sem", "raw_se", "lpf_sem_d2", "lpf_curv", "lpf_fit",
           "lpf_adaptive", "pe_sem", "mi"]
SNRS = [None, 10.0]
NOISE_SEEDS = [0, 1, 2]


def _features_for_signal(x, m):
    tau, dim, _ = psr.estimate_many(np.asarray(x, float), [m])[m]
    r = topology.signal_to_hypergraph(x, tau, dim)
    f = dict(r["features"])
    f["tau"] = tau
    return f


def run(n_per_class=12):
    X, y = data.load_rotor180()
    idx = _subset(y, n_per_class)
    yy = y[idx]
    out = {"n_per_class": n_per_class, "noise_seeds": NOISE_SEEDS, "methods": {}}
    for m in METHODS:
        rec = {}
        for snr in SNRS:
            key = "clean" if snr is None else f"{snr:g}"
            accs, fishes, disps, tmeans = [], [], [], []
            for sd in NOISE_SEEDS:
                rng = np.random.default_rng(2000 + sd)
                Xs = _condition(X, snr, rng)
                feats = [_features_for_signal(Xs[i], m) for i in idx]
                F = stats.to_matrix(feats, FEATURE_KEYS)
                s = stats.summarise(F, yy, FEATURE_KEYS)
                accs.append(s["nc_cv_acc"]); fishes.append(s["fisher_mean"])
                disps.append(_tau_dispersion(feats, yy))
                tmeans.append(float(np.mean([f["tau"] for f in feats])))
            rec[key] = {
                "nc_cv_acc": float(np.mean(accs)),
                "nc_cv_acc_std": float(np.std(accs)),
                "fisher_mean": float(np.mean(fishes)),
                "tau_dispersion": float(np.mean(disps)),
                "tau_mean": float(np.mean(tmeans)),
            }
            print(f"  {m:12s} {key:5s} acc={rec[key]['nc_cv_acc']:.3f}"
                  f"+-{rec[key]['nc_cv_acc_std']:.3f} "
                  f"disp={rec[key]['tau_dispersion']:.2f}", flush=True)
        out["methods"][m] = rec
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = run()
    res["runtime_s"] = time.time() - t0
    (RESULTS / "exp10_downstream.json").write_text(json.dumps(res, indent=1))
    print("exp10 done in", round(res["runtime_s"], 1),
          "s -> results/exp10_downstream.json")
