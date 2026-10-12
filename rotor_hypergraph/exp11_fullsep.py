"""Experiment 11: downstream separability of the estimator variants on the FULL
180-record measured set (the exp3-style authoritative protocol, clean).

The small-sample verification (exp10) found lpf_fit -- the saturating-
exponential fit with tau*=tau_c -- to give the highest clean separability.
Because that would revise the paper's Experiment-8 reading that the delay rule
barely moves separability, it is re-checked on every record of the measured
set, matching the protocol that produced exp3_methods.json.
"""
import json
import time
import numpy as np

import data
import psr
import stats
import topology
from config import RESULTS
from exp3_hypergraph import FEATURE_KEYS

METHODS = ["lpf_sem", "raw_se", "lpf_fit", "lpf_curv", "lpf_adaptive",
           "pe_sem", "mi"]


def main():
    X, y = data.load_rotor180()
    out = {"methods": {}}
    for m in METHODS:
        feats = []
        for i in range(len(X)):
            tau, dim, _ = psr.estimate_many(X[i], [m])[m]
            r = topology.signal_to_hypergraph(X[i], tau, dim)
            f = dict(r["features"])
            f["tau"] = tau
            feats.append(f)
        F = stats.to_matrix(feats, FEATURE_KEYS)
        s = stats.summarise(F, y, FEATURE_KEYS)
        out["methods"][m] = {
            "nc_cv_acc": s["nc_cv_acc"], "fisher_mean": s["fisher_mean"],
            "silhouette": s["silhouette"],
            "overlap_mean": float(np.mean([f["overlap_ratio"] for f in feats])),
            "tau_mean": float(np.mean([f["tau"] for f in feats])),
            "tau_std": float(np.std([f["tau"] for f in feats])),
        }
        print(f"  {m:12s} acc={out['methods'][m]['nc_cv_acc']:.3f} "
              f"fish={out['methods'][m]['fisher_mean']:.3f} "
              f"tau={out['methods'][m]['tau_mean']:.1f}"
              f"+-{out['methods'][m]['tau_std']:.2f}", flush=True)
    (RESULTS / "exp11_fullsep.json").write_text(json.dumps(out, indent=1))
    print("exp11 done -> results/exp11_fullsep.json")


if __name__ == "__main__":
    main()
