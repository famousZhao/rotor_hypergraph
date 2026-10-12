"""Experiment 12: downstream separability of the estimator variants on the full
180-record set AFTER per-record amplitude normalisation (the condition the
paper's Experiment 7 uses for the fair comparison).

The full-set experiment 11 found lpf_fit -- and the other small-delay variants
-- to raise the hypergraph separability far above lpf_sem.  Because the rotor
classes differ in amplitude, that advantage may be partly an amplitude cue.  This
check repeats exp11 with every record scaled to unit variance, which removes the
amplitude channel and isolates the waveform/higher-order structure.
"""
import json
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
    Xn = X / np.maximum(np.std(X, axis=1, keepdims=True), 1e-12)
    out = {"normalised": True, "methods": {}}
    for m in METHODS:
        feats = []
        for i in range(len(Xn)):
            tau, dim, _ = psr.estimate_many(Xn[i], [m])[m]
            r = topology.signal_to_hypergraph(Xn[i], tau, dim)
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
    (RESULTS / "exp12_normsep.json").write_text(json.dumps(out, indent=1))
    print("exp12 done -> results/exp12_normsep.json")


if __name__ == "__main__":
    main()
