"""Experiment 5: ablation.

A  replace the LPF-SEM delay estimate with the C-C method, everything else
   unchanged;
B  replace NI-Louvain with plain (non-overlapping) Louvain, everything else
   unchanged.

Both are compared with the full pipeline on the same signal subset.
"""
import json
import time
import numpy as np

import data
import psr
import stats
import topology
from exp3_hypergraph import FEATURE_KEYS, _features_for_signal, _subset
from config import RESULTS


def _run(X, idx, psr_method, community):
    return [_features_for_signal(X[i], psr_method, community) for i in idx]


def main(n_per_class=20):
    X, y = data.load_rotor180()
    idx = _subset(y, n_per_class)
    yy = y[idx]
    t0 = time.time()
    variants = {
        "full": ("lpf_sem", "ni_louvain"),
        "ablation_A_psr_cc": ("cc", "ni_louvain"),
        "ablation_B_louvain": ("lpf_sem", "louvain"),
    }
    res = {"n_per_class": n_per_class, "variants": {}}
    for name, (pm, cm) in variants.items():
        feats = _run(X, idx, pm, cm)
        F = stats.to_matrix(feats, FEATURE_KEYS)
        res["variants"][name] = {
            "psr_method": pm, "community": cm,
            "separability": stats.summarise(F, yy, FEATURE_KEYS),
            "hedge_mean": float(np.mean([f["n_hyperedges"] for f in feats])),
            "hedge_size_mean": float(np.mean([f["hedge_size_mean"] for f in feats])),
            "overlap_ratio_mean": float(np.mean([f["overlap_ratio"] for f in feats])),
            "interaction_mean": float(np.mean([f["hedge_interaction"] for f in feats])),
            "features": feats,
        }
    res["runtime_s"] = time.time() - t0
    (RESULTS / "exp5.json").write_text(json.dumps(res, indent=1))
    print("exp5 done in", round(res["runtime_s"], 1), "s")


if __name__ == "__main__":
    main()
