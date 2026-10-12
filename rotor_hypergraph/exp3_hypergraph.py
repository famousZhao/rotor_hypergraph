"""Experiments 3 and 4: hypergraph topology.

3   single-resolution hypergraphs for the four measured classes, comparing the
    reconstruction methods that feed the pipeline.
4   VMD multi-resolution hypergraphs, plus the secondary study on the
    simulated-coupling classes.

Results are written to results/exp3.json and results/exp4.json.
"""
import json
import time
import numpy as np

import data
import psr
import stats
import topology
import vmdutil
from config import RESULTS, SEED

PSR_METHODS = ["lpf_sem", "raw_se", "mi", "cc"]

FEATURE_KEYS = [
    "n_nodes", "n_hyperedges",
    "hedge_size_mean", "hedge_size_std", "hedge_size_max",
    "hedge_size_entropy", "node_degree_mean", "node_degree_std",
    "overlap_ratio", "hedge_interaction",
    "giant_fraction", "algebraic_connectivity", "spectral_radius",
    "spectral_gap", "spectral_entropy",
]


def _features_for_signal(x, psr_method="lpf_sem", community="ni_louvain"):
    tau, dim, _ = psr.estimate_many(np.asarray(x, float), [psr_method])[psr_method]
    r = topology.signal_to_hypergraph(x, tau, dim, method=community)
    f = dict(r["features"])
    f["tau"] = tau
    f["dim"] = dim
    return f


def _run(X, idx, psr_method="lpf_sem", community="ni_louvain"):
    feats = []
    for i in idx:
        feats.append(_features_for_signal(X[i], psr_method, community))
    return feats


def _subset(y, n_per_class):
    return np.concatenate([np.where(y == c)[0][:n_per_class]
                           for c in np.unique(y)])


def per_class_table(feats, y, keys):
    out = {}
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        out[str(c)] = {k: {"mean": float(np.mean([feats[i][k] for i in idx])),
                           "std": float(np.std([feats[i][k] for i in idx]))}
                       for k in keys}
    return out


def exp3_full():
    """All 180 measured signals, proposed method only."""
    X, y = data.load_rotor180()
    t0 = time.time()
    feats = _run(X, np.arange(len(X)), "lpf_sem", "ni_louvain")
    F = stats.to_matrix(feats, FEATURE_KEYS)
    res = {
        "n_signals": len(X),
        "per_class": per_class_table(feats, y, FEATURE_KEYS),
        "separability": stats.summarise(F, y, FEATURE_KEYS),
        "features": feats,
        "runtime_s": time.time() - t0,
    }
    return res


def exp3_method_comparison(n_per_class=20):
    """Does the reconstruction method change the downstream topology?"""
    X, y = data.load_rotor180()
    idx = _subset(y, n_per_class)
    yy = y[idx]
    res = {"n_per_class": n_per_class, "methods": {}}
    t0 = time.time()
    for m in PSR_METHODS:
        feats = _run(X, idx, m, "ni_louvain")
        F = stats.to_matrix(feats, FEATURE_KEYS)
        res["methods"][m] = {
            "separability": stats.summarise(F, yy, FEATURE_KEYS),
            "tau_mean": float(np.mean([f["tau"] for f in feats])),
            "tau_std": float(np.std([f["tau"] for f in feats])),
            "hedge_mean": float(np.mean([f["n_hyperedges"] for f in feats])),
        }
    res["runtime_s"] = time.time() - t0
    return res


def exp4_multiresolution(n_per_class=12, K=3):
    """Per-IMF hypergraphs.  Returns features at each resolution level."""
    from psr import reconstruct
    X, y = data.load_rotor180()
    idx = _subset(y, n_per_class)
    yy = y[idx]
    levels = {k: [] for k in range(K)}
    t0 = time.time()
    for i in idx:
        u, _w = vmdutil.decompose(X[i], K=K)
        for k in range(K):
            levels[k].append(_features_for_signal(u[k]))
    res = {"n_per_class": n_per_class, "K": K, "levels": {}}
    for k in range(K):
        F = stats.to_matrix(levels[k], FEATURE_KEYS)
        res["levels"][str(k)] = {
            "separability": stats.summarise(F, yy, FEATURE_KEYS),
            "hedge_mean": float(np.mean([f["n_hyperedges"] for f in levels[k]])),
            "size_mean": float(np.mean([f["hedge_size_mean"] for f in levels[k]])),
            "inter_mean": float(np.mean([f["hedge_interaction"] for f in levels[k]])),
            "gc_mean": float(np.mean([f["giant_fraction"] for f in levels[k]])),
            "features": levels[k],
        }
    # concatenated multi-resolution descriptor
    comb = []
    for j in range(len(yy)):
        d = {}
        for k in range(K):
            for key in FEATURE_KEYS:
                d[f"L{k}_{key}"] = levels[k][j][key]
        comb.append(d)
    ckeys = list(comb[0].keys())
    Fc = stats.to_matrix(comb, ckeys)
    res["combined"] = {"separability": stats.summarise(Fc, yy, ckeys)}
    res["runtime_s"] = time.time() - t0
    return res


def exp4_coupled(n_per_class=12, K=3):
    """Secondary study on the eight-class set (coupled classes are simulated)."""
    X, y = data.load_rotor360()
    idx = np.concatenate([np.where(y == c)[0][:n_per_class]
                          for c in range(8)])
    yy = y[idx]
    t0 = time.time()
    raw = [_features_for_signal(X[i]) for i in idx]
    F = stats.to_matrix(raw, FEATURE_KEYS)
    res = {
        "n_per_class": n_per_class,
        "note": "classes 4-7 are arithmetic means of classes 1-3 (simulated)",
        "single_resolution": {
            "separability_8class": stats.summarise(F, yy, FEATURE_KEYS),
            "per_class": per_class_table(raw, yy, FEATURE_KEYS),
        },
        "runtime_s": time.time() - t0,
    }
    # measured-only subset (classes 0-3) for reference
    mask = yy < 4
    res["measured_only"] = {
        "separability_4class": stats.summarise(F[mask], yy[mask], FEATURE_KEYS),
    }
    return res


def main(which="all", n_per_class=12):
    if which in ("all", "3"):
        print("exp3 full ...", flush=True)
        (RESULTS / "exp3_full.json").write_text(
            json.dumps(exp3_full(), indent=1))
        print("exp3 method comparison ...", flush=True)
        (RESULTS / "exp3_methods.json").write_text(
            json.dumps(exp3_method_comparison(), indent=1))
    if which in ("all", "4"):
        print("exp4 multi-resolution ...", flush=True)
        (RESULTS / "exp4_multires.json").write_text(
            json.dumps(exp4_multiresolution(n_per_class=n_per_class), indent=1))
        print("exp4 coupled secondary ...", flush=True)
        (RESULTS / "exp4_coupled.json").write_text(
            json.dumps(exp4_coupled(n_per_class=n_per_class), indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
