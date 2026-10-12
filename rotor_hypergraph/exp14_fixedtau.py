"""Experiment 14: disentangle the tau-scale collinearity of separability.

The reviewer worry: lpf_fit gives tau~2.3 and separability 0.939, while the
other rules give larger tau and lower separability, and the table looks like
separability grows as tau shrinks.  If that is a small-tau artefact -- the
recurrence graph simply gets denser / more connected at small lag regardless of
state -- then the 0.939 is not evidence that tau_c carries state information.

This experiment separates the two hypotheses:
  (H_artefact)  fix ONE global tau for every record; if small fixed tau alone
                already gives ~0.9 separability, the gain is a small-lag artefact.
  (H_adaptive) only a tau that ADAPTS per record (tau_c) separates well; a
                single global tau cannot, so the state-specific scale matters.

For every record we still estimate its own Cao dimension m (the embedding
dimension is a geometric quantity, independent of which delay rule we read),
but we force the delay to a global value tau0 scanned over a grid.  We then
build the hypergraph descriptor exactly as in exp11 and report the
near-neighbour CV accuracy at n-per-class = 12.
"""
import json
import numpy as np

import data
import psr
import stats
import topology
from config import RESULTS
from exp3_hypergraph import FEATURE_KEYS

GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]


def main():
    X, y = data.load_rotor180()

    # cache the per-record Cao dimension once (independent of the tau rule).
    dims = []
    for i in range(len(X)):
        _, dim, _ = psr.estimate_many(X[i], ["lpf_fit"])["lpf_fit"]
        if dim is None or dim <= 0:
            dim = 3
        dims.append(int(dim))
    print("dim cached, mean dim = %.2f" % float(np.mean(dims)), flush=True)

    out = {"fixed_tau": {}, "adaptive_tauc": None}

    # --- H_artefact: one global tau for every record.
    for tau0 in GRID:
        feats = []
        for i in range(len(X)):
            r = topology.signal_to_hypergraph(X[i], tau0, dims[i])
            feats.append(dict(r["features"]))
        F = stats.to_matrix(feats, FEATURE_KEYS)
        s = stats.summarise(F, y, FEATURE_KEYS)
        out["fixed_tau"][str(tau0)] = {
            "nc_cv_acc": s["nc_cv_acc"], "fisher_mean": s["fisher_mean"],
            "silhouette": s["silhouette"],
        }
        print(f"  fixed tau={tau0:2d}  acc={s['nc_cv_acc']:.3f}  "
              f"fish={s['fisher_mean']:.3f}", flush=True)

    # --- H_adaptive: per-record tau_c (lpf_fit), reproduced for the paired number.
    feats = []
    taus = []
    for i in range(len(X)):
        tau, dim, _ = psr.estimate_many(X[i], ["lpf_fit"])["lpf_fit"]
        r = topology.signal_to_hypergraph(X[i], tau, dim)
        feats.append(dict(r["features"]))
        taus.append(tau)
    F = stats.to_matrix(feats, FEATURE_KEYS)
    s = stats.summarise(F, y, FEATURE_KEYS)
    out["adaptive_tauc"] = {
        "nc_cv_acc": s["nc_cv_acc"], "fisher_mean": s["fisher_mean"],
        "silhouette": s["silhouette"],
        "tau_mean": float(np.mean(taus)), "tau_std": float(np.std(taus)),
    }
    print(f"  adaptive tau_c  acc={s['nc_cv_acc']:.3f}  "
          f"fish={s['fisher_mean']:.3f}  tau={np.mean(taus):.2f}", flush=True)

    (RESULTS / "exp14_fixedtau.json").write_text(json.dumps(out, indent=1))
    print("exp14 done -> results/exp14_fixedtau.json")


if __name__ == "__main__":
    main()
