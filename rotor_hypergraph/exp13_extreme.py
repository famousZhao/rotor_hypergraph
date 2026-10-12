"""Experiment 13: pushing the two weak spots.

(A) Extreme-noise delay robustness.  exp9 already showed that, at 0 dB, the
non-derivative readings (lpf_fit, lpf_adaptive) deviate *less* than the
derivative-threshold rule; this extends the rotor_noise sweep to -5 dB and
checks whether that ordering holds where the saturated rule is reported to
fail.
(B) Separability under noise.  exp10 measured the variant separability at
clean and 10 dB; this extends it to 0 dB and -5 dB, to see how far the
saturation-scale reading (lpf_fit) keeps its clean advantage of 0.939 under
perturbation.
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

METHODS = ["lpf_sem", "raw_se", "lpf_fit", "lpf_adaptive", "lpf_curv",
           "pe_sem", "mi"]
SNRS = [None, 10.0, 0.0, -5.0]
NOISE_SEEDS = [0, 1, 2]


def _features_for_signal(x, m):
    tau, dim, _ = psr.estimate_many(np.asarray(x, float), [m])[m]
    r = topology.signal_to_hypergraph(x, tau, dim)
    f = dict(r["features"])
    f["tau"] = tau
    return f


def downstream(n_per_class=12):
    X, y = data.load_rotor180()
    idx = _subset(y, n_per_class)
    yy = y[idx]
    out = {"n_per_class": n_per_class, "noise_seeds": NOISE_SEEDS, "methods": {}}
    for m in METHODS:
        rec = {}
        for snr in SNRS:
            key = "clean" if snr is None else f"{snr:g}"
            accs, disps = [], []
            for sd in NOISE_SEEDS:
                rng = np.random.default_rng(2000 + sd)
                Xs = _condition(X, snr, rng)
                feats = [_features_for_signal(Xs[i], m) for i in idx]
                F = stats.to_matrix(feats, FEATURE_KEYS)
                s = stats.summarise(F, yy, FEATURE_KEYS)
                accs.append(s["nc_cv_acc"])
                disps.append(_tau_dispersion(feats, yy))
            rec[key] = {
                "nc_cv_acc": float(np.mean(accs)),
                "nc_cv_acc_std": float(np.std(accs)),
                "tau_dispersion": float(np.mean(disps)),
            }
            print(f"  [down] {m:12s} {key:5s} acc={rec[key]['nc_cv_acc']:.3f}"
                  f"+-{rec[key]['nc_cv_acc_std']:.3f} "
                  f"disp={rec[key]['tau_dispersion']:.2f}", flush=True)
        out["methods"][m] = rec
    return out


def extreme_delay_dev(n_per_class=8, reps=3):
    """Mean absolute delay deviation from the clean delay, at 0 dB and -5 dB.

    Reuses the exp9 rotor_noise protocol but only the two extreme SNRs, on the
    methods that matter.
    """
    X, y = data.load_rotor180()
    idx = np.concatenate([np.where(y == c)[0][:n_per_class]
                          for c in range(4)])
    rng = np.random.default_rng(SEED)
    clean = {m: [] for m in METHODS}
    for i in idx:
        r = psr.estimate_many(X[i], METHODS, with_dim=False)
        for m in METHODS:
            clean[m].append(r[m][0])
    out = {"clean_tau_mean": {m: float(np.mean(clean[m])) for m in METHODS},
           "snr": {}}
    for snr in [0.0, -5.0]:
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
        out["snr"][f"{snr:g}"] = row
        print(f"  [noise] done snr={snr:g}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    dn = extreme_delay_dev()
    dd = downstream()
    res = {"delay_dev": dn, "downstream": dd, "runtime_s": time.time() - t0}
    (RESULTS / "exp13_extreme.json").write_text(json.dumps(res, indent=1))
    print("exp13 done in", round(res["runtime_s"], 1),
          "s -> results/exp13_extreme.json")
