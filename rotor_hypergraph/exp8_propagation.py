"""Experiment 8: does the delay-stability gain propagate into the downstream
hypergraph features?

The question is asked with the estimator held fixed and only its stability
varying.  For each delay rule the same recurrence-network / overlapping-
community / hypergraph pipeline is run, and two things are recorded at each
noise level:

  * the dispersion of the estimated delay within each class, which measures
    how stable the rule is on that record set;
  * the separability of the resulting hypergraph descriptor, which is the
    quantity the delay is supposed to feed.

If stability propagates, the separability of the stable rule should decay
more slowly with noise than that of the unstable rules.  The same comparison
is repeated on the VMD multi-resolution descriptor, where every intrinsic
mode is embedded separately.

Results go to results/exp8_<dataset>.json.
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
from exp3_hypergraph import FEATURE_KEYS

METHODS = ["lpf_sem", "raw_se", "mi"]
SNRS = [None, 15.0, 10.0, 5.0]           # None is the unperturbed record
NOISE_SEEDS = [0, 1, 2]                  # noise realisations averaged over
DISP_SNRS = [None, 20.0, 15.0, 10.0, 5.0, 0.0]
DISP_SEEDS = list(range(10))


def _features_for_signal(x, psr_method, community="ni_louvain"):
    tau, dim, _ = psr.estimate_many(np.asarray(x, float), [psr_method])[psr_method]
    r = topology.signal_to_hypergraph(x, tau, dim, method=community)
    f = dict(r["features"])
    f["tau"] = tau
    f["dim"] = dim
    return f


def _subset(y, n_per_class):
    return np.concatenate([np.where(y == c)[0][:n_per_class]
                           for c in np.unique(y)])


def _tau_dispersion(feats, y):
    """Mean within-class standard deviation of the estimated delay."""
    taus = np.array([f["tau"] for f in feats], float)
    return float(np.mean([taus[y == c].std() for c in np.unique(y)]))


def _condition(X, snr, rng):
    if snr is None:
        return X
    return np.array([data.add_awgn(X[i], snr, rng) for i in range(len(X))])


def delay_dispersion(X, y, tag, methods=METHODS, snrs=DISP_SNRS,
                     seeds=DISP_SEEDS, n_per_class=15):
    """How far the estimated delay moves under noise, averaged over many noise
    realisations.  Cheap, because only the estimator is run.

    A single realisation is not enough: at the lowest signal-to-noise ratio
    whether a given record keeps its delay depends on the realisation, so the
    mean over ten of them is what separates a stable rule from an unstable
    one.
    """
    idx = _subset(y, n_per_class)
    yy = y[idx]
    out = {"n_per_class": n_per_class, "n_noise_seeds": len(seeds),
           "conditions": {}}
    for snr in snrs:
        key = "clean" if snr is None else f"{snr:g}"
        entry = {}
        for m in methods:
            disps, means = [], []
            for sd in seeds:
                rng = np.random.default_rng(1000 + sd)
                Xs = _condition(X, snr, rng)
                taus = [psr.estimate_many(Xs[i], [m])[m][0] for i in idx]
                taus = np.array(taus, float)
                disps.append(float(np.mean([taus[yy == c].std()
                                            for c in np.unique(yy)])))
                means.append(float(taus.mean()))
            entry[m] = {"tau_dispersion": float(np.mean(disps)),
                        "tau_dispersion_std": float(np.std(disps)),
                        "tau_mean": float(np.mean(means))}
            print(f"  {tag} disp {m:8s} {key:5s} "
                  f"tau_disp={entry[m]['tau_dispersion']:.2f}"
                  f"+-{entry[m]['tau_dispersion_std']:.2f}", flush=True)
        out["conditions"][key] = entry
    return out


def run_single(X, y, tag, n_per_class=15, methods=METHODS, snrs=SNRS,
               noise_seeds=NOISE_SEEDS):
    """Separability of the hypergraph descriptor against noise.

    For a given noise seed every method is evaluated on exactly the same
    perturbed records, so the comparison is not confounded by the noise
    realisation.  Results are averaged over the seeds.
    """
    idx = _subset(y, n_per_class)
    yy = y[idx]
    out = {"dataset": tag, "n_per_class": n_per_class,
           "n_noise_seeds": len(noise_seeds),
           "snrs": ["clean" if s is None else s for s in snrs], "methods": {}}
    for m in methods:
        rec = {}
        for snr in snrs:
            key = "clean" if snr is None else f"{snr:g}"
            accs, fishes, sils, disps, hedges, tmeans = [], [], [], [], [], []
            for sd in noise_seeds:
                rng = np.random.default_rng(2000 + sd)
                Xs = _condition(X, snr, rng)
                feats = [_features_for_signal(Xs[i], m) for i in idx]
                F = stats.to_matrix(feats, FEATURE_KEYS)
                s = stats.summarise(F, yy, FEATURE_KEYS)
                accs.append(s["nc_cv_acc"]); fishes.append(s["fisher_mean"])
                sils.append(s["silhouette"])
                disps.append(_tau_dispersion(feats, yy))
                hedges.append(float(np.mean([f["n_hyperedges"] for f in feats])))
                tmeans.append(float(np.mean([f["tau"] for f in feats])))
            rec[key] = {
                "nc_cv_acc": float(np.mean(accs)),
                "nc_cv_acc_std": float(np.std(accs)),
                "fisher_mean": float(np.mean(fishes)),
                "fisher_std": float(np.std(fishes)),
                "silhouette": float(np.mean(sils)),
                "tau_mean": float(np.mean(tmeans)),
                "tau_dispersion": float(np.mean(disps)),
                "n_hyperedges": float(np.mean(hedges)),
            }
            print(f"  {tag} {m:8s} {key:5s} acc={rec[key]['nc_cv_acc']:.3f}"
                  f"+-{rec[key]['nc_cv_acc_std']:.3f} "
                  f"fish={rec[key]['fisher_mean']:.3f} "
                  f"tau_disp={rec[key]['tau_dispersion']:.2f}", flush=True)
        base = rec["clean"]["nc_cv_acc"]
        for k, v in rec.items():
            v["acc_retained"] = float(v["nc_cv_acc"] / base) if base else 0.0
        out["methods"][m] = rec
    return out


def run_multires(X, y, tag, methods=("lpf_sem", "raw_se"), n_per_class=12, K=3,
                 snrs=(None, 15.0, 10.0, 5.0), noise_seeds=(0, 1)):
    """The same question one level down: every VMD mode is embedded on its own.

    Narrow-band modes are shorter and less periodic than the raw record, so
    the entropy curve is rougher and the delay rule has less to hold on to.
    """
    idx = _subset(y, n_per_class)
    yy = y[idx]
    out = {"dataset": tag, "n_per_class": n_per_class, "K": K,
           "n_noise_seeds": len(noise_seeds), "methods": {}}
    for m in methods:
        rec = {}
        for snr in snrs:
            key = "clean" if snr is None else f"{snr:g}"
            comb_acc, lvl_acc, disps = [], [], []
            for sd in noise_seeds:
                rng = np.random.default_rng(3000 + sd)
                Xs = _condition(X, snr, rng)
                levels = {k: [] for k in range(K)}
                for i in idx:
                    u, _w = vmdutil.decompose(Xs[i], K=K)
                    for k in range(K):
                        levels[k].append(_features_for_signal(u[k], m))
                comb = []
                for j in range(len(yy)):
                    d = {}
                    for k in range(K):
                        for key2 in FEATURE_KEYS:
                            d[f"L{k}_{key2}"] = levels[k][j][key2]
                    comb.append(d)
                ckeys = list(comb[0].keys())
                Fc = stats.to_matrix(comb, ckeys)
                comb_acc.append(stats.summarise(Fc, yy, ckeys)["nc_cv_acc"])
                lvl_acc.append([
                    stats.summarise(stats.to_matrix(levels[k], FEATURE_KEYS),
                                    yy, FEATURE_KEYS)["nc_cv_acc"]
                    for k in range(K)])
                disps.append(_tau_dispersion(levels[0], yy))
            entry = {"combined": float(np.mean(comb_acc)),
                     "combined_std": float(np.std(comb_acc)),
                     "levels": [float(x) for x in np.mean(lvl_acc, axis=0)],
                     "tau_dispersion": float(np.mean(disps))}
            rec[key] = entry
            print(f"  {tag} mr {m:8s} {key:5s} combined="
                  f"{entry['combined']:.3f}+-{entry['combined_std']:.3f} "
                  f"tau_disp={entry['tau_dispersion']:.2f}", flush=True)
        base = rec["clean"]["combined"]
        for k, v in rec.items():
            v["acc_retained"] = float(v["combined"] / base) if base else 0.0
        out["methods"][m] = rec
    return out


def main(which="all", n_per_class=15):
    if which in ("all", "rotor"):
        print("exp8 rotor ...", flush=True)
        X, y = data.load_rotor180()
        r = {}
        print(" rotor delay dispersion ...", flush=True)
        r["delay_dispersion"] = delay_dispersion(X, y, "rotor")
        print(" rotor separability vs noise ...", flush=True)
        r.update(run_single(X, y, "rotor", n_per_class=n_per_class))
        print(" rotor multi-resolution ...", flush=True)
        r["multiresolution"] = run_multires(X, y, "rotor")
        (RESULTS / "exp8_rotor.json").write_text(json.dumps(r, indent=1))
    if which in ("all", "mfpt"):
        print("exp8 mfpt ...", flush=True)
        X, y = data.load_mfpt()
        r = {}
        print(" mfpt delay dispersion ...", flush=True)
        r["delay_dispersion"] = delay_dispersion(X, y, "mfpt")
        print(" mfpt separability vs noise ...", flush=True)
        r.update(run_single(X, y, "mfpt", n_per_class=n_per_class))
        print(" mfpt multi-resolution ...", flush=True)
        r["multiresolution"] = run_multires(X, y, "mfpt")
        (RESULTS / "exp8_mfpt.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(a)
