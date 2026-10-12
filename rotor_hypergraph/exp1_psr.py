"""Experiment 1: phase-space reconstruction parameter estimation.

1a  accuracy on five chaotic systems
1b  rotor delay-time estimates per class, raw symbolic entropy vs LPF-SEM
1c  robustness to additive noise and to impulsive glitches

Everything is written to results/exp1.json.  No number in the manuscript is
typed by hand; the tables are generated from this file.
"""
import json
import time
import numpy as np

import chaos
import data
import psr
from lyapunov import rosenstein
from config import RESULTS, SEED

METHODS = ["lpf_sem", "raw_se", "mi", "acf", "cc"]
SNRS = [20, 10, 5, 0, -5]
NCH = 4000          # working length for chaotic series
NREF = 20000        # length used to define the consensus reference


def _spikes(x, density, rng, amplitude=6.0):
    """Inject impulses scaled to the signal standard deviation."""
    y = x.copy()
    n = len(y)
    k = max(1, int(density * n))
    pos = rng.choice(n, size=k, replace=False)
    y[pos] += amplitude * np.std(x) * rng.choice([-1.0, 1.0], size=k)
    return y


def _consensus(x_long):
    """Reference (tau, dim): median of the five estimators on a long clean series."""
    res = psr.estimate_many(x_long, METHODS)
    taus = [res[m][0] for m in METHODS]
    dims = [res[m][1] for m in METHODS]
    return int(np.median(taus)), int(np.median(dims)), taus, dims


def chaos_accuracy():
    out = {}
    for nm in chaos.SYSTEMS:
        x_long, dt, lle_ref = chaos.build(nm, n=NREF)
        ref_tau, ref_dim, all_tau, all_dim = _consensus(x_long)
        x = x_long[:NCH]
        rec = {"ref_tau": ref_tau, "ref_dim": ref_dim, "lle_ref": lle_ref,
               "dt": dt, "consensus_taus": all_tau,
               "consensus_dims": all_dim, "methods": {}}
        res = psr.estimate_many(x, METHODS)
        for m in METHODS:
            tau, dim, _ = res[m]
            lle, _ = rosenstein(x, tau, dim, dt=dt)
            rec["methods"][m] = {
                "tau": tau, "dim": dim, "lle": lle,
                "tau_err": abs(tau - ref_tau) / ref_tau,
                "dim_err": abs(dim - ref_dim),
                "lle_err": abs(lle - lle_ref) / lle_ref,
            }
        out[nm] = rec
    return out


def chaos_noise():
    """Delay estimate against SNR, for all five systems."""
    out = {}
    for nm in chaos.SYSTEMS:
        x_long, _, _ = chaos.build(nm, n=NREF)
        ref_tau, ref_dim, _, _ = _consensus(x_long)
        x = x_long[:NCH]
        rng = np.random.default_rng(SEED)
        clean = psr.estimate_many(x, METHODS)
        rec = {"ref_tau": ref_tau, "ref_dim": ref_dim,
               "clean": {m: {"tau": clean[m][0], "dim": clean[m][1]}
                         for m in METHODS},
               "snr": {}}
        for snr in SNRS:
            per = {m: {"taus": [], "dims": []} for m in METHODS}
            for _rep in range(5):
                xn = data.add_awgn(x, snr, rng)
                r = psr.estimate_many(xn, METHODS)
                for m in METHODS:
                    per[m]["taus"].append(r[m][0])
                    per[m]["dims"].append(r[m][1])
            row = {}
            for m in METHODS:
                taus, dims = per[m]["taus"], per[m]["dims"]
                tau_c = float(np.median(taus))
                row[m] = {
                    "tau": tau_c,
                    "tau_std": float(np.std(taus)),
                    "dim": float(np.median(dims)),
                    "rel_err": float(abs(tau_c - clean[m][0]) /
                                     max(1, clean[m][0])),
                }
            rec["snr"][str(snr)] = row
        out[nm] = rec
    return out


def chaos_spikes():
    """Delay estimate against impulse density (the glitch mechanism)."""
    densities = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    out = {}
    for nm in chaos.SYSTEMS:
        x_long, _, _ = chaos.build(nm, n=NREF)
        x = x_long[:NCH]
        clean = psr.estimate_many(x, METHODS)
        curves = {m: [] for m in METHODS}
        for d in densities:
            per = {m: [] for m in METHODS}
            for rep in range(5):
                rng = np.random.default_rng(SEED + rep)
                xs = _spikes(x, d, rng) if d > 0 else x
                r = psr.estimate_many(xs, METHODS)
                for m in METHODS:
                    per[m].append(r[m][0])
            for m in METHODS:
                tau0 = clean[m][0]
                med = float(np.median(per[m]))
                curves[m].append({"density": d, "tau": med,
                                  "std": float(np.std(per[m])),
                                  "rel_err": float(abs(med - tau0) /
                                                   max(1, tau0))})
        rec = {m: {"clean_tau": int(clean[m][0]), "curve": curves[m]}
               for m in METHODS}
        out[nm] = rec
    return out


def rotor_tau():
    X, y = data.load_rotor180()
    taus = {m: [] for m in METHODS}
    dims = {m: [] for m in METHODS}
    for i in range(len(X)):
        r = psr.estimate_many(X[i], METHODS)
        for m in METHODS:
            taus[m].append(r[m][0])
            dims[m].append(r[m][1])
    out = {"per_method": {}, "per_class": {}}
    for m in METHODS:
        out["per_method"][m] = {
            "tau_mean": float(np.mean(taus[m])),
            "tau_std": float(np.std(taus[m])),
            "tau_median": float(np.median(taus[m])),
            "dim_mean": float(np.mean(dims[m])),
            "dim_std": float(np.std(dims[m])),
            "taus": [int(t) for t in taus[m]],
            "dims": [int(d) for d in dims[m]],
        }
        for c in range(4):
            idx = np.where(y == c)[0]
            out["per_class"].setdefault(str(c), {})[m] = {
                "tau_mean": float(np.mean([taus[m][i] for i in idx])),
                "tau_std": float(np.std([taus[m][i] for i in idx])),
                "dim_mean": float(np.mean([dims[m][i] for i in idx])),
            }
    return out


def rotor_noise(n_per_class=8, reps=3):
    X, y = data.load_rotor180()
    idx = np.concatenate([np.where(y == c)[0][:n_per_class] for c in range(4)])
    rng = np.random.default_rng(SEED)
    clean_tau = {m: [] for m in METHODS}
    for i in idx:
        r = psr.estimate_many(X[i], METHODS)
        for m in METHODS:
            clean_tau[m].append(r[m][0])
    out = {"clean": {m: {"tau_mean": float(np.mean(clean_tau[m])),
                         "tau_std": float(np.std(clean_tau[m]))}
                     for m in METHODS}, "snr": {}}
    for snr in SNRS:
        row = {}
        for m in METHODS:
            devs, stds = [], []
            for pos, i in enumerate(idx):
                taus = []
                for _rep in range(reps):
                    xn = data.add_awgn(X[i], snr, rng)
                    taus.append(psr.estimate_many(xn, METHODS)[m][0])
                devs.append(abs(float(np.median(taus)) - clean_tau[m][pos]))
                stds.append(float(np.std(taus)))
            row[m] = {"mean_abs_dev": float(np.mean(devs)),
                      "mean_std": float(np.mean(stds))}
        out["snr"][str(snr)] = row
    return out


def main():
    t0 = time.time()
    res = {}
    print("1a chaos accuracy ...", flush=True)
    res["chaos_accuracy"] = chaos_accuracy()
    print("1b rotor tau ...", flush=True)
    res["rotor_tau"] = rotor_tau()
    print("1c chaos noise ...", flush=True)
    res["chaos_noise"] = chaos_noise()
    print("1c chaos spikes ...", flush=True)
    res["chaos_spikes"] = chaos_spikes()
    print("1c rotor noise ...", flush=True)
    res["rotor_noise"] = rotor_noise()
    res["runtime_s"] = time.time() - t0
    (RESULTS / "exp1.json").write_text(json.dumps(res, indent=1))
    print("done in", round(res["runtime_s"], 1), "s -> results/exp1.json")


if __name__ == "__main__":
    main()
