"""Experiment 6: computational cost.

Two things are measured.

6a  wall-clock cost of every stage of the pipeline on a single measured
    signal, so the reader can see where the time goes.
6b  how the two dominant stages (symbolic-entropy curve and recurrence
    network) scale with the working length N.

Timings are averaged over repeats; the first call of each stage is discarded
because of import and cache warm-up.
"""
import json
import time
import numpy as np

import data
import chaos
import psr
import topology
import topology_reference
import vmdutil
from config import RESULTS, SEED, TAU_MAX

REPS = 5
LENGTHS = [512, 1024, 2048, 4096, 8192]


def _time(fn, reps=REPS):
    """Mean and std wall-clock seconds over `reps` calls."""
    ts = []
    for r in range(reps + 1):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        if r > 0:
            ts.append(dt)
    return float(np.mean(ts)), float(np.std(ts)), out


def stage_costs(x):
    """Per-stage cost on one signal of the given length."""
    taus = np.arange(1, TAU_MAX + 1)

    def _curve():
        return psr.symbolic_entropy_curve(x, taus)

    _, _, H = _time(_curve)
    tau = int(psr.lpf_sem_from_curve(H, taus)["tau"])
    dim, _ = psr.cao_fnn(x, tau, 12)
    traj = psr.reconstruct(np.asarray(x, float), tau, dim)
    A, n_nodes = topology.recurrence_network(traj)
    stages = {"signal_length": int(len(x)), "tau": tau, "dim": dim,
              "n_nodes": int(n_nodes), "n_edges": int(A.nnz // 2)}

    m, s, _ = _time(lambda: psr.symbolic_entropy_curve(x, taus))
    stages["symbolic_entropy_curve"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: psr.lpf_sem_from_curve(H, taus))
    stages["lpf_smooth_and_derivative"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: psr.mutual_information(x, taus))
    stages["mutual_information"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: psr.autocorrelation(x, taus))
    stages["autocorrelation"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: psr.cc_method(x))
    stages["cc_method"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: psr.cao_fnn(x, tau, 12))
    stages["cao_fnn"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: topology.recurrence_network(traj))
    stages["recurrence_network"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: topology.ni_louvain(A))
    stages["ni_louvain"] = {"mean_s": m, "std_s": s}
    comms = topology.ni_louvain(A)
    hg = topology.build_hypergraph(comms, n_nodes)
    m, s, _ = _time(lambda: topology.hypergraph_features(hg))
    stages["hypergraph_features"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: vmdutil.decompose(x, K=3))
    stages["vmd_K3"] = {"mean_s": m, "std_s": s}
    m, s, _ = _time(lambda: topology.signal_to_hypergraph(x, tau, dim))
    stages["full_pipeline"] = {"mean_s": m, "std_s": s}
    # the implementation that the optimised stage replaces, same signal
    m, s, _ = _time(lambda: topology_reference.ni_louvain_reference(A))
    stages["ni_louvain_reference"] = {"mean_s": m, "std_s": s}
    return stages


def scaling(x_long):
    """Cost of the two dominant stages against the working length N."""
    taus = np.arange(1, TAU_MAX + 1)
    out = {}
    for n in LENGTHS:
        if n > len(x_long):
            continue
        x = x_long[:n]
        m, s, H = _time(lambda: psr.symbolic_entropy_curve(x, taus),
                        reps=3)
        tau = int(psr.lpf_sem_from_curve(H, taus)["tau"])
        dim, _ = psr.cao_fnn(x, tau, 12)
        traj = psr.reconstruct(np.asarray(x, float), tau, dim)
        mr, sr, (A, nn) = _time(lambda: topology.recurrence_network(traj),
                                reps=3)
        out[str(n)] = {
            "symbolic_entropy_curve": {"mean_s": m, "std_s": s},
            "recurrence_network": {"mean_s": mr, "std_s": sr},
            "n_nodes": int(nn), "n_edges": int(A.nnz // 2),
        }
    return out


def main():
    t0 = time.time()
    X, y = data.load_rotor180()
    rng = np.random.default_rng(SEED)
    res = {"reps": REPS, "lengths": LENGTHS}
    print("6a stage costs ...", flush=True)
    res["stages"] = stage_costs(X[0])
    # average the per-stage costs over five signals to reduce timing noise
    acc = {k: [v["mean_s"]] for k, v in res["stages"].items()
           if isinstance(v, dict) and "mean_s" in v}
    for i in [1, 2, 3, 4]:
        st = stage_costs(X[i])
        for k, v in st.items():
            if isinstance(v, dict) and "mean_s" in v:
                acc.setdefault(k, []).append(v["mean_s"])
    res["stages_mean_over_5"] = {k: float(np.mean(v)) for k, v in acc.items()}
    print("6b scaling ...", flush=True)
    # The measured records are 2048 points, too short for a scaling law, so a
    # chaotic record of 10000 points is used as the scaling series instead.
    x_long, _, _ = chaos.build("Lorenz", n=12000)
    res["scaling_source"] = "Lorenz, %d points" % len(x_long)
    res["scaling"] = scaling(x_long)
    res["runtime_s"] = time.time() - t0
    (RESULTS / "exp6.json").write_text(json.dumps(res, indent=1))
    print("exp6 done in", round(res["runtime_s"], 1), "s")


if __name__ == "__main__":
    main()
