"""Generate every figure of the manuscript from the results JSON files.

Nothing here recomputes a result; the figures are drawn from what the
experiment scripts wrote, except Fig. 2 which needs one symbolic-entropy
curve and reads it straight from the data.
"""
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
import psr
from config import FIGURES, RESULTS, TAU_MAX

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

METHODS = ["lpf_sem", "raw_se", "mi", "acf", "cc"]
LABEL = {"lpf_sem": "LPF-SEM (proposed)", "raw_se": "Symbolic entropy",
         "mi": "Mutual information", "acf": "Autocorrelation",
         "cc": "C-C method"}
MARK = {"lpf_sem": "o", "raw_se": "s", "mi": "^", "acf": "v", "cc": "D"}
COLOR = {"lpf_sem": "#b2182b", "raw_se": "#2166ac", "mi": "#4d9221",
         "acf": "#b8860b", "cc": "#762a83"}
STYLE = {"lpf_sem": "-", "raw_se": "--", "mi": "-.", "acf": ":", "cc": "-"}


def _load(name):
    p = RESULTS / name
    if not p.exists():
        print("  missing", name)
        return None
    return json.loads(p.read_text())


def _save(fig, name):
    out = FIGURES / name
    fig.savefig(out)
    plt.close(fig)
    print("  wrote", name)


# ------------------------------------------------------------------ fig 1
def fig_pipeline():
    """Schematic of the single-resolution chain and its multi-resolution use."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig = plt.figure(figsize=(8.0, 2.20))
    ax = fig.add_axes([0.015, 0.08, 0.60, 0.86])
    ax2 = fig.add_axes([0.70, 0.08, 0.285, 0.86])

    def box(a, x, y, text, w=1.86, h=0.62, fc="#eef3fb", ec="#31506e",
            fs=6.0, ls="-"):
        a.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=0.7, edgecolor=ec, facecolor=fc, linestyle=ls))
        a.text(x, y, text, ha="center", va="center", fontsize=fs,
               linespacing=1.2)

    def arrow(a, x1, y1, x2, y2, ls="-"):
        a.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
            linewidth=0.7, color="#31506e", linestyle=ls))

    # ---- panel (a), the single-resolution chain
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.1)
    ax.set_axis_off()
    ax.grid(False)
    ax.set_title("(a) single-resolution chain", fontsize=8.5)

    ya, yb = 2.35, 1.15
    xa = [1.0, 3.0, 5.0, 7.0, 9.0]
    box(ax, xa[0], ya, "measured\nvibration $x(t)$",
        fc="#e8f3e8", ec="#2f5d2f")
    box(ax, xa[1], ya, "symbolic entropy\ncurve $H(\\tau)$")
    box(ax, xa[2], ya, "Butterworth\nlow-pass filter")
    box(ax, xa[3], ya, "derivative\nthreshold")
    box(ax, xa[4], ya, "delay $\\tau^\\star$", fc="#fbf0e3", ec="#8a5a1e")
    for i in range(4):
        arrow(ax, xa[i] + 0.93, ya, xa[i + 1] - 0.93, ya)

    xb = list(reversed(xa))
    box(ax, xb[0], yb, "dimension $m$\nCao criterion")
    box(ax, xb[1], yb, "delay\nembedding")
    box(ax, xb[2], yb, "recurrence\nnetwork")
    box(ax, xb[3], yb, "NI-Louvain\noverlap",
        fc="#fbf0e3", ec="#8a5a1e")
    box(ax, xb[4], yb, "hypergraph\nfeatures $\\mathbf{f}$",
        fc="#e8f3e8", ec="#2f5d2f")
    for i in range(4):
        arrow(ax, xb[i] - 0.93, yb, xb[i + 1] + 0.93, yb)
    arrow(ax, xa[4], ya - 0.31, xb[0], yb + 0.31)

    # ---- panel (b), the multi-resolution extension
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 3.1)
    ax2.set_axis_off()
    ax2.grid(False)
    ax2.set_title("(b) multi-resolution extension", fontsize=8.5)
    box(ax2, 6.0, 2.35, "VMD into $K$ modes,\nordered by centre frequency",
        w=9.6, h=0.62, fc="#e8f3e8", ec="#2f5d2f", fs=6.0)
    box(ax2, 6.0, 1.42, "chain (a) applied to every mode", w=9.6, h=0.5,
        fc="#f5f5f5", ec="#666666", ls="--", fs=6.0)
    box(ax2, 6.0, 0.5, "concatenated descriptor\n$\\mathbf{f}^{\\mathrm{multi}}$",
        w=9.6, h=0.62, fc="#e8f3e8", ec="#2f5d2f", fs=6.0)
    arrow(ax2, 6.0, 2.04, 6.0, 1.68)
    arrow(ax2, 6.0, 1.16, 6.0, 0.82)
    _save(fig, "fig1_pipeline.pdf")


# ------------------------------------------------------------------ fig 2
def fig_entropy_curve():
    """Why the unfiltered rule fails and what the saturation rule does."""
    X, y = data.load_rotor180()
    taus = np.arange(1, TAU_MAX + 1)
    idx = np.where(y == 1)[0][:3]          # three rub-impact signals
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55),
                             gridspec_kw={"wspace": 0.35})
    for n, i in enumerate(idx):
        H = psr.symbolic_entropy_curve(X[i], taus)
        info = psr.lpf_sem_from_curve(H, taus)
        axes[0].plot(taus, H, lw=0.9, alpha=0.85)
        axes[1].plot(taus, H, lw=0.9, alpha=0.45,
                     label=r"raw $H(\tau)$" if n == 0 else None)
        axes[1].plot(taus, info["Hf"], lw=1.3,
                     label="low-pass filtered" if n == 0 else None)
        axes[1].axvline(info["tau"], color="k", ls=":", lw=1.0,
                        label=r"saturation onset $\tau^\star$" if n == 0
                        else None)
        r = psr.raw_se_from_curve(H, taus)
        axes[0].axvline(r["tau"], color="k", ls="--", lw=0.9,
                        label="first local maximum" if n == 0 else None)
    axes[0].set_xlabel(r"delay $\tau$")
    axes[0].set_ylabel(r"normalised symbolic entropy $H(\tau)$")
    axes[0].set_title("(a) unfiltered curves, first local maximum", fontsize=9)
    axes[1].set_xlabel(r"delay $\tau$")
    axes[1].set_title("(b) low-pass filtered, saturation onset", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[1].legend(fontsize=7.5, loc="lower right")
    _save(fig, "fig2_entropy_curve.pdf")


# ------------------------------------------------------------------ fig 3
def fig_delay_estimation(d):
    """Delay accuracy on the five chaotic systems and on the measured signals."""
    a = d["chaos_accuracy"]
    systems = list(a.keys())
    rt = d["rotor_tau"]["per_class"]
    classes = sorted(rt.keys())
    names = ["Normal", "Rub", "Unbalance", "Misalignment"][:len(classes)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85),
                             gridspec_kw={"wspace": 0.45})
    w = 0.16
    xpos = np.arange(len(systems))
    for j, m in enumerate(METHODS):
        te = [a[s]["methods"][m]["tau_err"] for s in systems]
        de = [a[s]["methods"][m]["dim_err"] for s in systems]
        axes[0].bar(xpos + (j - 2) * w, te, w * 0.9, label=LABEL[m],
                    color=COLOR[m], edgecolor="k", linewidth=0.3)
        axes[1].bar(xpos + (j - 2) * w, de, w * 0.9,
                    color=COLOR[m], edgecolor="k", linewidth=0.3)
    for ax, t in zip(axes[:2], ["(a) relative delay error",
                                "(b) absolute dimension error"]):
        ax.set_xticks(xpos)
        ax.set_xticklabels(systems, rotation=25, ha="right", fontsize=7)
        ax.set_title(t, fontsize=8.5)
    axes[0].legend(fontsize=6, ncol=1, loc="upper right", frameon=True,
                   framealpha=0.9, edgecolor="#cccccc")
    xc = np.arange(len(classes))
    for j, m in enumerate(METHODS):
        mu = [rt[c][m]["tau_mean"] for c in classes]
        sd = [rt[c][m]["tau_std"] for c in classes]
        axes[2].bar(xc + (j - 2) * w, mu, w * 0.9, yerr=sd, capsize=1.4,
                    color=COLOR[m], edgecolor="k", linewidth=0.3)
    axes[2].set_xticks(xc)
    axes[2].set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    axes[2].set_ylabel(r"delay $\tau$ (samples)", fontsize=8)
    axes[2].set_title("(c) measured rotor signals", fontsize=8.5)
    _save(fig, "fig3_delay_estimation.pdf")


# ------------------------------------------------------------------ fig 4
def fig_robustness(d):
    """Delay accuracy against noise and against impulsive glitches."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.38),
                             gridspec_kw={"wspace": 0.45})
    cn = d["chaos_noise"]
    snrs = sorted({float(s) for v in cn.values() for s in v["snr"]})
    systems = list(cn.keys())
    for m in METHODS:
        vals = [np.mean([cn[s]["snr"][str(int(k))][m]["rel_err"]
                         for s in systems]) for k in snrs]
        axes[0].plot(snrs, vals, marker=MARK[m], ms=4, lw=1.1,
                     color=COLOR[m], ls=STYLE[m], label=LABEL[m])
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("mean relative delay error")
    axes[0].set_title("(a) chaotic systems, noise", fontsize=8.5)
    axes[0].legend(fontsize=6)
    rn = d["rotor_noise"]
    rsnr = sorted({float(s) for s in rn["snr"]})
    for m in METHODS:
        vals = [rn["snr"][str(int(k))][m]["mean_abs_dev"] for k in rsnr]
        axes[1].plot(rsnr, vals, marker=MARK[m], ms=4, lw=1.1,
                     color=COLOR[m], ls=STYLE[m])
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel(r"mean $|\Delta\tau|$ (samples)")
    axes[1].set_title("(b) measured signals, noise", fontsize=8.5)
    cs = d["chaos_spikes"]
    ssys = list(cs.keys())
    dens = [c["density"] for c in cs[ssys[0]][METHODS[0]]["curve"]]
    for m in METHODS:
        vals = [np.mean([cs[s][m]["curve"][j]["rel_err"]
                         for s in ssys]) for j in range(len(dens))]
        axes[2].plot(np.array(dens) * 100, vals, marker=MARK[m], ms=4, lw=1.1,
                     color=COLOR[m], ls=STYLE[m])
    axes[2].set_xlabel("impulse density (percent)")
    axes[2].set_ylabel("mean relative delay error")
    axes[2].set_title("(c) impulsive glitches", fontsize=8.5)
    _save(fig, "fig4_robustness.pdf")


# ------------------------------------------------------------------ fig 7
def fig_hypergraph_example():
    """One recurrence network and its NI-Louvain hypergraph, per class."""
    import topology
    X, y = data.load_rotor180()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.90),
                             gridspec_kw={"wspace": 0.35})
    for ax, c, nm in zip(axes, [0, 2], ["Normal", "Unbalance"]):
        i = np.where(y == c)[0][0]
        tau, dim, _ = psr.estimate_many(X[i], ["lpf_sem"])["lpf_sem"]
        r = topology.signal_to_hypergraph(X[i], tau, dim)
        A = r["adjacency"]
        G = nx.from_scipy_sparse_array(A)
        pos = nx.spring_layout(G, seed=1, iterations=60)
        comms = r["hypergraph"]["communities"]
        node_c = {}
        for ci, cm in enumerate(comms):
            for v in cm:
                node_c.setdefault(v, ci)
        cols = [node_c.get(n, 0) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=6,
                              node_color=cols, cmap="tab20", linewidths=0)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.3)
        ax.set_axis_off()
        ax.set_title(f"{nm}: {len(comms)} hyperedges", fontsize=9)
    _save(fig, "fig6_hypergraph_example.pdf")


# ------------------------------------------------------------------ fig 8
def fig_feature_separability(d):
    pc = d["per_class"]
    classes = sorted(pc.keys(), key=int)
    names = ["Normal", "Rub", "Unbalance", "Misalignment"][:len(classes)]
    keys = ["hedge_size_mean", "hedge_interaction", "overlap_ratio",
            "algebraic_connectivity"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.29),
                             gridspec_kw={"wspace": 0.55})
    for ax, k in zip(axes, keys):
        mu = [pc[c][k]["mean"] for c in classes]
        sd = [pc[c][k]["std"] for c in classes]
        ax.bar(range(len(classes)), mu, yerr=sd, capsize=2,
               color="#4393c3", edgecolor="k", linewidth=0.4)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_title(k.replace("_", " "), fontsize=8)
    _save(fig, "fig7_feature_separability.pdf")


# ------------------------------------------------------------------ fig 9
def fig_methods_multires(dm, d4):
    """Downstream separability per estimator and per resolution level."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.38),
                             gridspec_kw={"wspace": 0.55})
    ms = dm["methods"]
    order = [m for m in METHODS if m in ms]
    acc = [ms[m]["separability"]["nc_cv_acc"] for m in order]
    sd = [ms[m]["separability"]["nc_cv_std"] for m in order]
    fish = [ms[m]["separability"]["fisher_mean"] for m in order]
    axes[0].bar(range(len(order)), acc, yerr=sd, capsize=2,
                color=[COLOR[m] for m in order], edgecolor="k",
                linewidth=0.4)
    axes[1].bar(range(len(order)), fish, color=[COLOR[m] for m in order],
                edgecolor="k", linewidth=0.4)
    for ax, t in zip(axes[:2], ["(a) CV accuracy", "(b) Fisher ratio"]):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([LABEL[m].split(" (")[0] for m in order],
                           rotation=30, ha="right", fontsize=6.5)
        ax.set_title(t, fontsize=8.5)
    axes[0].set_ylabel("nearest-centroid CV accuracy")
    axes[1].set_ylabel("mean Fisher ratio")
    lv = d4["levels"]
    ks = sorted(lv.keys(), key=int)
    acc4 = [lv[k]["separability"]["nc_cv_acc"] for k in ks]
    inter = [lv[k]["inter_mean"] for k in ks]
    comb = d4["combined"]["separability"]["nc_cv_acc"]
    axes[2].bar(range(len(ks)), acc4, color="#4393c3", edgecolor="k",
                linewidth=0.4, label="per IMF")
    axes[2].axhline(comb, color="#b2182b", ls="--", lw=1.2,
                    label="concatenated")
    axes[2].set_xticks(range(len(ks)))
    axes[2].set_xticklabels([f"IMF {int(k)+1}" for k in ks], fontsize=7)
    axes[2].set_ylabel("nearest-centroid CV accuracy")
    axes[2].set_title("(c) per resolution level", fontsize=8.5)
    axes[2].legend(fontsize=6.5)
    axes[3].bar(range(len(ks)), inter, color="#762a83", edgecolor="k",
                linewidth=0.4)
    axes[3].set_xticks(range(len(ks)))
    axes[3].set_xticklabels([f"IMF {int(k)+1}" for k in ks], fontsize=7)
    axes[3].set_ylabel("mean hyperedge interaction")
    axes[3].set_title("(d) higher-order coupling", fontsize=8.5)
    _save(fig, "fig8_methods_multires.pdf")


# ------------------------------------------------------------------ fig 11
def fig_sensitivity(d):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.20),
                             gridspec_kw={"wspace": 0.55})
    f = d["filter"]
    for ax, key, xlab in [
            (axes[0], "orders", "Butterworth order"),
            (axes[1], "cutoffs", "normalised cutoff"),
            (axes[2], "alphas", "threshold fraction alpha")]:
        ks = sorted(f[key].keys(), key=float)
        mu = [f[key][k]["tau_mean"] for k in ks]
        sd = [f[key][k]["tau_std"] for k in ks]
        ax.errorbar([float(k) for k in ks], mu, yerr=sd, marker="o", ms=4,
                    lw=1.1, capsize=2, color="#b2182b")
        ax.set_xlabel(xlab)
    axes[0].set_ylabel(r"estimated delay $\tau$")
    c = d["community"]["remove_ratio"]
    ks = sorted(c.keys(), key=float)
    axes[3].plot([float(k) for k in ks], [c[k]["nc_cv_acc"] for k in ks],
                 marker="s", ms=4, lw=1.1, color="#2166ac")
    axes[3].set_xlabel("edge-removal ratio")
    axes[3].set_ylabel("CV accuracy")
    for ax, t in zip(axes, ["(a) filter order", "(b) cutoff", "(c) alpha",
                            "(d) edge removal"]):
        ax.set_title(t, fontsize=8)
    _save(fig, "fig5_sensitivity.pdf")


# ------------------------------------------------------------------ fig 12
def fig_ablation(d):
    v = d["variants"]
    order = ["full", "ablation_A_psr_cc", "ablation_B_louvain"]
    order = [k for k in order if k in v]
    acc = [v[k]["separability"]["nc_cv_acc"] for k in order]
    sd = [v[k]["separability"]["nc_cv_std"] for k in order]
    fish = [v[k]["separability"]["fisher_mean"] for k in order]
    over = [v[k]["overlap_ratio_mean"] for k in order]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.29),
                             gridspec_kw={"wspace": 0.45})
    for ax, vals, err, ylab, t in [
            (axes[0], acc, sd, "CV accuracy", "(a) separability"),
            (axes[1], fish, None, "Fisher ratio", "(b) variance ratio"),
            (axes[2], over, None, "overlap ratio", "(c) overlap")]:
        ax.bar(range(len(order)), vals, yerr=err, capsize=2,
               color=["#b2182b", "#4393c3", "#4d9221"][:len(order)],
               edgecolor="k", linewidth=0.4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["full", "A: C-C", "B: Louvain"][:len(order)])
        ax.set_ylabel(ylab)
        ax.set_title(t, fontsize=9)
    _save(fig, "fig9_ablation.pdf")


# ------------------------------------------------------------------ fig 13
def fig_complexity(d):
    stages = d["stages_mean_over_5"]
    keys = ["symbolic_entropy_curve", "lpf_smooth_and_derivative",
            "mutual_information", "autocorrelation", "cc_method", "cao_fnn",
            "recurrence_network", "ni_louvain", "hypergraph_features",
            "vmd_K3"]
    keys = [k for k in keys if k in stages]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.46),
                             gridspec_kw={"wspace": 0.35})
    axes[0].barh(range(len(keys)), [stages[k] for k in keys],
                 color="#4393c3", edgecolor="k", linewidth=0.4)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([k.replace("_", " ") for k in keys], fontsize=7)
    axes[0].set_xlabel("mean wall-clock time (s)")
    axes[0].set_title("(a) cost per stage", fontsize=9)
    sc = d["scaling"]
    ns = sorted(sc.keys(), key=int)
    axes[1].plot([int(k) for k in ns],
                 [sc[k]["symbolic_entropy_curve"]["mean_s"] for k in ns],
                 marker="o", ms=4, lw=1.1, label="symbolic entropy",
                 color="#b2182b")
    axes[1].plot([int(k) for k in ns],
                 [sc[k]["recurrence_network"]["mean_s"] for k in ns],
                 marker="s", ms=4, lw=1.1, label="recurrence network",
                 color="#2166ac")
    axes[1].set_xlabel("working length N")
    axes[1].set_ylabel("time (s)")
    axes[1].set_title("(b) scaling of the dominant stages", fontsize=9)
    axes[1].legend(fontsize=7)
    _save(fig, "fig10_complexity.pdf")


# ------------------------------------------------------------------ fig 11
HG_PIPES = ["hypergraph, multi-resolution", "hypergraph, saturated delay",
            "hypergraph, entropy delay", "hypergraph, MI delay"]


def fig_classification(dr, dm):
    """Best cross-validated accuracy per feature pipeline, two datasets.

    The hypergraph pipelines are drawn in the accent colour so that the
    reader can see at a glance where the proposed representation sits among
    the mainstream ones.  The condition is the amplitude-normalised one, so
    the bars compare waveform shape rather than record level.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.80), sharex=True,
                             gridspec_kw={"wspace": 0.70})
    for ax, d, tag in ((axes[0], dr, "rotor"), (axes[1], dm, "MFPT")):
        if not d or "normalised" not in d["conditions"]:
            ax.set_visible(False)
            continue
        blk = d["conditions"]["normalised"]["pipelines"]
        names = list(blk.keys())
        vals = [blk[n]["best_acc"] for n in names]
        order = np.argsort(vals)
        names = [names[i] for i in order]
        vals = [vals[i] for i in order]
        cols = ["#b2182b" if n in HG_PIPES else "#9ecae1" for n in names]
        ax.barh(range(len(names)), vals, color=cols, edgecolor="k",
                linewidth=0.4)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace(", ", ",\n") for n in names], fontsize=6.5)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("best CV accuracy")
        ax.set_title(f"({'ab'[tag == 'MFPT']}) {tag}", fontsize=9)
    _save(fig, "fig11_classification.pdf")


# ------------------------------------------------------------------ fig 12
def fig_propagation(d):
    """Whether delay stability reaches the descriptor.

    (a) Hypergraph separability against the noise level for each delay rule.
    (b) Dispersion of the estimated delay on the same records, averaged over
    ten noise realisations, which is the stability the separability is
    supposed to inherit.
    """
    if not d or "methods" not in d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.50),
                             gridspec_kw={"wspace": 0.35})
    order = [k for k in ("clean", "15", "10", "5")
             if all(k in d["methods"][m] for m in d["methods"])]
    xs = np.arange(len(order))
    for m in d["methods"]:
        if m not in COLOR:
            continue
        ys = [d["methods"][m][k]["nc_cv_acc"] for k in order]
        axes[0].plot(xs, ys, marker=MARK[m], color=COLOR[m], lw=1.1, ms=4,
                     label=LABEL[m])
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["clean"] + [f"{k} dB" for k in order[1:]])
    axes[0].set_ylabel("hypergraph separability")
    axes[0].set_title("(a) descriptor separability", fontsize=9)
    axes[0].legend(fontsize=6.5)

    dd = d.get("delay_dispersion", {}).get("conditions", {})
    dorder = [k for k in ("clean", "20", "15", "10", "5", "0") if k in dd]
    dxs = np.arange(len(dorder))
    for m in d["methods"]:
        if m not in dd.get(dorder[0], {}):
            continue
        ys = [dd[k][m]["tau_dispersion"] for k in dorder]
        axes[1].plot(dxs, ys, marker=MARK[m], color=COLOR[m], lw=1.1, ms=4)
    axes[1].set_xticks(dxs)
    axes[1].set_xticklabels(["clean"] + [f"{k} dB" for k in dorder[1:]])
    axes[1].set_ylabel(r"within-class delay std")
    axes[1].set_title("(b) delay dispersion", fontsize=9)
    _save(fig, "fig12_propagation.pdf")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("figures ...")
    fig_pipeline()
    fig_entropy_curve()
    d1 = _load("exp1.json")
    if d1:
        fig_delay_estimation(d1)
        fig_robustness(d1)
    d3 = _load("exp3_full.json")
    if d3:
        fig_feature_separability(d3)
    dm = _load("exp3_methods.json")
    d4 = _load("exp4_multires.json")
    if dm and d4:
        fig_methods_multires(dm, d4)
    d2 = _load("exp2.json")
    if d2:
        fig_sensitivity(d2)
    d5 = _load("exp5.json")
    if d5:
        fig_ablation(d5)
    d6 = _load("exp6.json")
    if d6:
        fig_complexity(d6)
    c7r = _load("exp7_rotor.json")
    c7m = _load("exp7_mfpt.json")
    if c7r or c7m:
        fig_classification(c7r, c7m)
    p8 = _load("exp8_rotor.json")
    if p8:
        fig_propagation(p8)
    fig_hypergraph_example()
    print("figures done")


if __name__ == "__main__":
    main()
