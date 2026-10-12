"""Format every table of the manuscript as a LaTeX fragment.

Each function writes tables/<name>.tex, which the manuscript pulls in with
\\input.  Numbers come only from the results JSON, never typed by hand.
"""
import json
import numpy as np

from config import TABLES, RESULTS, FS, N_POINTS

METHODS = ["lpf_sem", "raw_se", "mi", "acf", "cc"]
MLABEL = {"lpf_sem": "saturated SE", "raw_se": "SE", "mi": "MI", "acf": "ACF",
          "cc": "C-C"}
CLASSES = ["Normal", "Rub", "Unbalance", "Misalignment"]
FEATURE_ROWS = [
    ("n_hyperedges", "number of hyperedges"),
    ("hedge_size_mean", "mean hyperedge size"),
    ("hedge_size_std", "std. hyperedge size"),
    ("hedge_size_entropy", "hyperedge size entropy"),
    ("node_degree_mean", "mean node degree"),
    ("overlap_ratio", "overlap ratio"),
    ("hedge_interaction", "hyperedge interaction"),
    ("giant_fraction", "giant component fraction"),
    ("algebraic_connectivity", "algebraic connectivity"),
    ("spectral_radius", "spectral radius"),
    ("spectral_gap", "spectral gap"),
    ("spectral_entropy", "spectral entropy"),
]


def _load(name):
    p = RESULTS / name
    if not p.exists():
        print("  missing", name)
        return None
    return json.loads(p.read_text())


def _w(name, text):
    (TABLES / name).write_text(text)
    print("  wrote", name)


def _fmt(v, d=3):
    if v is None:
        return "--"
    return f"{v:.{d}f}"


def _sci(v, d=2):
    if v is None or v == 0:
        return "--"
    e = int(np.floor(np.log10(abs(v))))
    return f"{v / 10 ** e:.{d}f}$\\times10^{{{e}}}$"


# ------------------------------------------------------------------ setup
def tab_setup():
    from config import MFPT_FS, N_MFPT_POINTS, N_MFPT_PER_CLASS
    rows = [
        ("measured sets", f"rotor at {FS:.0f} Hz, MFPT bearing at {MFPT_FS:.0f} Hz"),
        ("signal length", f"{N_POINTS} samples ({N_MFPT_POINTS} for MFPT)"),
        ("classes", "4 rotor states, 3 bearing states"),
        ("samples per class", "45, both sets"),
        ("symbols $p$", "6"),
        ("amplitude threshold $e$", r"$1.2\,\sigma_x$"),
        ("delay search range", r"$1 \ldots 150$"),
        ("Butterworth order", "3"),
        ("Butterworth cutoff", "0.1"),
        ("threshold fraction $\\alpha$", "0.10"),
        ("recurrence rate", "5th percentile"),
        ("edge-removal ratio", "0.75"),
        ("VMD modes $K$", "3"),
        ("VMD penalty $\\alpha_{\\mathrm{VMD}}$", "2000"),
        ("random seed", "20250929"),
    ]
    body = " \\\\\n".join(f"{k} & {v}" for k, v in rows)
    tex = ("\\begin{tabular}{ll}\n\\toprule\n"
           "setting & value \\\\\n\\midrule\n"
           f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    _w("tab_setup.tex", tex)


# ------------------------------------------------------------------ chaos
def tab_chaos(d):
    a = d["chaos_accuracy"]
    systems = list(a.keys())
    cols = ["system", "$\\tau_{\\rm ref}$", "$m_{\\rm ref}$"]
    for m in METHODS:
        cols += [f"$\\tau$", f"$m$"]
    lines = []
    for s in systems:
        r = a[s]
        cells = [s, str(r["ref_tau"]), str(r["ref_dim"])]
        for m in METHODS:
            mm = r["methods"][m]
            cells.append(str(mm["tau"]))
            cells.append(str(mm["dim"]))
        lines.append(" & ".join(cells) + " \\\\")
    tex = ("\\begin{tabular}{l" + "cc" * (1 + len(METHODS)) + "}\n\\toprule\n"
           + " & ".join(cols) + " \\\\\n"
           + "\\cmidrule(lr){2-3}" + "".join(
               f"\\cmidrule(lr){{{4 + 2 * j}-{5 + 2 * j}}}" for j in range(len(METHODS)))
           + "\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_chaos_est.tex", tex)

    lines = []
    for s in systems:
        r = a[s]["methods"]
        te = np.mean([r[m]["tau_err"] for m in METHODS])
        de = np.mean([r[m]["dim_err"] for m in METHODS])
        cells = [s]
        for m in METHODS:
            cells.append(_fmt(r[m]["tau_err"], 3))
        for m in METHODS:
            cells.append(str(r[m]["dim_err"]))
        lines.append(" & ".join(cells) + " \\\\")
    tex = ("\\begin{tabular}{l" + "c" * (2 * len(METHODS)) + "}\n\\toprule\n"
           + " & ".join(["system"] + [f"{MLABEL[m]}" for m in METHODS] * 2)
           + " \\\\\n\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\n"
           % (1 + len(METHODS), 2 + len(METHODS), 1 + 2 * len(METHODS))
           + " & ".join([""] + ["$\\tau$ err"] * len(METHODS)
                        + ["$m$ err"] * len(METHODS)) + " \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_chaos_err.tex", tex)


def tab_rotor_tau(d):
    """Delay per state for the four selectors discussed in the text, plus the
    embedding dimension, which is shared because Cao's criterion is used for
    every method.  The ACF selector is omitted here to keep the table within
    the text block; its delay error is reported in the chaotic table instead."""
    rt = d["rotor_tau"]
    methods = ["lpf_sem", "raw_se", "mi", "cc"]
    nc = len(CLASSES)
    lines = []
    for c in range(nc):
        cells = [CLASSES[c]]
        for m in methods:
            v = rt["per_class"][str(c)][m]
            cells.append(f"{v['tau_mean']:.1f} $\\pm$ {v['tau_std']:.1f}")
        for m in methods:
            v = rt["per_class"][str(c)][m]
            cells.append(f"{v['dim_mean']:.0f}")
        lines.append(" & ".join(cells) + " \\\\")
    tex = ("\\begin{tabular}{l" + "c" * (2 * len(methods)) + "}\n\\toprule\n"
           + " & ".join(["state"] + [MLABEL[m] for m in methods] * 2)
           + " \\\\\n\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\n"
           % (1 + len(methods), 2 + len(methods), 1 + 2 * len(methods))
           + " & ".join([""] + ["$\\tau$"] * len(methods)
                        + ["$m$"] * len(methods)) + " \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_rotor_tau.tex", tex)


# ------------------------------------------------------------------ features
def tab_features(d):
    pc = d["per_class"]
    keys = [k for k, _ in FEATURE_ROWS]
    lines = []
    for k, label in FEATURE_ROWS:
        cells = [label]
        for c in range(len(CLASSES)):
            v = pc[str(c)][k]
            cells.append(f"{v['mean']:.3f} $\\pm$ {v['std']:.3f}")
        lines.append(" & ".join(cells) + " \\\\")
    tex = ("\\begin{tabular}{lcccc}\n\\toprule\n"
           "feature & Normal & Rub & Unbalance & Misalignment \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_features.tex", tex)


def tab_features_p(d):
    """Kruskal-Wallis table for the same features, as a compact list."""
    lines = []
    for r in d["separability"]["kw"]:
        lines.append(f"{r['feature'].replace('_', ' ')} & {r['H']:.1f} & "
                     f"{_sci(r['p'], 1)} \\\\")
    tex = ("\\begin{tabular}{lcc}\n\\toprule\nfeature & $H$ & $p$ \\\\\n"
           "\\midrule\n" + "\n".join(lines)
           + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_features_kw.tex", tex)


# ------------------------------------------------------------------ methods
def tab_methods(d):
    ms = d["methods"]
    order = [m for m in METHODS if m in ms]
    lines = []
    for m in order:
        s = ms[m]["separability"]
        lines.append(" & ".join([
            MLABEL[m], _fmt(s["nc_cv_acc"], 3),
            _fmt(s["silhouette"], 3), _fmt(s["fisher_mean"], 3),
            _fmt(ms[m]["tau_mean"], 1), _fmt(ms[m]["tau_std"], 1),
            _fmt(ms[m]["hedge_mean"], 1)]) + " \\\\")
    tex = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
           "estimator & CV accuracy & silhouette & Fisher & mean $\\tau$ & "
           "std $\\tau$ & hyperedges \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_methods.tex", tex)


def tab_multires(d):
    lv = d["levels"]
    ks = sorted(lv.keys(), key=int)
    lines = []
    for k in ks:
        v = lv[k]
        s = v["separability"]
        lines.append(" & ".join([
            f"IMF {int(k) + 1}", _fmt(s["nc_cv_acc"], 3),
            _fmt(s["silhouette"], 3), _fmt(s["fisher_mean"], 3),
            _fmt(v["hedge_mean"], 1), _fmt(v["size_mean"], 1),
            _fmt(v["inter_mean"], 3), _fmt(v["gc_mean"], 3)]) + " \\\\")
    s = d["combined"]["separability"]
    lines.append(" & ".join([
        "concatenated", _fmt(s["nc_cv_acc"], 3), _fmt(s["silhouette"], 3),
        _fmt(s["fisher_mean"], 3), "--", "--", "--", "--"]) + " \\\\")
    tex = ("\\begin{tabular}{lccccccc}\n\\toprule\n"
           "level & CV accuracy & silhouette & Fisher & hyperedges & mean size "
           "& interaction & giant fraction \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_multires.tex", tex)


def tab_coupled(d):
    s8 = d["single_resolution"]["separability_8class"]
    s4 = d["measured_only"]["separability_4class"]
    lines = [
        " & ".join(["four measured states", _fmt(s4["nc_cv_acc"], 3),
                    _fmt(s4["silhouette"], 3), _fmt(s4["fisher_mean"], 3)])
        + " \\\\",
        " & ".join(["eight states (four simulated)", _fmt(s8["nc_cv_acc"], 3),
                    _fmt(s8["silhouette"], 3), _fmt(s8["fisher_mean"], 3)])
        + " \\\\",
    ]
    tex = ("\\begin{tabular}{lccc}\n\\toprule\n"
           "setting & CV accuracy & silhouette & Fisher \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_coupled.tex", tex)


# ------------------------------------------------------------------ sensitivity
def tab_sensitivity(d):
    f = d["filter"]
    frows = []
    for key, lab in [("orders", "filter order"), ("cutoffs", "cutoff"),
                     ("alphas", "$\\alpha$")]:
        ks = sorted(f[key].keys(), key=float)
        frows.append((lab, [f"{f[key][k]['tau_mean']:.1f}" for k in ks]))
    width = max(len(v) for _, v in frows)

    def _row(lab, vals, fmt=None):
        cells = [lab] + [fmt(v) if fmt else v for v in vals]
        cells += [""] * (width - len(vals))
        return " & ".join(cells) + " \\\\"

    c = d["community"]
    rr = sorted(c["remove_ratio"].keys(), key=float)
    pp = sorted(c["recurrence_percentile"].keys(), key=float)
    kk = sorted(c["vmd_K"].keys(), key=int)
    body = [_row(lab, vals) for lab, vals in frows]
    body.append("\\midrule")
    body.append(_row("removal ratio", rr, lambda k: k))
    body.append("\\midrule")
    body.append(_row("CV acc. (removal)",
                     [c["remove_ratio"][k]["nc_cv_acc"] for k in rr],
                     lambda v: f"{v:.2f}"))
    body.append("\\midrule")
    body.append(_row("recurrence pct.", pp, lambda k: k))
    body.append(_row("CV acc. (recurrence)",
                     [c["recurrence_percentile"][k]["nc_cv_acc"] for k in pp],
                     lambda v: f"{v:.2f}"))
    body.append("\\midrule")
    body.append(_row("VMD $K$", kk, lambda k: k))
    body.append(_row("CV acc. (VMD)",
                     [c["vmd_K"][k]["nc_cv_acc"] for k in kk],
                     lambda v: f"{v:.2f}"))
    tex = ("\\begin{tabular}{l" + "c" * width + "}\n\\toprule\n"
           "setting & " + " & ".join(["value"] * width) + " \\\\\n\\midrule\n"
           + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_sensitivity.tex", tex)


# ------------------------------------------------------------------ ablation
def tab_ablation(d):
    v = d["variants"]
    order = ["full", "ablation_A_psr_cc", "ablation_B_louvain"]
    nm = {"full": "full pipeline", "ablation_A_psr_cc": "A: C-C instead of saturated SE",
          "ablation_B_louvain": "B: Louvain instead of NI-Louvain"}
    lines = []
    for k in order:
        if k not in v:
            continue
        s = v[k]["separability"]
        lines.append(" & ".join([
            nm[k], _fmt(s["nc_cv_acc"], 3), _fmt(s["silhouette"], 3),
            _fmt(s["fisher_mean"], 3), _fmt(v[k]["hedge_mean"], 1),
            _fmt(v[k]["hedge_size_mean"], 1),
            _fmt(v[k]["overlap_ratio_mean"], 3),
            _fmt(v[k]["interaction_mean"], 3)]) + " \\\\")
    tex = ("\\begin{tabular}{lccccccc}\n\\toprule\n"
           "variant & CV accuracy & silhouette & Fisher & hyperedges & mean size "
           "& overlap & interaction \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_ablation.tex", tex)


# ------------------------------------------------------------------ complexity
def tab_complexity(d):
    st = d["stages_mean_over_5"]
    order = ["symbolic_entropy_curve", "lpf_smooth_and_derivative",
             "mutual_information", "autocorrelation", "cc_method", "cao_fnn",
             "recurrence_network", "ni_louvain", "hypergraph_features",
             "vmd_K3"]
    lines = [f"{k.replace('_', ' ')} & {st[k]:.3f} \\\\" for k in order
             if k in st]
    if "ni_louvain_reference" in st:
        lines.append("\\midrule")
        lines.append(f"overlapping communities, reference & "
                     f"{st['ni_louvain_reference']:.3f} \\\\")
        lines.append(f"overlapping communities, optimised & "
                     f"{st['ni_louvain']:.3f} \\\\")
        lines.append(f"speed-up & "
                     f"{st['ni_louvain_reference'] / max(st['ni_louvain'], 1e-9):.0f}$\\times$ \\\\")
    tex = ("\\begin{tabular}{lc}\n\\toprule\nstage & mean time (s) \\\\\n"
           "\\midrule\n" + "\n".join(lines)
           + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_complexity.tex", tex)

    sc = d["scaling"]
    ns = sorted(sc.keys(), key=int)
    lines = []
    for n in ns:
        lines.append(" & ".join([
            n, str(sc[n]["n_nodes"]), str(sc[n]["n_edges"]),
            f"{sc[n]['symbolic_entropy_curve']['mean_s']:.3f}",
            f"{sc[n]['recurrence_network']['mean_s']:.3f}"]) + " \\\\")
    tex = ("\\begin{tabular}{ccccc}\n\\toprule\n$N$ & nodes & edges & "
           "entropy time (s) & recurrence time (s) \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_scaling.tex", tex)


# ------------------------------------------------------------------ new studies
PIPELINES = ["time-domain statistics", "spectral band energies",
             "wavelet-packet energies", "VMD mode entropies",
             "multiscale perm. entropy", "recurrence quantification",
             "hypergraph, multi-resolution", "hypergraph, saturated delay",
             "hypergraph, entropy delay", "hypergraph, MI delay"]


def tab_classification(dr, dm):
    """Cross-validated accuracy of every pipeline on both datasets.

    The normalised condition is the fair comparison: both sets contain a real
    amplitude difference between classes, which an amplitude-sensitive
    descriptor can exploit without describing the waveform.
    """
    order = list(PIPELINES)
    clfs = ["SVM (RBF)", "kNN (k=5)", "random forest"]
    HG_FIRST = "hypergraph, multi-resolution"

    def _block(d, cond):
        if not d or cond not in d["conditions"]:
            return None
        return d["conditions"][cond]["pipelines"]

    blocks = [("rotor", _block(dr, "normalised")),
              ("MFPT", _block(dm, "normalised"))]
    cols = [b for b in blocks if b[1] is not None]
    SHORT = {"hypergraph, multi-resolution": "hypergraph, multi-res",
             "hypergraph, saturated delay": "hypergraph, saturated",
             "hypergraph, entropy delay": "hypergraph, entropy",
             "hypergraph, MI delay": "hypergraph, MI"}
    lines = []
    for name in order:
        if name == HG_FIRST:
            lines.append("\\midrule")
        cells = [SHORT.get(name, name).replace("%", "\\%")]
        for _tag, blk in cols:
            if name not in blk:
                cells += ["--"] * len(clfs)
                continue
            cv = blk[name]["classifiers"]
            cells += [f"{cv[c]['acc']:.3f}" for c in clfs]
        lines.append(" & ".join(cells) + " \\\\")
    short = {"SVM (RBF)": "SVM", "kNN (k=5)": "kNN",
             "random forest": "RF"}
    head = ["pipeline"]
    for tag, _ in cols:
        head += [f"{tag} {short[c]}" for c in clfs]
    tex = ("\\begin{tabular}{l" + "c" * (len(clfs) * len(cols)) + "}\n\\toprule\n"
           + " & ".join(head) + " \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_classification.tex", tex)


def tab_propagation(d, d2=None):
    """Separability and delay dispersion against noise, per delay rule."""
    if not d:
        return
    methods = [m for m in METHODS[:5] if m in d["methods"]]
    keys = d["snrs"]
    lines = []
    for m in methods:
        r = d["methods"][m]
        cells = [MLABEL[m]]
        for k in keys:
            kk = "clean" if k == "clean" else f"{float(k):g}"
            v = r.get(kk)
            cells.append(f"{v['nc_cv_acc']:.2f}" if v else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{l}{delay dispersion (samples)}\\\\"
                 % (len(keys) + 1))
    for m in methods:
        r = d["methods"][m]
        cells = [MLABEL[m]]
        for k in keys:
            kk = "clean" if k == "clean" else f"{float(k):g}"
            v = r.get(kk)
            cells.append(f"{v['tau_dispersion']:.2f}" if v else "--")
        lines.append(" & ".join(cells) + " \\\\")
    head = ["method"] + ["clean" if k == "clean" else f"{float(k):g} dB"
                         for k in keys]
    tex = ("\\begin{tabular}{l" + "c" * len(keys) + "}\n\\toprule\n"
           + " & ".join(head) + " \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_propagation.tex", tex)


# ------------------------------------------------------------------ estimator variants (Experiment 9)
VARIANTS = ["lpf_sem", "raw_se", "lpf_sem_d2", "lpf_curv", "lpf_fit",
            "lpf_adaptive", "pe_sem", "mi", "acf", "cc"]
VARLABEL = {"lpf_sem": "sat. SE ($\\alpha$ thr.)",
            "raw_se": "SE (first max)",
            "lpf_sem_d2": "SE (|$H''$| max)",
            "lpf_curv": "SE (curvature max)",
            "lpf_fit": "SE (exp. fit, $\\tau_c$)",
            "lpf_adaptive": "SE (adaptive $\\alpha$)",
            "pe_sem": "perm. entropy (sat.)",
            "mi": "MI", "acf": "ACF", "cc": "C-C"}


def tab_variants(d):
    """Review Path-1 estimator comparison from results/exp9_variants.json.

    Estimator-level columns only: the clean benchmark delay error, the
    moderate-noise delay error on the chaotic systems, the rotor delay std
    across records, and the between/within stability ratio.  The downstream
    separability of the delay rules is not tabulated here because a single
    small-sample pass is high-variance; it is handled in the text with the
    exp8 protocol numbers.
    """
    ca = d["chaos_accuracy"]
    cn = d["chaos_noise"]
    rt = d["rotor_tau"]["per_method"]
    systems = list(ca.keys())
    lines = []
    for m in VARIANTS:
        cvals = [ca[s]["methods"][m]["tau_err"] for s in systems]
        chaosc = float(np.mean(cvals))
        nvals = [cn[s]["snr"]["10"][m]["rel_err"] for s in cn]
        chaoscn = float(np.mean(nvals))
        rstd = rt[m]["tau_std"]
        ratio = rt[m]["stability_ratio"]
        lines.append(" & ".join([
            VARLABEL[m], _fmt(chaosc, 3), _fmt(chaoscn, 3),
            _fmt(rstd, 2), _fmt(ratio, 1)]) + " \\\\")
    tex = ("\\begin{tabular}{lcccc}\n\\toprule\n"
           "estimator & chaos $\\tau$ err & noise $\\tau$ err & rotor $\\tau$ "
           "std & stabi. ratio \\\\\n"
           " & (clean) & (10 dB) & (records) & (bet./within) \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_variants.tex", tex)


# ------------------------------------------------------------------ delay scale vs descriptor (Experiments 10-12)
DELAYSCALE = ["lpf_sem", "raw_se", "lpf_fit", "lpf_curv", "lpf_adaptive",
              "pe_sem", "mi"]


def tab_delayscale(d11, d12, d10):
    """Downstream separability of the delay scale, from the full-180 set.

    exp11 gives the clean separability and the delay on every measured record;
    exp12 the same after per-record amplitude normalisation (removing the
    amplitude channel the paper's Experiment 7 guards against); exp10 the
    separability and delay dispersion under 10 dB additive noise (n-per-class
    = 12, three seeds).
    """
    lines = []
    for m in DELAYSCALE:
        tau = d11["methods"][m]["tau_mean"] if d11 and m in d11["methods"] else None
        acc = d11["methods"][m]["nc_cv_acc"] if d11 and m in d11["methods"] else None
        norm = d12["methods"][m]["nc_cv_acc"] if d12 and m in d12["methods"] else None
        ten = d10["methods"][m]["10"]["nc_cv_acc"] if d10 and m in d10["methods"] and "10" in d10["methods"][m] else None
        lines.append(" & ".join([
            VARLABEL[m], _fmt(tau, 1), _fmt(acc, 3), _fmt(norm, 3),
            _fmt(ten, 3)]) + " \\\\")
    tex = ("\\begin{tabular}{lcccc}\n\\toprule\n"
           "delay rule & delay & sep. & sep. (norm.) & sep. (10 dB) \\\\\n"
           " & $\\tau$ & (clean) & (ampl.-free) & \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_delayscale.tex", tex)


# ------------------------------------------------------------------ extreme noise (Experiment 13)
def tab_noise_sweep(d13, d10):
    """How the delay rules behave under extreme additive noise.

    `dev` is the mean absolute rotor delay deviation from the clean delay
    (samples); `sep` is the hypergraph separability at n-per-class = 12, three
    seeds.  The point is that the derivative-threshold rule collapses at -5 dB
    while the non-derivative exponential-fit reading of the *same* saturated
    family does not, and that the clean separability advantage of the fit
    reading holds up at 10 dB before every rule degrades together below.
    """
    dd = d13["delay_dev"]["snr"]
    dn = d13["downstream"]["methods"]
    lines = []
    for m in DELAYSCALE:
        dev0 = dd["0"][m]["mean_abs_dev"] if m in dd.get("0", {}) else None
        devn5 = dd["-5"][m]["mean_abs_dev"] if m in dd.get("-5", {}) else None
        ten = d10["methods"][m]["10"]["nc_cv_acc"] if d10 and m in d10["methods"] and "10" in d10["methods"][m] else None
        zero = dn[m]["0"]["nc_cv_acc"] if m in dn and "0" in dn[m] else None
        n5 = dn[m]["-5"]["nc_cv_acc"] if m in dn and "-5" in dn[m] else None
        lines.append(" & ".join([
            VARLABEL[m], _fmt(dev0, 2), _fmt(devn5, 1),
            _fmt(ten, 3), _fmt(zero, 3), _fmt(n5, 3)]) + " \\\\")
    tex = ("\\begin{tabular}{lccccc}\n\\toprule\n"
           "delay rule & dev 0 dB & dev $-5$ dB & sep 10 dB & sep 0 dB & sep $-5$ dB \\\\\n"
           " & (samples) & (samples) & & & \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_noise_sweep.tex", tex)


def tab_fixedtau(d14):
    """Separability as a function of ONE global delay (Experiment 14).

    The embedding dimension is still chosen per record; only the delay is
    forced to a common value.  This de-collinearises the delay scale from the
    separability: a fixed small delay already reaches the peak, so the high
    separability belongs to the small-delay window, not to the per-record
    adaptive tau_c.
    """
    if not d14:
        return
    ft = d14["fixed_tau"]
    order = ["1", "2", "3", "4", "5", "6", "8", "10", "12", "15"]
    lines = []
    for k in order:
        if k in ft:
            lines.append(" & ".join([k, _fmt(ft[k]["nc_cv_acc"], 3),
                                     _fmt(ft[k]["fisher_mean"], 3)]) + " \\\\")
    ad = d14["adaptive_tauc"]
    lines.append(f"adaptive $\\tau_c$={ad['tau_mean']:.1f} & "
                 f"{ad['nc_cv_acc']:.3f} & {ad['fisher_mean']:.3f} \\\\")
    tex = ("\\begin{tabular}{lcc}\n\\toprule\n"
           "global delay $\\tau$ & sep. (acc) & Fisher ratio \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    _w("tab_fixedtau.tex", tex)


ALL_TABLES = [
    "tab_setup.tex", "tab_chaos_est.tex", "tab_chaos_err.tex",
    "tab_rotor_tau.tex", "tab_features.tex", "tab_features_kw.tex",
    "tab_methods.tex", "tab_multires.tex", "tab_coupled.tex",
    "tab_sensitivity.tex", "tab_ablation.tex", "tab_complexity.tex",
    "tab_scaling.tex", "tab_classification.tex", "tab_propagation.tex",
    "tab_variants.tex", "tab_delayscale.tex", "tab_noise_sweep.tex",
    "tab_fixedtau.tex",
]


def _placeholders():
    """So the manuscript compiles before every result exists."""
    for name in ALL_TABLES:
        p = TABLES / name
        if p.exists():
            continue
        p.write_text("\\begin{tabular}{lc}\n\\toprule\n"
                     "result & value \\\\\n\\midrule\n"
                     "pending & -- \\\\\n\\bottomrule\n\\end{tabular}\n")
        print("  placeholder", name)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print("tables ...")
    tab_setup()
    d1 = _load("exp1.json")
    if d1:
        tab_chaos(d1)
        tab_rotor_tau(d1)
    d3 = _load("exp3_full.json")
    if d3:
        tab_features(d3)
        tab_features_p(d3)
    dm = _load("exp3_methods.json")
    if dm:
        tab_methods(dm)
    d4 = _load("exp4_multires.json")
    if d4:
        tab_multires(d4)
    d4c = _load("exp4_coupled.json")
    if d4c:
        tab_coupled(d4c)
    d2 = _load("exp2.json")
    if d2:
        tab_sensitivity(d2)
    d5 = _load("exp5.json")
    if d5:
        tab_ablation(d5)
    d6 = _load("exp6.json")
    if d6:
        tab_complexity(d6)
    dr7 = _load("exp7_rotor.json")
    dm7 = _load("exp7_mfpt.json")
    if dr7 or dm7:
        tab_classification(dr7, dm7)
    d8 = _load("exp8_rotor.json")
    if d8:
        tab_propagation(d8)
    d9 = _load("exp9_variants.json")
    if d9:
        tab_variants(d9)
    d11 = _load("exp11_fullsep.json")
    d12 = _load("exp12_normsep.json")
    d10 = _load("exp10_downstream.json")
    if d11 or d12 or d10:
        tab_delayscale(d11, d12, d10)
    d13 = _load("exp13_extreme.json")
    if d13:
        tab_noise_sweep(d13, d10)
    d14 = _load("exp14_fixedtau.json")
    if d14:
        tab_fixedtau(d14)
    _placeholders()
    print("tables done")


if __name__ == "__main__":
    main()
