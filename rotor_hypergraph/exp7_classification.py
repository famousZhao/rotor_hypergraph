"""Experiment 7: classification comparison against mainstream fault-diagnosis
feature pipelines.

The claim of the paper is about the feature representation, not about a new
classifier, so a single fixed classifier backbone is applied to every feature
set and the comparison is between representations.  Three off-the-shelf
classifiers (RBF support-vector machine, k-nearest neighbours, random forest)
are cross-validated on identical stratified folds.

Pipelines compared
------------------
time-domain statistics        classical statistical moments and shape factors
spectral band energies        amplitude spectrum in log-spaced bands
wavelet-packet energies       db4 packet decomposition, band energy ratios
VMD mode entropies            variational mode decomposition, per-mode entropy
multiscale permutation entropy coarse-grained permutation entropy
recurrence quantification     RQA measures of the recurrence plot
hypergraph, saturated delay   the proposed descriptor
hypergraph, entropy delay     the same descriptor built on the replaced rule
hypergraph, MI delay          the same descriptor built on the mutual
                              information rule
hypergraph, multi-resolution  the chain run per VMD mode, concatenated (the
                              full framework of Section 3.4)

Results go to results/exp7_<dataset>.json.
"""
import json
import math
import time

import numpy as np
import pywt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import data
import psr
import topology
import vmdutil
from config import RESULTS, SEED
from exp3_hypergraph import FEATURE_KEYS

CLASSIFIERS = {
    "SVM (RBF)": lambda: SVC(C=10.0, gamma="scale", kernel="rbf"),
    "kNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
    "random forest": lambda: RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1),
}


# ------------------------------------------------------------------ features
def feat_td(x):
    """Classical time-domain statistics and dimensionless shape factors."""
    x = np.asarray(x, float)
    a = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2))
    m_abs = np.mean(a)
    peak = a.max()
    return np.array([
        x.mean(), x.std(), rms, np.mean(a),
        np.mean((x - x.mean()) ** 3) / (x.std() ** 3 + 1e-12),      # skewness
        np.mean((x - x.mean()) ** 4) / (x.std() ** 4 + 1e-12),      # kurtosis
        peak, peak / (rms + 1e-12),                                 # crest
        rms / (m_abs + 1e-12),                                      # shape
        peak / (m_abs + 1e-12),                                     # clearance
        peak / (np.mean(a) + 1e-12),                                # impulse
        float(np.mean(np.diff(np.sign(x)) != 0)),                   # zcr
        float(np.percentile(a, 95) - np.percentile(a, 5)),
    ])


def feat_fft(x, n_bands=10):
    """Energy in log-spaced amplitude-spectrum bands, plus shape descriptors."""
    x = np.asarray(x, float)
    x = x - x.mean()
    f = np.fft.rfft(x * np.hanning(len(x)))
    p = np.abs(f) ** 2
    p = p[1:]
    edges = np.unique(np.round(np.logspace(0, np.log10(len(p)), n_bands + 1))
                      .astype(int))
    edges = np.clip(edges, 1, len(p))
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        bands.append(p[a:b].sum() if b > a else 0.0)
    bands = np.asarray(bands)
    frac = bands / (bands.sum() + 1e-12)
    fr = np.arange(1, len(p) + 1)
    centroid = float((fr * p).sum() / (p.sum() + 1e-12))
    ent = float(-np.sum(frac[frac > 0] * np.log(frac[frac > 0])))
    return np.concatenate([frac, [np.log1p(p.sum()), centroid, ent,
                                  float(p.max() / (p.mean() + 1e-12))]])


def feat_wpt(x, wavelet="db4", level=4):
    """Wavelet-packet band energy ratios (a standard envelope-free baseline)."""
    x = np.asarray(x, float)
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, maxlevel=level)
    nodes = [n.path for n in wp.get_level(level, order="natural")]
    e = np.array([np.sum(np.asarray(wp[p].data) ** 2) for p in nodes])
    return e / (e.sum() + 1e-12)


def _perm_entropy(x, order=3, delay=1):
    """Normalised permutation entropy."""
    x = np.asarray(x, float)
    n = len(x) - (order - 1) * delay
    if n < 10:
        return 0.0
    idx = np.arange(n)[:, None] + delay * np.arange(order)[None, :]
    emb = x[idx]
    pat = np.argsort(np.argsort(emb, axis=1), axis=1)
    code = pat.dot(1 << np.arange(order))
    _, cnt = np.unique(code, return_counts=True)
    p = cnt / cnt.sum()
    return float(-np.sum(p * np.log(p)) / math.log(math.factorial(order)))


def feat_mpe(x, scales=5, order=3):
    """Multiscale permutation entropy."""
    x = np.asarray(x, float)
    out = []
    for s in range(1, scales + 1):
        n = len(x) // s
        cg = x[:n * s].reshape(n, s).mean(1)
        out.append(_perm_entropy(cg, order=order))
    return np.array(out)


def feat_vmd_entropy(x, K=3):
    """Per-mode energy, permutation entropy and kurtosis after VMD."""
    u, w = vmdutil.decompose(x, K=K)
    feats = []
    tot = np.sum(u ** 2) + 1e-12
    for k in range(u.shape[0]):
        m = u[k]
        feats += [float(np.sum(m ** 2) / tot),
                  _perm_entropy(m),
                  float(np.mean((m - m.mean()) ** 4) / (m.std() ** 4 + 1e-12))]
    feats += list(np.asarray(w, float))
    return np.array(feats)


def _run_lengths(v):
    """Lengths of the maximal runs of ones in a 1-D 0/1 vector."""
    if not v.any():
        return np.zeros(0, dtype=int)
    d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def _rqa(x, tau, dim, percentile=5.0, max_nodes=1000):
    """Recurrence quantification analysis of the recurrence plot.

    DET and LAM are the fractions of recurrence points that lie on a diagonal
    or a vertical line of length at least two, the usual fast estimators of
    determinism and laminarity.
    """
    traj = psr.reconstruct(np.asarray(x, float), int(tau), int(dim))
    if len(traj) > max_nodes:
        traj = traj[::int(np.ceil(len(traj) / max_nodes))]
    n = len(traj)
    d = np.linalg.norm(traj[:, None, :] - traj[None, :, :], axis=2)
    thr = np.percentile(d, percentile)
    R = (d <= thr).astype(np.int8)
    np.fill_diagonal(R, 0)
    tot = max(int(R.sum()), 1)
    rr = float(R.mean())

    # diagonal neighbours: a point is on a line of length >= 2
    adj = R[:-1, :-1] & R[1:, 1:]
    diag_pts = np.zeros_like(R)
    diag_pts[:-1, :-1] |= adj
    diag_pts[1:, 1:] |= adj
    det = float(diag_pts.sum() / tot)

    vert_adj = R[:-1, :] & R[1:, :]
    vert_pts = np.zeros_like(R)
    vert_pts[:-1, :] |= vert_adj
    vert_pts[1:, :] |= vert_adj
    lam = float(vert_pts.sum() / tot)

    # longest diagonal line and the entropy of the diagonal line lengths
    lens = [_run_lengths(R.diagonal(k)) for k in range(1, n)]
    lens = np.concatenate([l for l in lens if len(l)]) if lens else np.zeros(0)
    lens = lens[lens > 0] if len(lens) else lens
    lmax = float(lens.max()) if len(lens) else 0.0
    if len(lens):
        _, cnt = np.unique(lens, return_counts=True)
        p = cnt / cnt.sum()
        ent = float(-np.sum(p * np.log(p)))
    else:
        ent = 0.0
    cols = R.sum(0)
    return np.array([rr, det, lam, np.log1p(lens.mean()) if len(lens) else 0.0,
                     np.log1p(lmax), ent, float(cols.mean()), float(cols.std())])


def feat_rqa(x):
    tau, dim, _ = psr.estimate_many(np.asarray(x, float), ["lpf_sem"])["lpf_sem"]
    return _rqa(x, tau, dim)


def _hg_vector(x, method):
    tau, dim, _ = psr.estimate_many(np.asarray(x, float), [method])[method]
    r = topology.signal_to_hypergraph(x, tau, dim)
    f = r["features"]
    return np.array([f[k] for k in FEATURE_KEYS], float)


def _hg_vector_multires(x, method="lpf_sem", K=3):
    """Multi-resolution hypergraph descriptor: the chain per VMD mode, fused.

    This is the framework of Section 3.4.  Each intrinsic mode function is
    embedded on its own and the per-mode hypergraph descriptors are
    concatenated, so the returned vector is K times the single-resolution one.
    """
    u, _w = vmdutil.decompose(np.asarray(x, float), K=K)
    feats = []
    for k in range(K):
        tau, dim, _ = psr.estimate_many(u[k], [method])[method]
        f = topology.signal_to_hypergraph(u[k], tau, dim)["features"]
        feats.extend(f[key] for key in FEATURE_KEYS)
    return np.array(feats, float)


PIPELINES = {
    "time-domain statistics": feat_td,
    "spectral band energies": feat_fft,
    "wavelet-packet energies": feat_wpt,
    "VMD mode entropies": feat_vmd_entropy,
    "multiscale perm. entropy": feat_mpe,
    "recurrence quantification": feat_rqa,
    "hypergraph, saturated delay": lambda x: _hg_vector(x, "lpf_sem"),
    "hypergraph, entropy delay": lambda x: _hg_vector(x, "raw_se"),
    "hypergraph, MI delay": lambda x: _hg_vector(x, "mi"),
    "hypergraph, multi-resolution": lambda x: _hg_vector_multires(x, "lpf_sem"),
}


# ------------------------------------------------------------------ evaluation
def _cv(F, y, n_splits=5, seed=0):
    """Stratified CV with train-fold standardisation, per classifier."""
    y = np.asarray(y)
    n_min = int(np.bincount(y)[np.bincount(y) > 0].min())
    n_splits = int(max(2, min(n_splits, n_min)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(F, y))
    out = {}
    for name, make in CLASSIFIERS.items():
        accs, f1s = [], []
        for tr, te in folds:
            sc = StandardScaler().fit(F[tr])
            clf = make().fit(sc.transform(F[tr]), y[tr])
            pred = clf.predict(sc.transform(F[te]))
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro"))
        out[name] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                     "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
    return out, n_splits


def _zscore_rows(X):
    """Per-record unit variance, as recommended for dimensionless features."""
    X = np.asarray(X, float)
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def run_dataset(X, y, tag, pipelines=None, conditions=None):
    """Evaluate every pipeline under each input condition.

    The two conditions are the records as recorded and the same records
    scaled to unit variance.  The rotor and MFPT sets both contain a real
    amplitude difference between conditions (the MFPT variable-load files
    are recorded at roughly twice the amplitude of the baseline files), so
    the as-recorded setting lets an amplitude-sensitive feature set separate
    the classes without describing the waveform at all.  The normalised
    setting removes that cue.
    """
    pipelines = pipelines or PIPELINES
    if conditions is None:
        conditions = {"as-recorded": lambda X: np.asarray(X, float),
                      "normalised": _zscore_rows}
    res = {"dataset": tag, "n_signals": int(len(X)),
           "n_per_class": int(np.bincount(y).min()), "conditions": {}}
    for cname, prep in conditions.items():
        Xc = prep(X)
        cblock = {"pipelines": {}}
        for name, fn in pipelines.items():
            t0 = time.time()
            F = np.array([fn(x) for x in Xc], float)
            F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
            cvs, n_splits = _cv(F, y)
            cblock["pipelines"][name] = {
                "n_features": int(F.shape[1]),
                "classifiers": cvs,
                "best_acc": max(v["acc"] for v in cvs.values()),
                "runtime_s": time.time() - t0,
            }
            print(f"  [{cname:11s}] {name:30s} best acc "
                  f"{cblock['pipelines'][name]['best_acc']:.3f} "
                  f"[{cblock['pipelines'][name]['runtime_s']:.0f}s]", flush=True)
        res["conditions"][cname] = cblock
    return res


def main(which="all"):
    if which in ("all", "rotor"):
        print("exp7 rotor ...", flush=True)
        X, y = data.load_rotor180()
        r = run_dataset(X, y, "rotor")
        (RESULTS / "exp7_rotor.json").write_text(json.dumps(r, indent=1))
    if which in ("all", "mfpt"):
        print("exp7 mfpt ...", flush=True)
        X, y = data.load_mfpt()
        r = run_dataset(X, y, "mfpt")
        (RESULTS / "exp7_mfpt.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
