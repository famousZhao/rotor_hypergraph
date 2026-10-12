"""Phase-space reconstruction parameter estimation.

Estimators
----------
lpf_sem   low-pass filtered symbolic entropy (the proposed method).  The
          symbolic-entropy curve H(tau) is smoothed with a Butterworth
          low-pass filter and the optimal delay is the onset of saturation,
          i.e. the first tau after the slope peak where the slope falls to a
          fixed fraction of its maximum.
lpf_d2    variant of lpf_sem using the largest curvature (|d2H/dtau2|).
raw_se    the original symbolic-entropy rule that lpf_sem replaces: the first
          local maximum of the unfiltered H(tau).
mi        first local minimum of the average mutual information.
acf       first crossing of the autocorrelation at 1/e.
cc        Kim-Eykholt-Salas C-C method.
fnn       embedding dimension only (Cao's false nearest neighbours).
"""
import numpy as np
from scipy import signal
from scipy.spatial import cKDTree

from config import (BUTTER_CUTOFF, BUTTER_ORDER, E_FACTOR, SYMBOLS_P,
                    TAU_MAX)


# ------------------------------------------------------------------ phase space
def reconstruct(data, delay, dim):
    """Time-delay embedding, rows are phase-space vectors."""
    n = len(data) - (dim - 1) * delay
    if n <= 0:
        raise ValueError("series too short for the requested embedding")
    idx = np.arange(n)[:, None] + delay * np.arange(dim)[None, :]
    return data[idx]


# ------------------------------------------------------------------ symbolic entropy
def symbolic_entropy(data, delay, p=SYMBOLS_P, e=None):
    """Shannon entropy of the symbolic partition (the author's definition).

    A phase-space vector is encoded as a (p-1)-bit word.  The sign rule is
    fixed by the first coordinate: if |v0| < e the remaining bits record
    |v_i| < e, otherwise they record |v_i| >= e.
    """
    data = np.asarray(data, float)
    if e is None:
        e = E_FACTOR * np.std(data)
    n = len(data) - (p - 1) * delay
    if n < 2:
        return 0.0
    idx = np.arange(n)[:, None] + delay * np.arange(p)[None, :]
    a = np.abs(data[idx])
    first = a[:, 0] < e
    rest = a[:, 1:]
    bits = np.where(first[:, None], rest < e, rest >= e)
    codes = bits.dot(1 << np.arange(p - 1))
    counts = np.bincount(codes)
    prob = counts[counts > 0] / float(counts.sum())
    return float(-np.sum(prob * np.log(prob)))


def symbolic_entropy_curve(data, taus, p=SYMBOLS_P, e=None):
    """H(tau) normalised by p, evaluated on the given delay grid."""
    if e is None:
        e = E_FACTOR * np.std(data)
    return np.array([symbolic_entropy(data, int(t), p, e) / p for t in taus])


def _smooth_curve(data, taus, p, e):
    H = symbolic_entropy_curve(data, taus, p, e)
    b, a = signal.butter(BUTTER_ORDER, BUTTER_CUTOFF, "low")
    pad = 3 * max(len(a), len(b))
    if len(H) <= pad:
        Hf = H.copy()
    else:
        Hf = signal.filtfilt(b, a, H)
    return H, Hf


def _filter_curve(H):
    """Butterworth low-pass of a curve, zero phase, with padding guard."""
    b, a = signal.butter(BUTTER_ORDER, BUTTER_CUTOFF, "low")
    pad = 3 * max(len(a), len(b))
    return signal.filtfilt(b, a, H) if len(H) > pad else H.copy()


def lpf_sem_from_curve(H, taus, alpha=0.10):
    """Saturation-onset delay from a precomputed symbolic-entropy curve.

    The junction rule is the derivative threshold of Eq. (tau_star).  The
    diagnostics additionally expose the unnormalised second derivative peak
    (``tau_d2``) and the normalised-curvature peak (``tau_kappa``) of Eq.
    (kappa), so that the variants tested in Experiment 9 share this one curve.
    """
    Hf = _filter_curve(H)
    d1 = np.gradient(Hf)
    d2 = np.gradient(d1)
    i_peak = int(np.argmax(d1))
    thr = alpha * d1[i_peak]
    below = np.where(d1[i_peak:] <= thr)[0]
    i_sat = i_peak + int(below[0]) if len(below) else len(taus) - 1
    i_d2 = int(np.argmax(np.abs(d2)))
    kappa = np.abs(d2) / np.maximum((1.0 + d1 ** 2) ** 1.5, 1e-12)
    i_kappa = int(np.argmax(kappa))
    return {"tau": int(taus[i_sat]), "tau_d2": int(taus[i_d2]),
            "tau_kappa": int(taus[i_kappa]),
            "taus": taus, "H": H, "Hf": Hf, "d1": d1, "d2": d2, "kappa": kappa,
            "i_sat": i_sat, "i_peak": i_peak, "i_d2": i_d2, "i_kappa": i_kappa}


def lpf_curv_from_curve(H, taus):
    """Path-1 variant A: normalised-curvature maximum, Eq. (kappa).

    tau* = argmax_tau |H''| / (1 + [H']^2)^{3/2}.  On the clean benchmark this
    should sit closer to the physical transition than the fixed derivative
    threshold, which is biased toward the flat saturated region.
    """
    Hf = _filter_curve(H)
    d1 = np.gradient(Hf)
    d2 = np.gradient(d1)
    kappa = np.abs(d2) / np.maximum((1.0 + d1 ** 2) ** 1.5, 1e-12)
    i = int(np.argmax(kappa))
    return {"tau": int(taus[i]), "taus": taus, "H": H, "Hf": Hf,
            "d1": d1, "d2": d2, "kappa": kappa, "i": i}


def lpf_fit_from_curve(H, taus):
    """Path-1 variant B: saturating-exponential fit, tau* = tau_c.

    H(tau) ~ A (1 - exp(-tau/tau_c)) + B is fitted by nonlinear least
    squares over the whole curve, and the time constant tau_c is the estimate.
    This uses every point rather than a single threshold and gives tau_c a
    clear physical reading (the scale of the transition).
    """
    from scipy.optimize import curve_fit

    def _model(t, A, tc, B):
        return A * (1.0 - np.exp(-np.clip(t, 0.0, None) / tc)) + B

    Hf = _filter_curve(H)
    tt = np.arange(len(taus), dtype=float) + 1.0
    lo, hi = float(Hf.min()), float(Hf.max())
    try:
        popt, _ = curve_fit(_model, tt, Hf, p0=[hi - lo, 5.0, lo],
                            bounds=([0.0, 1e-3, -np.inf],
                                    [2.0 * (hi - lo + 1e-6), 1e3,
                                     np.inf]),
                            maxfev=20000)
        tc = popt[1]
    except Exception:
        tc = 5.0
    tau = int(round(max(1.0, min(float(len(taus)), tc))))
    return {"tau": int(taus[tau - 1]) if 1 <= tau <= len(taus)
            else int(taus[-1]), "tau_c": tc, "taus": taus, "H": H,
            "Hf": Hf, "A": float(popt[0]), "B": float(popt[1])}


def lpf_adaptive_from_curve(H, taus):
    """Path-1 variant C: adaptive saturation fraction.

    The fixed alpha = 0.1 is replaced by alpha = H'(tau_mid)/H'(tau_p), where
    tau_mid is the half-height crossing of the filtered curve, so the junction
    tracks the actual shape instead of a hard-coded fraction.
    """
    Hf = _filter_curve(H)
    d1 = np.gradient(Hf)
    i_peak = int(np.argmax(d1))
    lo, hi = float(Hf.min()), float(Hf.max())
    target = 0.5 * (lo + hi)
    cross = np.where(Hf >= target)[0]
    if len(cross) == 0:
        alpha = 0.1
    else:
        i_mid = int(cross[0])
        alpha = d1[i_mid] / max(d1[i_peak], 1e-12)
        alpha = float(np.clip(alpha, 0.01, 1.0))
    thr = alpha * d1[i_peak]
    below = np.where(d1[i_peak:] <= thr)[0]
    i_sat = i_peak + int(below[0]) if len(below) else len(taus) - 1
    return {"tau": int(taus[i_sat]), "alpha": alpha, "i_mid": int(cross[0])
            if len(cross) else None, "taus": taus, "H": H, "Hf": Hf,
            "d1": d1, "i_sat": i_sat, "i_peak": i_peak}


def permutation_entropy_curve(data, taus, m=4, normalize=True):
    """Bandt-Pompe permutation entropy as a function of the lag tau.

    This is the standard ordinal-pattern symbolisation (the review's "standard
    scheme"), in contrast to the first-coordinate rule of Eq. (H).  A delay
    vector is replaced by the permutation ranks of its m coordinates and the
    entropy of the rank-distribution is evaluated on the same delay grid.
    """
    import math
    data = np.asarray(data, float)
    out = np.empty(len(taus))
    denom = math.log(math.factorial(m)) if normalize else 1.0
    for k, t in enumerate(taus):
        t = int(t)
        n = len(data) - (m - 1) * t
        if n < 2:
            out[k] = 0.0
            continue
        idx = np.arange(n)[:, None] + t * np.arange(m)[None, :]
        w = data[idx]
        ranks = np.argsort(np.argsort(w, axis=1), axis=1)
        codes = ranks.dot(1 << np.arange(m)[::-1])
        counts = np.bincount(codes)
        p = counts[counts > 0] / float(counts.sum())
        out[k] = -np.sum(p * np.log(p)) / denom
    return out


def pe_sem_from_curve(PE, taus, alpha=0.10):
    """The saturated-shoulder rule applied to a permutation-entropy curve.

    The framework (filter then derivative threshold) is kept identical to
    lpf_sem; only the symbolisation is the standard ordinal-pattern one.  This
    isolates the effect of the symbolisation on the delay estimate.
    """
    Hf = _filter_curve(PE)
    d1 = np.gradient(Hf)
    i_peak = int(np.argmax(d1))
    thr = alpha * d1[i_peak]
    below = np.where(d1[i_peak:] <= thr)[0]
    i_sat = i_peak + int(below[0]) if len(below) else len(taus) - 1
    return {"tau": int(taus[i_sat]), "taus": taus, "H": PE, "Hf": Hf,
            "d1": d1, "i_sat": i_sat, "i_peak": i_peak}


def raw_se_from_curve(H, taus):
    """Original rule: first local maximum of the unfiltered H(tau)."""
    idx, _ = signal.find_peaks(H)
    i = int(idx[0]) if len(idx) else int(np.argmax(H))
    return {"tau": int(taus[i]), "taus": taus, "H": H, "i": i}


def lpf_sem(data, p=SYMBOLS_P, e=None, taus=None, alpha=0.10):
    """Proposed estimator.  Returns a dict with tau and the diagnostics."""
    if taus is None:
        taus = np.arange(1, TAU_MAX + 1)
    H, _ = _smooth_curve(data, taus, p, e)
    return lpf_sem_from_curve(H, taus, alpha)


def lpf_curv(data, p=SYMBOLS_P, e=None, taus=None):
    """Path-1 variant: normalised-curvature maximum on the SE curve."""
    if taus is None:
        taus = np.arange(1, TAU_MAX + 1)
    H = symbolic_entropy_curve(data, taus, p, e)
    return lpf_curv_from_curve(H, taus)


def lpf_fit(data, p=SYMBOLS_P, e=None, taus=None):
    """Path-1 variant: saturating-exponential fit, tau* = tau_c."""
    if taus is None:
        taus = np.arange(1, TAU_MAX + 1)
    H = symbolic_entropy_curve(data, taus, p, e)
    return lpf_fit_from_curve(H, taus)


def lpf_adaptive(data, p=SYMBOLS_P, e=None, taus=None):
    """Path-1 variant: adaptive saturation fraction."""
    if taus is None:
        taus = np.arange(1, TAU_MAX + 1)
    H = symbolic_entropy_curve(data, taus, p, e)
    return lpf_adaptive_from_curve(H, taus)


def pe_sem(data, taus=None, m=4, alpha=0.10):
    """Standard ordinal-pattern symbolisation with the saturated-shoulder rule."""
    if taus is None:
        taus = np.arange(1, TAU_MAX + 1)
    PE = permutation_entropy_curve(data, taus, m=m)
    return pe_sem_from_curve(PE, taus, alpha)


def raw_se(data, p=SYMBOLS_P, e=None, taus=None):
    """Original rule: first local maximum of the unfiltered symbolic entropy."""
    if taus is None:
        taus = np.arange(1, TAU_MAX + 1)
    H = symbolic_entropy_curve(data, taus, p, e)
    return raw_se_from_curve(H, taus)


def estimate_many(data, methods=("lpf_sem", "raw_se", "mi", "acf", "cc"),
                  max_dim=12, with_dim=True):
    """Estimate several delays for one signal, sharing the entropy curve.

    ``with_dim`` skips Cao's embedding-dimension pass, which is the dominant
    cost of the function and is not needed when only the delay is compared.
    """
    data = np.asarray(data, float)
    taus = np.arange(1, TAU_MAX + 1)
    out = {}
    need_curve = any(m in ("lpf_sem", "raw_se", "lpf_sem_d2", "lpf_curv",
                           "lpf_fit", "lpf_adaptive") for m in methods)
    need_pe = "pe_sem" in methods
    H = symbolic_entropy_curve(data, taus) if need_curve else None
    PE = permutation_entropy_curve(data, taus) if need_pe else None
    for m in methods:
        if m == "lpf_sem":
            info = lpf_sem_from_curve(H, taus)
            tau = info["tau"]
        elif m == "lpf_sem_d2":
            info = lpf_sem_from_curve(H, taus)
            tau = info["tau_d2"]
            info["tau"] = tau
        elif m == "lpf_curv":
            info = lpf_curv_from_curve(H, taus)
            tau = info["tau"]
        elif m == "lpf_fit":
            info = lpf_fit_from_curve(H, taus)
            tau = info["tau"]
        elif m == "lpf_adaptive":
            info = lpf_adaptive_from_curve(H, taus)
            tau = info["tau"]
        elif m == "pe_sem":
            info = pe_sem_from_curve(PE, taus)
            tau = info["tau"]
        elif m == "raw_se":
            info = raw_se_from_curve(H, taus)
            tau = info["tau"]
        elif m == "mi":
            info = mutual_information(data, taus)
            tau = info["tau"]
        elif m == "acf":
            info = autocorrelation(data, taus)
            tau = info["tau"]
        elif m == "cc":
            info = cc_method(data)
            tau = info["tau"]
        else:
            raise ValueError(m)
        if with_dim:
            dim, e1 = cao_fnn(data, tau, max_dim)
            info["e1"] = e1
        else:
            dim = -1
        out[m] = (int(tau), int(dim), info)
    return out


def mutual_information(data, taus=None, bins=32):
    """Average mutual information; optimal delay is the first local minimum."""
    data = np.asarray(data, float)
    if taus is None:
        taus = np.arange(1, TAU_MAX + 1)
    vals = []
    for t in taus:
        x, y = data[:-t], data[t:]
        h, xe, ye = np.histogram2d(x, y, bins=bins)
        h = h / h.sum()
        px = h.sum(1, keepdims=True)
        py = h.sum(0, keepdims=True)
        nz = h > 0
        vals.append(float(np.sum(h[nz] * np.log(h[nz] /
                                              (px @ py)[nz]))))
    vals = np.array(vals)
    idx, _ = signal.find_peaks(-vals)
    i = int(idx[0]) if len(idx) else int(np.argmin(vals))
    return {"tau": int(taus[i]), "taus": taus, "mi": vals, "i": i}


def autocorrelation(data, taus=None):
    """First delay at which the normalised autocorrelation falls below 1/e."""
    data = np.asarray(data, float)
    data = data - data.mean()
    if taus is None:
        taus = np.arange(1, TAU_MAX + 1)
    denom = np.dot(data, data)
    vals = np.array([np.dot(data[:-t], data[t:]) / denom for t in taus])
    below = np.where(vals < 1.0 / np.e)[0]
    i = int(below[0]) if len(below) else int(np.argmin(vals))
    return {"tau": int(taus[i]), "taus": taus, "acf": vals, "i": i}


def cc_method(data, max_tau=None, m_list=(2, 3, 4, 5)):
    """Kim-Eykholt-Salas C-C method.  Returns tau and the embedding window."""
    data = np.asarray(data, float)
    n = len(data)
    if max_tau is None:
        max_tau = min(100, n // 20)

    sigma = np.std(data)
    rng = np.random.default_rng(0)
    cap = 400                      # cap on the correlation-integral sample size
    S = np.zeros(max_tau + 1)
    dS = np.zeros(max_tau + 1)
    for t in range(1, max_tau + 1):
        s_m_list = []
        for m in m_list:
            nb = n // t
            if nb < m + 1:
                continue
            blocks = data[:nb * t].reshape(nb, t)
            if nb > cap:
                blocks = blocks[rng.choice(nb, cap, replace=False)]
            s_m = []
            for j in range(t):
                col = blocks[:, j]
                if np.std(col) < 1e-12:
                    continue
                d = np.abs(col[:, None] - col[None, :])
                s_m.append(np.mean(d < 0.5 * sigma))
            if s_m:
                s_m_list.append(np.mean(s_m))
        if s_m_list:
            S[t] = np.mean(s_m_list)
            dS[t] = float(np.max(s_m_list) - np.min(s_m_list))

    t_vals = np.arange(1, max_tau + 1)
    S_head = S[1:]
    Scor_head = np.abs(S_head) + dS[1:]
    # first zero crossing of S(t) gives the delay
    zero_cross = np.where(np.diff(np.sign(S_head)) != 0)[0]
    tau = int(t_vals[zero_cross[0]]) if len(zero_cross) else int(np.argmin(np.abs(S_head)) + 1)
    # global minimum of the correlation statistic gives the embedding window
    tau_w = int(t_vals[int(np.argmin(Scor_head))])
    Scor = np.concatenate([[0.0], Scor_head])
    if tau <= 0:
        tau = 1
    return {"tau": tau, "tau_w": tau_w, "S": S, "Scor": Scor,
            "dim": max(2, int(round(tau_w / tau)) + 1)}


def cao_fnn(data, delay, max_dim=12, tol=10.0):
    """Cao's false-nearest-neighbour criterion for the embedding dimension."""
    data = np.asarray(data, float)
    e1 = []
    for m in range(1, max_dim + 1):
        n_vec = len(data) - m * delay   # vectors available at dimension m+1
        if n_vec < 20:
            break
        emb = reconstruct(data, delay, m)[:n_vec]
        emb1 = reconstruct(data, delay, m + 1)
        tree = cKDTree(emb)
        dist, idx = tree.query(emb, k=2)
        nn = idx[:, 1]
        good = dist[:, 1] > 0
        num = np.linalg.norm(emb1 - emb1[nn], axis=1)
        e1.append(np.mean((num / np.maximum(dist[:, 1], 1e-12))[good]))
    e1 = np.array(e1)
    if len(e1) < 3:
        return 3, e1
    d = np.abs(np.diff(e1))
    for i in range(1, len(d)):
        if d[i] < 0.05:
            return i + 2, e1
    return int(np.argmin(d)) + 2, e1


# ------------------------------------------------------------------ interface
METHODS = ["lpf_sem", "raw_se", "mi", "acf", "cc"]
DIM_METHOD = {"cc": "cc", "fnn": "fnn"}


def estimate(data, method="lpf_sem", max_dim=12):
    """Return (tau, dim, info) for a named delay estimator."""
    if method == "lpf_sem":
        info = lpf_sem(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    elif method == "lpf_sem_d2":
        info = lpf_sem(data)
        tau = info["tau_d2"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
        info["tau"] = tau
    elif method == "lpf_curv":
        info = lpf_curv(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    elif method == "lpf_fit":
        info = lpf_fit(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    elif method == "lpf_adaptive":
        info = lpf_adaptive(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    elif method == "pe_sem":
        info = pe_sem(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    elif method == "raw_se":
        info = raw_se(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    elif method == "mi":
        info = mutual_information(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    elif method == "acf":
        info = autocorrelation(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    elif method == "cc":
        info = cc_method(data)
        tau = info["tau"]
        dim, e1 = cao_fnn(data, tau, max_dim)
        info["e1"] = e1
    else:
        raise ValueError(method)
    return int(tau), int(dim), info
