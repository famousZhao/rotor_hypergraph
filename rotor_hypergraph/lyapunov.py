"""Largest Lyapunov exponent from a scalar time series (Rosenstein et al.)."""
import numpy as np
from scipy.spatial import cKDTree

from psr import reconstruct


def mean_period(x):
    """Average orbital period estimated from the dominant spectral peak."""
    x = np.asarray(x, float)
    x = x - x.mean()
    f = np.abs(np.fft.rfft(x))
    f[0] = 0.0
    k = int(np.argmax(f))
    if k <= 0:
        return 10
    return max(2, int(round(len(x) / k)))


def rosenstein(x, tau, dim, dt=1.0, theiler=None, max_t=None):
    """Largest Lyapunov exponent in nats per unit time, plus the curve.

    Consecutive trajectory points are tau samples apart, so the divergence
    slope (per trajectory step) is divided by tau*dt to give a per-unit-time
    exponent.  Divergence is tracked between each point and its nearest
    neighbour outside the Theiler window, and the slope is fitted from the
    first step up to the onset of saturation.
    """
    x = np.asarray(x, float)
    traj = reconstruct(x, int(tau), int(dim))
    n = len(traj)
    if n < 80:
        return 0.0, np.array([0.0])

    if theiler is None:
        theiler = max(2, mean_period(x))
    if max_t is None:
        max_t = int(min(n // 3, 6 * theiler, 300))

    tree = cKDTree(traj)
    k = min(16, n - 1)
    _, idx = tree.query(traj, k=k + 1)
    nn = np.full(n, -1)
    for i in range(n):
        for j in range(1, k + 1):
            if abs(int(idx[i, j]) - i) > theiler:
                nn[i] = int(idx[i, j])
                break
    if (nn >= 0).sum() < 20:
        return 0.0, np.array([0.0])

    d = np.full(max_t, np.nan)
    for t in range(1, max_t):
        a = np.arange(n - t)
        a = a[nn[a] >= 0]
        b = nn[a] + t
        m = b <= n - 1
        a, b = a[m], b[m]
        if len(a) < 10:
            continue
        dd = np.linalg.norm(traj[a] - traj[b], axis=1)
        dd = dd[dd > 0]
        if len(dd):
            d[t] = np.mean(np.log(dd))

    valid = ~np.isnan(d)
    if valid.sum() < 8:
        return 0.0, d
    t_ax = np.arange(max_t)[valid]
    d_ok = d[valid]

    # saturation onset: first index reaching 90% of the total rise
    lo, hi = d_ok.min(), d_ok.max()
    if hi - lo <= 0:
        return 0.0, d
    onset = int(np.argmax(d_ok >= lo + 0.9 * (hi - lo)))
    t_end = max(3, min(onset, len(d_ok) - 1))
    slope = np.polyfit(t_ax[:t_end], d_ok[:t_end], 1)[0]
    scale = float(max(1, int(tau)) * dt)
    return float(slope / scale), d
