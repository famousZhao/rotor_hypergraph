"""Loaders for the rotor datasets, with a small local cache.

Primary dataset  : 180 measured segments, 4 classes x 45, 2048 points each.
Secondary dataset: 360 segments whose last 180 rows are arithmetic averages of
                   the four measured classes.  Used only for the clearly
                   labelled simulated-coupling study.
"""
import numpy as np
import scipy.io as sio

from config import (CACHE, D_ROTOR180, D_ROTOR360, CLASSES, CLASSES8, D_MFPT,
                    MFPT_DECIM, MFPT_CLASSES, N_MFPT_PER_CLASS, N_MFPT_POINTS)


def _load_mat(path, key="Y_wavedeno"):
    d = sio.loadmat(str(path))
    return np.asarray(d[key], dtype=float)


def load_rotor180():
    """Return (X, y) with X (180, 2048) and y (180,) in 0..3."""
    cache = CACHE / "rotor180.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]
    X = _load_mat(D_ROTOR180)
    y = np.repeat(np.arange(4), 45)
    assert X.shape == (180, 2048), X.shape
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def load_rotor360():
    """Return (X, y) with X (360, 2048) and y (360,) in 0..7.

    Rows 0..179 are the measured segments; rows 180..359 are simulated
    couplings, i.e. arithmetic means of the measured single-fault classes.
    """
    cache = CACHE / "rotor360.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]
    X = _load_mat(D_ROTOR360)
    y = np.repeat(np.arange(8), 45)
    assert X.shape == (360, 2048), X.shape
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def add_awgn(x, snr_db, rng):
    """Add white Gaussian noise at a target SNR (dB) to a 1-D signal."""
    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(p_noise), size=x.shape)


# ------------------------------------------------------------------ MFPT
_MFPT_PREFIX = [("baseline", 0), ("InnerRaceFault", 1), ("OuterRaceFault", 2)]


def _mfpt_files():
    """(path, class) for every MFPT .mat file, sorted for determinism."""
    out = []
    for p in sorted(D_MFPT.rglob("*.mat")):
        for pre, c in _MFPT_PREFIX:
            if p.name.startswith(pre):
                out.append((p, c))
                break
    return out


def _mfpt_signal(path):
    """Vibration vector of one MFPT file, decimated to the common rate."""
    from scipy.signal import decimate
    d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    b = d["bearing"]
    x = np.asarray(b.gs, dtype=float).ravel()
    x = x - x.mean()
    if int(b.sr) > 48828:
        x = decimate(x, MFPT_DECIM, ftype="fir", zero_phase=True)
    return x


def load_mfpt():
    """Return (X, y) with X (135, 2048) and y in 0..2.

    Three bearing conditions: healthy, inner-race fault, outer-race fault.
    Records are cut from every file of a class and spread over the record so
    that the 45 segments do not come from a single file or a single segment
    of the run.
    """
    cache = CACHE / "mfpt.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]

    by_class = {}
    for p, c in _mfpt_files():
        by_class.setdefault(c, []).append(p)

    X, y = [], []
    for c in sorted(by_class):
        sigs = [_mfpt_signal(p) for p in by_class[c]]
        # candidate (start, signal) pairs, walking each file from the top
        starts = []
        for s in sigs:
            for a in range(0, len(s) - N_MFPT_POINTS + 1, N_MFPT_POINTS):
                starts.append((a, s))
        n_avail = len(starts)
        pick = np.linspace(0, n_avail - 1, N_MFPT_PER_CLASS).astype(int)
        for j in pick:
            a, s = starts[j]
            X.append(s[a:a + N_MFPT_POINTS])
            y.append(c)

    X = np.asarray(X, float)
    y = np.asarray(y, int)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def class_names(eight=False):
    return CLASSES8 if eight else CLASSES


if __name__ == "__main__":
    X, y = load_rotor180()
    print("rotor180", X.shape, np.bincount(y))
    X8, y8 = load_rotor360()
    print("rotor360", X8.shape, np.bincount(y8))
