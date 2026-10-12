"""Variational mode decomposition wrapper (vmdpy)."""
import numpy as np
from vmdpy import VMD

from config import VMD_ALPHA, VMD_K


def decompose(x, K=VMD_K, alpha=VMD_ALPHA, tol=1e-7):
    """Return the K modes, shape (K, len(x)), ordered low to high frequency."""
    x = np.asarray(x, float)
    u, _, omega = VMD(x, alpha, 0.0, int(K), 0, 1, tol)
    w = np.asarray(omega)
    w = w[-1] if w.ndim > 1 else w          # final centre frequencies
    order = np.argsort(w)
    return u[order], w[order]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import data
    X, y = data.load_rotor180()
    u, w = decompose(X[0])
    print("modes", u.shape, "center freqs", np.round(w, 4))
    resid = X[0] - u.sum(0)
    print("reconstruction residual (max abs)", np.abs(resid).max())
