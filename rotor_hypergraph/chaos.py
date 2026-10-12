"""Chaotic benchmark systems used to validate phase-space reconstruction.

All integrators are fourth-order Runge-Kutta except Mackey-Glass, which is a
delay differential equation integrated with an explicit Euler step plus linear
interpolation of the delayed term (the convention used in the author's
reference code).

Reference largest Lyapunov exponents (natural log, per time unit) are taken
from the literature for the parameter sets used here and are only used as a
plausibility anchor, not as a fitting target.
"""
import numpy as np


def _rk4(f, state0, step, n):
    s = np.zeros((n, len(state0)))
    s[0] = state0
    for i in range(n - 1):
        y = s[i]
        k1 = f(y)
        k2 = f(y + 0.5 * step * k1)
        k3 = f(y + 0.5 * step * k2)
        k4 = f(y + step * k3)
        s[i + 1] = y + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def lorenz(n=12000, step=0.01, sigma=10.0, beta=8.0 / 3.0, rho=28.0,
           x0=(-1.0, 0.0, 1.0)):
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    return _rk4(f, np.array(x0, float), step, n)


def rossler(n=12000, step=0.05, a=0.2, b=0.2, c=5.7, x0=(0.1, 0.1, 0.1)):
    def f(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])
    return _rk4(f, np.array(x0, float), step, n)


def duffing(n=12000, step=0.05, delta=0.2, alpha=-1.0, beta=1.0,
            gamma=0.3, omega=1.0, x0=(0.0, 0.0)):
    def f(s, t):
        x, y = s
        return np.array([y,
                         -delta * y - alpha * x - beta * x ** 3
                         + gamma * np.cos(omega * t)])
    s = np.zeros((n, 2))
    s[0] = x0
    for i in range(n - 1):
        t = i * step
        k1 = f(s[i], t)
        k2 = f(s[i] + 0.5 * step * k1, t + 0.5 * step)
        k3 = f(s[i] + 0.5 * step * k2, t + 0.5 * step)
        k4 = f(s[i] + step * k3, t + step)
        s[i + 1] = s[i] + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def chen(n=20000, step=0.01, a=35.0, b=3.0, c=28.0, x0=(-1.0, 0.0, 1.0)):
    def f(s):
        x, y, z = s
        return np.array([a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z])
    return _rk4(f, np.array(x0, float), step, n)


def mackey_glass(n=8000, tau=17, a=0.2, b=0.1, c=10.0, x0=1.2, dt=1.0):
    """Mackey-Glass DDE, explicit Euler with linear interpolation."""
    hist_len = int(np.ceil(tau / dt)) + 2
    X = np.full(hist_len, x0, dtype=float)
    out = np.empty(n)
    out[0] = x0
    # index -> time mapping: X[j] holds x(t0 + j*dt)
    for i in range(1, n):
        t = i * dt
        t_tau = t - tau
        if t_tau < 0:
            x_tau = x0
        else:
            # locate in stored history
            pos = t_tau / dt
            j = int(np.floor(pos))
            frac = pos - j
            if j + 1 < len(X):
                x_tau = X[j] + (X[j + 1] - X[j]) * frac
            else:
                x_tau = X[-1]
        dx = a * x_tau / (1.0 + x_tau ** c) - b * out[i - 1]
        out[i] = out[i - 1] + dt * dx
        X = np.append(X, out[i])
    return out


# name -> (generator callable, sample step, transient, reference LLE, ref dim)
def build(name, n=12000):
    """Return (x, dt, lle_ref) for one of the five benchmark systems."""
    if name == "Lorenz":
        s = lorenz(n=n)
        return s[2000:, 0], 0.01, 0.906
    if name == "Rossler":
        s = rossler(n=n)
        return s[2000:, 0], 0.05, 0.071
    if name == "Duffing":
        s = duffing(n=n)
        return s[2000:, 0], 0.05, 0.12
    if name == "Chen":
        s = chen(n=n)
        return s[2000:, 0], 0.01, 2.027
    if name == "MackeyGlass":
        x = mackey_glass(n=n)
        return x[2000:], 1.0, 0.006
    raise ValueError(name)


SYSTEMS = ["Lorenz", "Rossler", "Duffing", "MackeyGlass", "Chen"]


if __name__ == "__main__":
    for nm in SYSTEMS:
        x, dt, lle = build(nm, n=4000)
        print(f"{nm:12s} n={len(x):6d} dt={dt} std={np.std(x):.4f} "
              f"range=({x.min():.3f},{x.max():.3f}) lle_ref={lle}")
