"""Near-recurrences ending at the current state: period / shift guesses for a Newton seed.

Given a stored trajectory u(t_0), …, u(t_n) of a 1D RealFourier problem on a periodic interval of
length L (coefficients ``[cos_0, msin_0, cos_1, msin_1, …]``, see :mod:`conatus.ecs.fourier_ops`),
the *endpoint* recurrence distance for a lag T = t_n − t_j is

    d(T) = min_ℓ ‖S τ_ℓ u(t_n) − u(t_n − T)‖ / ‖u(t_n)‖,        S = I (RPO) or R (pre-periodic),

with τ_ℓ f(x) = f(x + ℓ) and R f(x) = −f(−x). A small local minimum of d means the trajectory came
back close to (a symmetry copy of) itself after T; since the loop ends at u(t_n), the pair
(T, ℓ) is a guess for an orbit **through the current state** (equivariance: S τ_ℓ commutes with the
flow). This is the same measure as the KS benchmark, restricted to the
last state so that it is cheap (one lag sweep, O(n · n_ell · N)) and can run inside the job when
a monitor asks for a checkpoint. Norm: the L²(0, L) norm / L (mean weight 1, pairs weight 1/2).
Everything is an argument; nothing is chosen here.
"""
from __future__ import annotations

import numpy as np

from ..errors import OptionError

__all__ = ["endpoint_recurrences"]


def endpoint_recurrences(coeffs: np.ndarray, times: np.ndarray, L: float, *, Tmin: float,
                         Tmax: float, reflect: bool, n_ell: int, top: int) -> list[dict]:
    """Local minima of d(T) (see module doc), best first, at most ``top``, distinct by > 1 sample.

    coeffs  (n, N) RealFourier coefficients of the stored states, oldest first, last = current
    times   (n,) their times (increasing; need not be uniform)
    Returns dicts {T, ell, d, reflect, t_start}; ``ell`` in [−L/2, L/2). Empty if nothing qualifies.
    """
    c = np.asarray(coeffs, dtype=float)
    t = np.asarray(times, dtype=float)
    if c.ndim != 2 or c.shape[0] != t.size or c.shape[1] % 2:
        raise OptionError(f"coeffs must be (n, even N) with n = len(times); got {c.shape}, {t.shape}",
                          fix="stack the stored RealFourier coefficient vectors as rows, one per "
                              "time")
    if t.size >= 2 and np.any(np.diff(t) <= 0):
        raise OptionError("times must be strictly increasing",
                          fix="store the states oldest first, without repeated times")
    if not (0 < Tmin < Tmax) or n_ell < 1 or top < 1:
        raise OptionError(f"need 0 < Tmin < Tmax, n_ell >= 1, top >= 1 (got {Tmin}, {Tmax}, {n_ell}, "
                          f"{top})", fix="declare a positive window of return times Tmin < Tmax")
    a = c[:, 0::2] + 1j * c[:, 1::2]                    # a_k with f = Re(a_k e^{i k x}) per pair
    K = a.shape[1]
    w = np.where(np.arange(K) == 0, 1.0, 0.5)           # ∫ |.|² dx / L weights
    # coefficients of S τ_ℓ u(t_n) = end_k · e^{±ikℓ}: τ_ℓ multiplies a_k by e^{ikℓ}, then
    # R: a -> −conj(a) (i.e. (c, s) -> (−c, s)) turns it into −conj(a_k) e^{−ikℓ}
    end = -np.conj(a[-1]) if reflect else a[-1]
    norm2_end = float(np.sum(w * np.abs(a[-1]) ** 2))
    if norm2_end == 0.0:
        return []
    ells = np.arange(n_ell) * L / n_ell
    phase = np.exp((-1j if reflect else 1j) * np.outer(np.arange(K) * 2 * np.pi / L, ells))
    lags = t[-1] - t[:-1]
    sel = np.flatnonzero((lags >= Tmin) & (lags <= Tmax))
    if sel.size == 0:
        return []
    past = a[sel]                                          # (m, K)
    cross = np.real((np.conj(past) * end * w) @ phase)     # ⟨u(t_j), S τ_ℓ u(t_n)⟩ for each ℓ
    j = np.argmax(cross, axis=1)
    norm2_past = np.sum(w * np.abs(past) ** 2, axis=1)
    d2 = norm2_past + norm2_end - 2 * cross[np.arange(sel.size), j]
    d = np.sqrt(np.maximum(d2, 0.0) / norm2_end)
    # sel is in increasing time = decreasing lag; walk in increasing lag for the local-minimum test
    order = np.argsort(lags[sel])
    d, j, idx = d[order], j[order], sel[order]
    mins = [k for k in range(1, d.size - 1) if d[k] < d[k - 1] and d[k] <= d[k + 1]]
    mins.sort(key=lambda k: d[k])
    out: list[dict] = []
    for k in mins:
        if all(abs(k - o["_k"]) > 1 for o in out):
            ell = float((ells[j[k]] + L / 2) % L - L / 2)
            out.append({"_k": k, "T": float(lags[idx[k]]), "ell": ell, "d": float(d[k]),
                        "reflect": bool(reflect), "t_start": float(t[idx[k]])})
        if len(out) == top:
            break
    for o in out:
        del o["_k"]
    return out
