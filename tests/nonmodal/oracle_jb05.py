"""Closed forms of Jovanović & Bamieh (2005), J. Fluid Mech. 534, 145–183, Appendix E (test oracle).

Componentwise H₂ norms of the linearized Navier–Stokes equations at kx = 0 (streamwise-constant
perturbations), channel y in [−1, 1], white body forcing, energy ∫|·|² dy (Theorem 1):
‖H_rs‖₂² = f_rs Re or g_rs Re³. Written from the paper only (independent of conatus).
"""
from __future__ import annotations

import numpy as np


def f_x(k):
    """: (2k coth 2k − 1) / (4k²) (Bamieh & Dahleh 2001 eq. 27 as corrected by JB05)."""
    return (2 * k / np.tanh(2 * k) - 1) / (4 * k ** 2)


def f_y(k):
    """, first formula, as printed (leading minus sign included; the denominator
    24k²(4k² − sinh² 2k) is negative, so the value (k²/2) tr((Δ²)⁻¹) is positive)."""
    s = np.sinh
    return -(16 * k ** 4 + 24 * k ** 2 + 3 * k * s(4 * k) - 9 * s(2 * k) ** 2) \
        / (24 * k ** 2 * (4 * k ** 2 - s(2 * k) ** 2))


def f_z(k):
    """, second formula."""
    s = np.sinh
    return (16 * k ** 4 - 3 * k * s(4 * k) + 3 * s(2 * k) ** 2) \
        / (24 * k ** 2 * (4 * k ** 2 - s(2 * k) ** 2))


def g_u_couette(k, nmax=20000):
    """g_u(kz) of Couette flow U = y, the formula printed after (JB05 p. 175), literally:

        g_u = ¼ Σ_{n≥1} kz²/γ_n⁴ {1 − (n²π²/γ_n)[A_n + B_n]},
        γ_n := −(n²π²/4 + kz²)  (negative, as printed),   α_n := √(2kz² + n²π²/4),
        n even: A_n = 2kz(cosh 2kz − 1) / (γ_n (sinh 2kz − 2kz)),
                B_n = 1 / (4(α_n coth α_n − kz coth kz)),
        n odd:  cosh 2kz + 1, sinh 2kz + 2kz, α_n tanh α_n − kz tanh kz.

    Checked on the typeset page (rendered PDF of the journal version) on the formula is
    unambiguous once the minus sign in the definition of γ_n is kept (a text extraction drops its
    layout, which made it look garbled). The values are also reproduced by an independent
    Legendre–Galerkin discretization (``oracle_galerkin_kx0.py``)."""
    n = np.arange(1, nmax, dtype=float)
    g = -(n ** 2 * np.pi ** 2 / 4 + k * k)
    a = np.sqrt(2 * k * k + n ** 2 * np.pi ** 2 / 4)
    even = (n % 2) == 0
    A = np.where(even, 2 * k * (np.cosh(2 * k) - 1) / (g * (np.sinh(2 * k) - 2 * k)),
                 2 * k * (np.cosh(2 * k) + 1) / (g * (np.sinh(2 * k) + 2 * k)))
    with np.errstate(over="ignore"):
        B = np.where(even, 1 / (4 * (a / np.tanh(a) - k / np.tanh(k))),
                     1 / (4 * (a * np.tanh(a) - k * np.tanh(k))))
    return float(np.sum(k * k / g ** 4 * (1 - n ** 2 * np.pi ** 2 / g * (A + B))) / 4)
