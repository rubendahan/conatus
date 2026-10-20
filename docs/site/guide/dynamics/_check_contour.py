"""Executable snippets of the guide page "Eigenvalues in a region" (docs/site/guide/dynamics/contour.md).

Every code block of that page is a region of this file (``# [name]`` ... ``# [/name]``), included
verbatim with ``literalinclude`` (``:dedent: 4``): what the reader sees is what ran. The regions build
on each other in reading order; the asserts after each region are the checks (not shown on the page).
Not a Sphinx source (leading underscore).

Run (CPU, about ten seconds; as a job, not on the login node):
"""
from __future__ import annotations

import numpy as np


def contour() -> None:
    # [ct-region]
    import conatus as cn

    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")

    box = cn.box(re=(0.1, 0.5), im=(-0.3, 0.05))       # a region of the c plane
    res = m.eigenvalues(N=64, region=box)
    print(res.count, res.count_method)                  # certified number of eigenvalues inside
    print(res.explain())                                # every mode inside, with its residual
    # [/ct-region]
    dense = m.eigenvalues(N=64)
    inside = dense.values[box.contains(dense.values)]
    assert res.count == inside.size == len(res) == 12
    assert max(np.min(np.abs(res.values - z)) for z in inside) < 1e-9
    assert res.residuals.max() < 1e-12

    # [ct-count]
    from conatus.expert import HillBlocks

    a, q = 1.5, 0.5                      # Mathieu x'' + (a - 2q cos 2t) x = 0, period pi
    L0 = np.array([[0, -1], [a, 0]], complex)
    C = np.array([[0, 0], [-q, 0]], complex)
    H = HillBlocks(diag=lambda mu, n: L0 + (mu + 2j * n) * np.eye(2), couplings={1: C, -1: C},
                   Nh=24, size=2, label="Mathieu", omega=2.0, mass=np.eye(2))

    pair = cn.box(re=(-0.5, 0.5), im=(0.25, 1.75))       # around the neutral pair
    gap = cn.box(re=(-0.5, 0.5), im=(-0.9, 0.9))         # known to hold none
    n = cn.contour.count(H, pair, control=(gap, 0))
    print(n.explain())
    roots = cn.contour.beyn(H, pair, counted=n)          # the two exponents, certified
    print(roots.values, roots.residuals)
    # [/ct-count]
    assert n.count == 2 and n.poles == 0
    assert np.abs(roots.values.real).max() < 1e-12 and roots.residuals.max() < 1e-12

    # [ct-beyn]
    def T(z):                                            # a nonlinear eigenproblem T(z) v = 0
        return np.array([[np.exp(z) - 2, 1.0], [0.0, z * z + 1]])

    def dT(z):
        return np.array([[np.exp(z), 0.0], [0.0, 2 * z]])

    r = cn.contour.beyn(T, cn.box(re=(-1, 1), im=(-2, 2)), poles=0, derivative=dT)
    print(r.explain())                                   # roots log 2, i, -i
    # [/ct-beyn]
    exact = np.array([np.log(2), 1j, -1j])
    assert r.count == 3 and all(np.min(np.abs(r.values - z)) < 1e-12 for z in exact)

    # [ct-refuse]
    from conatus.errors import ConvergenceError

    try:                                                 # the pair sits on the edge Re mu = 0
        cn.contour.count(H, cn.box(re=(0.0, 0.5), im=(0.25, 1.75)))
    except ConvergenceError as err:
        print(err.message)                               # refused, never a guessed integer
    # [/ct-refuse]
    else:
        raise AssertionError("an eigenvalue on the boundary was counted")


if __name__ == "__main__":
    contour()
    print("contour guide snippets: PASS")
