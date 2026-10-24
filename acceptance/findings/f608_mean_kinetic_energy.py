"""F608 (audit of 0.1.2, F08): the mean kinetic energy of a vector field is (1/V) ∫ |u|²/2 dV:
components summed, then the volume average by quadrature (the reduction the Studio guide shows).

Oracle: u = (1, 0): 1/2; u = (0, z) on z in [-1/2, 1/2] (Chebyshev): (1/V) ∫ z²/2 = 1/24."""
from _acceptance import check, passed, start

start(608, "mean kinetic energy of a vector field")
import numpy as np  # noqa: E402

from conatus.diagnostics import Chebyshev, Grid, Periodic  # noqa: E402

quad = Grid([Periodic(2.0, 16), Chebyshev(-0.5, 0.5, 12)])    # x periodic, z between walls


def kinetic(u):
    return float(quad.mean(0.5 * np.sum(u ** 2, axis=0)))


x, z = quad.mesh()
e1 = kinetic(np.stack([np.ones_like(x), np.zeros_like(x)]))
e2 = kinetic(np.stack([np.zeros_like(x), z]))
check(abs(e1 - 0.5) < 1e-13, f"u = (1, 0): {e1} instead of 1/2")
check(abs(e2 - 1 / 24) < 1e-13, f"u = (0, z): {e2} instead of 1/24")
passed(f"{e1:.15g} and {e2:.15g}")
