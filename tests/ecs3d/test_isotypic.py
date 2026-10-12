"""Isotypic spectrum (D7) on laminar Couette: Arnoldi on the exact tangent of the discrete
time-T map, restricted to each isotypic component of S, gives the eigenvalues of the compiled
linear operator (problem linearized about U = y, all Fourier groups); the tangent map does not
leak out of the components."""
import numpy as np
import pytest
import scipy.linalg as sl

from couette_cases import BOX_GHC, couette_pt, shift_reflect_group
from conatus.ecs.steady import isotypic_spectrum
from conatus.timestep import IVPSolver


@pytest.mark.slow
def test_isotypic_spectrum_of_laminar_couette():
    N, Re, T, n = (6, 13, 6), 400.0, 0.5, 10   # |Im λ| T < π: no branch ambiguity of log μ
    plan, f = couette_pt(*N, Re, BOX_GHC)
    s = IVPSolver(plan, "RK443", T / n, f_backend="fused", decouple=True)
    s.set_state({f["U"]: lambda X, Y, Z: Y + 0 * X})
    G = shift_reflect_group(s)
    sp = isotypic_spectrum(s, s.X.copy(), G, T=T, nsteps=n, k=3, tol=1e-10)
    pl, _ = couette_pt(*N, Re, BOX_GHC, linear_about="couette")
    ev = []
    for g in pl.groups:
        m = pl.matrices(g)
        w = sl.eigvals(-m.L.toarray(), m.M.toarray())
        ev += list(w[np.isfinite(w) & (np.abs(w) < 1e3)])
    ev = np.array(ev)
    for ch, lam in sp["eigenvalues"].items():
        assert sp["leak"][ch] < 1e-12
        for z in lam:          # every Arnoldi eigenvalue is one of the operator's (dt^3 error)
            assert np.min(np.abs(ev - z)) < 1e-6 * max(1.0, abs(z))
    lead = max(z.real for lam in sp["eigenvalues"].values() for z in lam)
    assert abs(lead - ev.real.max()) < 1e-6
