"""conatus.problems periodic models: exact linear growth rates (Kuramoto--Sivashinsky,
Swift--Hohenberg), Kolmogorov Re_lin (Chandler & Kerswell 2013), KS L = 22 equilibria (slow)."""
import numpy as np
import pytest

from conatus import problems as P
from conatus.discretize import discretize
from conatus.ir import Parameter, Zero
from conatus.linearize import linearize
from conatus.solvers import eigenvalues
from conatus.timestep import IVPSolver


def _group_rates(r):
    """{wavenumber tuple: eigenvalues} of the linearization about u = 0."""
    evp = linearize(r.problem, around={r.fields["u"]: Zero(())},
                    eigenvalue=Parameter("s", dtype=complex))
    plan = discretize(evp)
    return {tuple(gi.k for gi in g): eigenvalues(plan, g).values for g in plan.groups}


def test_kuramoto_sivashinsky_linear_rates_exact():
    L = 22.0
    rates = _group_rates(P.kuramoto_sivashinsky(N=32, L=L, dealias=1.5))
    assert len(rates) == 16
    for (k,), v in rates.items():
        assert np.abs(v - (k ** 2 - k ** 4)).max() < 1e-12 * max(1, k ** 4), (k, v)
    assert any(abs(k - 2 * np.pi / L) < 1e-14 for (k,) in rates)


def test_swift_hohenberg_linear_rates_exact():
    r = 0.1
    rates = _group_rates(P.swift_hohenberg(Nx=8, Ny=8, Lx=2 * np.pi, Ly=4 * np.pi, r=r, dealias=2))
    for (kx, ky), v in rates.items():
        K2 = kx ** 2 + ky ** 2
        assert np.abs(v - (r - (1 - K2) ** 2)).max() < 1e-12 * max(1, K2 ** 2), (kx, ky, v)
    growing = [k for k, v in rates.items() if v.real.max() > 0]
    assert all(abs(np.hypot(*k) - 1) < 0.35 for k in growing) and (1.0, 0.0) in growing


def _kolmogorov_lead(Re, N=24):
    from conatus.ecs.kolmogorov import laminar_state, linear_stability_dense
    r = P.kolmogorov(N=N, n=4, Re=Re, dealias=1.5)
    s = IVPSolver(discretize(r.problem, ncc=r.ncc), "RK443", 0.01)
    X = laminar_state(s, Re=Re, n=4)
    assert np.abs(s.F(X, 0.0) - s.system.L @ X).max() < 1e-12      # laminar state is steady
    return linear_stability_dense(s, X, dense_max=2000, finite_max=1e8)[0]


@pytest.mark.slow                   # 5.7 s in the fast suite (qual_durations0)
def test_kolmogorov_re_lin_chandler_kerswell():
    """Re_lin = 9.9669 (CK13): sign change of the (real) leading eigenvalue in [9.96685, 9.96695]."""
    lo, hi = _kolmogorov_lead(9.96685), _kolmogorov_lead(9.96695)
    assert lo.real < 0 < hi.real
    assert abs(lo.imag) < 1e-10 and abs(hi.imag) < 1e-10


@pytest.mark.slow
def test_kuramoto_sivashinsky_l22_equilibria():
    """E1, E2 energies 0.2609, 0.4382 (Cvitanovic, Davidchack & Siminos 2010, Table 5.1),
    Newton--Krylov--hookstep on the flow map of this IVP (as tutorial t05)."""
    from conatus.ecs import NewtonKrylovHookstep, find_equilibrium
    N = 32
    r = P.kuramoto_sivashinsky(N=N, L=22.0, dealias=1.5)
    ks = IVPSolver(discretize(r.problem), "RK443", 0.01)
    out = {}
    for m, ref in ((1, 0.2609), (2, 0.4382)):
        c = np.zeros(N)
        c[2 * m + 1] = -1.0                                   # u0 = sin(2 pi m x / L)
        nk = NewtonKrylovHookstep(tol=1e-12, max_iterations=40, gmres_rtol=1e-3, gmres_max_iterations=60,
                                  delta0=1.0, delta_min=1e-10, delta_max=10.0)
        sol = find_equilibrium(ks, ks.system.gather_state([c]), nk)
        out[m] = (sol.energy, sol.residual)
        assert sol.residual < 1e-12 and abs(sol.energy - ref) < 6e-5, (m, sol.energy)
    print("KS L = 22 equilibria (energy, residual):", out)
