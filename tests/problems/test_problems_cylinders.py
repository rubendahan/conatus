"""conatus.problems cylinders: Hagen--Poiseuille eigenvalues (Meseguer & Trefethen 2003),
Taylor--Couette onset (Recktenwald et al. 1993), laminar pipe flow steady in the IVP."""
import warnings

import numpy as np
import pytest

from conatus import problems as P
from conatus.discretize import discretize
from conatus.solvers import eigenvalues
from conatus.timestep import IVPSolver

MT_CENTRE, MT_WALL = -0.0708640053 + 0.8467498288j, -0.0911426036 + 0.4691428789j
MT_K0 = -0.0146819706


def test_pipe_meseguer_trefethen():
    """Re = 1000: (1, 1) centre and wall modes (conjugates of Table 1), (1, 0) least-stable."""
    r = P.pipe_evp(N=32, Re=1000.0, m=1, k=1.0)
    plan = discretize(r.problem, ncc=r.ncc)
    res = eigenvalues(plan, plan.group(1, 1.0))
    for ref in (MT_CENTRE, MT_WALL):
        assert np.min(np.abs(res.values - np.conj(ref))) < 1e-10
    assert res.residuals.max() < 1e-12
    r0 = P.pipe_evp(N=32, Re=1000.0, m=1, k=1.0)
    plan0 = discretize(r0.problem, ncc=r0.ncc)
    v0 = eigenvalues(plan0, plan0.group(1, 0.0)).values
    assert abs(v0[np.argmax(v0.real)] - MT_K0) < 1e-10


def _tc_growth(Re, k=3.16, N=24):
    r = P.taylor_couette_evp(N=N, Re=Re, eta=0.5, mu=0.0, m=0, k=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = discretize(r.problem, ncc=r.ncc)
    return eigenvalues(plan, plan.group(0, k)).values.real.max()


def test_taylor_couette_brackets_onset():
    """Re_c(k = 3.16) = 68.19 (Recktenwald et al. 1993): stable at 68.0, unstable at 68.4."""
    assert _tc_growth(68.0) < 0 < _tc_growth(68.4)


@pytest.mark.slow
def test_taylor_couette_critical_reynolds():
    """Neutral Re at k = 3.10, 3.16, 3.22: the minimum is at 3.16 and equals 68.19 (4 digits),
    N = 24 as in tutorial t08; N = 32 agrees to 1e-6."""
    from scipy.optimize import brentq
    Re_c = {k: brentq(lambda Re: _tc_growth(Re, k), 60.0, 80.0, xtol=1e-8) for k in (3.10, 3.16, 3.22)}
    fine = brentq(lambda Re: _tc_growth(Re, 3.16, N=32), 60.0, 80.0, xtol=1e-8)
    print("Taylor-Couette neutral Re:", Re_c, "N = 32 at k = 3.16:", fine)
    assert min(Re_c, key=Re_c.get) == 3.16
    assert abs(Re_c[3.16] - 68.19) < 0.005
    assert abs(fine - Re_c[3.16]) < 1e-6


def test_pipe_ivp_keeps_hagen_poiseuille():
    """W = 1 - r^2 with G = 4/Re is a steady state of the pipe IVP."""
    Re = 100.0
    r = P.pipe(Nphi=4, Nr=8, Nz=4, Lz=2 * np.pi, Re=Re, pressure_gradient=4 / Re, dealias=1.5)
    s = IVPSolver(discretize(r.problem), "RK443", 0.01)
    s.set_state({r.fields["u"]: lambda Ph, R, Z: np.array([0 * R, 0 * R, 1 - R ** 2]) + 0 * Ph + 0 * Z})
    g0 = s.grid("u").copy()
    s.run(10)
    assert np.abs(g0).max() > 0.5
    assert np.abs(s.grid("u") - g0).max() < 1e-12


def test_taylor_couette_ivp_pencil_matches_evp():
    """The IVP (continuity-closed tau form) linearized in the group (m, k) = (0, pi) has the
    spectrum of taylor_couette_evp (tutorial t08 form): three least-stable modes at Re = 68."""
    import scipy.linalg as sla
    kw = dict(Re=68.0, eta=0.5, mu=0.0)
    r = P.taylor_couette(Nphi=4, Nr=24, Nz=4, Lz=2.0, dealias=1.5, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = discretize(r.problem, ncc=r.ncc)
        gm = plan.matrices(plan.group(0, np.pi))
        e = P.taylor_couette_evp(N=24, m=0, k=np.pi, **kw)
        eplan = discretize(e.problem, ncc=e.ncc)
        ref = eigenvalues(eplan, eplan.group(0, np.pi)).values
    w = sla.eigvals(-gm.L.toarray(), gm.M.toarray())      # growth rates of M dX/dt + L X = 0
    w = w[np.isfinite(w)]
    top, ref_top = w[np.argsort(-w.real)[:3]], ref[np.argsort(-ref.real)[:3]]
    assert np.abs(np.sort(top.real) - np.sort(ref_top.real)).max() < 1e-8, (top, ref_top)
    s = IVPSolver(plan, "RK443", 0.01)                     # and the IVP factorizes and runs
    s.run(2)
