"""conatus.problems convection: Rayleigh--Benard onset (Reid & Harris 1958) and growth rate
(tutorial t02a), rotating shell onset (Christensen et al. 2001 geometry; slow)."""
import warnings

import numpy as np
import pytest

from conatus import problems as P
from conatus.discretize import discretize
from conatus.solvers import ParametricEVP, critical, eigenvalues


def _rb_growth(N, Ra, k):
    r = P.rayleigh_benard_evp(N=N, Ra=Ra, Pr=1.0, k=k)
    v = eigenvalues(discretize(r.problem)).values
    v = v[np.isfinite(v) & (np.abs(v) < 1e4)]
    return v[np.argmax(v.real)]


def test_rayleigh_benard_growth_rate_matches_ivp_linearization():
    """sigma(Ra = 2500, k = pi, Pr = 1) = 5.5425059: tutorial t02a, obtained there by
    linearize() of the 2D IVP (a different formulation of the same physics)."""
    s = _rb_growth(24, 2500.0, np.pi)
    assert abs(s.imag) < 1e-10 and abs(s.real - 5.5425059) < 1e-6, s


def test_rayleigh_benard_onset_reid_harris():
    """Ra_c = 1707.762 at k_c = 3.117 (Reid & Harris 1958), by critical() on the pencil."""
    r = P.rayleigh_benard_evp(N=24, Ra=1700.0, Pr=1.0, k=3.1)
    pe = ParametricEVP(r.problem, [r.parameters["Ra"], r.parameters["k"]])
    crit = critical(pe, "Ra", bracket=(1600.0, 1800.0), over={"k": (3.0, 3.25)},
                    select=lambda s: np.abs(s) < 1e3, xtol=1e-10)
    assert abs(crit.value - 1707.762) < 1e-3, crit.value
    assert abs(crit.over["k"] - 3.117) < 1e-3, crit.over


@pytest.mark.slow
def test_rotating_shell_onset_christensen_geometry():
    """Ra_c = 55.904299, m = 4, omega_c = -0.0230918 Omega at (Nr, Lmax) = (32, 48) (run
   ); Skene & Tobias 2025: 55.9. Brent on the growth rate of the
    shift-invert mode nearest the drift frequency; residual reported."""
    from scipy.optimize import brentq

    from conatus.solvers.evp import solve_sparse

    def growth(Ra, out):
        r = P.rotating_shell_convection_evp(Nr=32, Lmax=48, m=4, eta=0.35, Ra=Ra, E=1e-3,
                                            Pr=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = discretize(r.problem, ncc=r.ncc)
        gm = plan.matrices(plan.group(4))
        res = solve_sparse(gm.L, gm.M, -23j, n_modes=10)
        i = int(np.argmax(res.values.real))
        out.update(s=complex(res.values[i]), residual=float(res.residuals[i]))
        return out["s"].real

    last: dict = {}
    Ra_c = brentq(lambda Ra: growth(Ra, last), 50.0, 62.0, xtol=1e-8)
    growth(Ra_c, last)
    print(f"shell onset: Ra_c = {Ra_c:.6f}, omega_c = {last['s'].imag * 1e-3:.7f} Omega, "
          f"residual {last['residual']:.1e}")
    assert abs(Ra_c - 55.904299) < 1e-5
    assert abs(last["s"].imag * 1e-3 - (-0.0230918)) < 1e-6
    assert last["residual"] < 1e-8
