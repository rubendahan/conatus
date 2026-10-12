"""Rayleigh–Bénard onset from linearize() of the nonlinear Boussinesq IVP."""
import numpy as np
import pytest

from cases import K_C, RA_C, boussinesq, rb_critical, rb_onset_evp, rb_pencil
from conatus.discretize import discretize
from conatus.ir import ProblemError, Parameter
from conatus.linearize import linearize
from conatus.solvers import eigenvalues


def test_onset_critical_rayleigh_and_wavenumber():
    Ra_c, k_c = rb_critical(24)
    assert abs(Ra_c - RA_C) < 1e-3 and abs(k_c - K_C) < 1e-4    # references given to 1e-3 / 1e-4


@pytest.mark.slow                   # 3.0 s in the fast suite (qual_durations0)
def test_onset_in_symmetry_subspace_same_value_half_size():
    L0, L1, M, gm = rb_pencil(K_C, 24, symmetric=False)
    L0s, L1s, Ms, gms = rb_pencil(K_C, 24, symmetric=True)
    assert gms.shape[0] * 2 == gm.shape[0] and gms.reduced
    Ra_s, k_s = rb_critical(24, symmetric=True)
    Ra_f, k_f = rb_critical(24, symmetric=False)
    assert abs(Ra_s - Ra_f) < 1e-8 and abs(k_s - k_f) < 1e-5


def test_linearized_problem_structure_and_explain():
    evp, T0 = rb_onset_evp(K_C)
    ivp, f = boussinesq(K_C)
    from conatus.ir import expr as E
    alt = ivp.linearize({f["u"]: E.Zero((f["cs"],)), f["T"]: f["T0"]}, Parameter("s", dtype=complex))
    assert [e.lhs for e in alt.equations] == [e.lhs for e in evp.equations]
    names = [v.name for v in evp.variables]
    assert names[:3] == ["du", "dT", "dp"]
    text = evp.explain()
    assert "Linearization of IVP 'Boussinesq RB'" in text and "u = 0" in text and "T = T0" in text
    plan = discretize(evp, ncc={T0: lambda X, Z: -Z}, params={"Ra": RA_C})
    g = next(g for g in plan.groups if g[0].k > 0)
    r = eigenvalues(plan, g)
    lead = r.values[np.argmax(r.values.real)]
    assert abs(lead) < 1e-4 and abs(lead.imag) < 1e-12 and r.residuals.max() < 1e-12


def test_missing_base_state_refused():
    ivp, f = boussinesq(K_C)
    with pytest.raises(ProblemError, match="base state"):
        linearize(ivp, around={f["T"]: f["T0"]}, eigenvalue=Parameter("s", dtype=complex))
