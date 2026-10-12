"""Native free-slip walls on a Chebyshev axis: ``m.at("walls", u=cn.free_slip)``.

One vector condition B(u) = (I - n n)·dz(u) + n (n·u) = 0 per wall (model/conditions.py), so the
vector tau machinery applies unchanged. Validation: the linear onset of Rayleigh-Benard between
stress-free isothermal walls (Rayleigh 1916; Chandrasekhar 1961, ch. II): Ra_c = 27 pi^4 / 4 at
k_c = pi / sqrt(2), and the exact growth rate of the mode n = 1 (``exact_growth``).
"""
from __future__ import annotations

import numpy as np
import pytest

import conatus as cn
from conatus.errors import BoundaryConditionError

RA_C = 27 * np.pi ** 4 / 4          # 657.5113644795163
K_C = np.pi / np.sqrt(2)


def exact_growth(k: float, Ra: float, Pr: float, n: int = 1, depth: float = 1.0) -> float:
    """Growth rate s of the mode w = sin(n pi z / d) exp(i k x) of RB between stress-free
    isothermal walls in thermal-diffusion units (d = 1):
        dt(u) + grad(p) - Pr lap(u) - Pr Ra T ez = 0,  div(u) = 0,  dt(T) - lap(T) - w = 0.
    With K^2 = k^2 + (n pi / d)^2, curl curl of momentum and the heat equation give
        (s + Pr K^2) K^2 W = Pr Ra k^2 Theta,   (s + K^2) Theta = W
    so (s/Pr + K^2)(s + K^2) K^2 = Ra k^2, i.e. s^2 + (1 + Pr) K^2 s + Pr K^4 - Pr Ra k^2 / K^2 = 0;
    the larger root is returned. s = 0 gives Ra = K^6 / k^2, minimal at k^2 = pi^2 / 2:
    Ra_c = 27 pi^4 / 4."""
    K2 = k * k + (n * np.pi / depth) ** 2
    B, C = (1 + Pr) * K2, Pr * K2 * K2 - Pr * Ra * k * k / K2
    return (-B + np.sqrt(B * B - 4 * C)) / 2


def rb_free_slip(basis: str = "chebyshev", Ra: float = RA_C, Pr: float = 1.0, k: float = K_C):
    """Rayleigh-Benard between stress-free isothermal walls, written for the temperature
    perturbation theta about conduction (homogeneous conditions): the same text runs with
    basis="chebyshev" (taus) and basis="sincos" (parities)."""
    box = cn.Box(x=cn.periodic(2 * np.pi / k), z=cn.walls(0, 1, basis=basis))
    m = cn.Model(box, name="RB free-slip", Ra=Ra, Pr=Pr)
    u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*T*ez = -u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) - u·ez = -u·grad(T)
    """)
    m.at("walls", u=cn.free_slip, T=0)
    return m, u, p, T


def _top(res) -> complex:
    return res.values[np.argmax(res.values.real)]


def test_free_slip_explain_and_taus():
    m, *_ = rb_free_slip()
    text = m.explain(full=True)
    assert "u: free-slip (u·n = 0, dz(u_t) = 0) at walls (left: z=0): tau_u1" in text
    assert "u: free-slip (u·n = 0, dz(u_t) = 0) at walls (right: z=1): tau_u2" in text
    assert "divergence tau dot(ez, lift(tau_u2, z, -1))" in text
    assert "at walls: u = cn.free_slip" in text                    # as written
    comp = m.compile((8, 16))
    assert [t.name for t in comp.taus] == ["tau_u1", "tau_u2", "tau_T1", "tau_T2"]
    assert all(t.target == 0 for t in comp.taus[:2])               # momentum (order 2 in u)


def test_free_slip_rank_and_conditions_hold():
    """Square, full-rank groups; the computed onset mode satisfies u·n = 0 and dz(u_x) = 0."""
    m, u, p, T = rb_free_slip(Ra=RA_C * 1.2)
    lin = m.linearize(about={u: 0, T: 0})
    plan = lin.discretize(16, k=K_C)
    g = lin.group(plan, K_C)
    rep = plan.rank_report(g)
    assert rep.ok and "full" in rep.numerical, rep
    res = lin.eigenvalues(16, k=K_C)
    i = np.argmax(res.values.real)
    assert abs(res.values[i] - exact_growth(K_C, RA_C * 1.2, 1.0)) < 1e-11
    assert res.residuals[i] < 1e-10


@pytest.mark.parametrize("Pr", [1.0, 0.7])
def test_free_slip_onset_chebyshev(Pr):
    """Sign change across Ra_c and the exact growth rate, spectrally (N = 16)."""
    errs = []
    for Ra in (0.99 * RA_C, 1.01 * RA_C, 2000.0):
        m, u, p, T = rb_free_slip(Ra=Ra, Pr=Pr)
        s = _top(m.linearize(about={u: 0, T: 0}).eigenvalues(16, k=K_C))
        assert abs(s.imag) < 1e-10
        errs.append(abs(s - exact_growth(K_C, Ra, Pr)))
        assert (s.real > 0) == (Ra > RA_C)
    assert max(errs) < 1e-11, errs


def test_no_slip_sentinel_is_zero_vector():
    m1, *_ = rb_free_slip()
    m1.conditions.clear()
    m1.at("walls", u=cn.no_slip, T=0)
    m2, *_ = rb_free_slip()
    m2.conditions.clear()
    m2.at("walls", u=0, T=0)
    a, b = m1.discretize((8, 12)), m2.discretize((8, 12))
    for g1, g2 in zip(a.groups, b.groups):
        assert np.array_equal(a.matrices(g1).L.toarray(), b.matrices(g2).L.toarray())
    assert "u = 0 (no-slip)" in m1.explain()


def test_free_slip_refusals():
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1)), name="heat")
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    m.at("walls", T=cn.free_slip)
    with pytest.raises(BoundaryConditionError, match="condition on a velocity"):
        m.compile((8, 8))
    # curved walls (disk, annulus, pipe, ball, shell) are supported since
    # tests/model/test_free_slip_curved.py


def test_component_condition_message_points_to_free_slip():
    m, *_ = rb_free_slip()
    m.conditions.clear()
    m.at("walls", "dz(u@ex) = 0", T=0)
    with pytest.raises(BoundaryConditionError, match="cn.free_slip"):
        m.compile((8, 8))
