"""A linear boundary-value solve states whether its solution is resolved, and the values of a call
that lower the order are refused. The oracle is the exact solution of nu u'' - u = 1 with u = 0 at
0 and 1: u = -1 + cosh((z - 1/2)/sqrt(nu)) / cosh(1/(2 sqrt(nu)))."""
import warnings

import numpy as np
import pytest

import conatus as cn
from conatus.errors import BoundaryConditionError, UnderResolvedWarning
from conatus.resolution import SOLVE_TAIL_TOL


def layer(nu, N):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="layer", nu=nu)
    m.scalar("u")
    m.equations("nu*dz(dz(u)) - u = 1")
    m.at("walls", u=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = m.solve(N)
    return sol, [x for x in w if issubclass(x.category, UnderResolvedWarning)]


def exact(z, nu):
    s = np.sqrt(nu)
    return -1 + np.exp(-z / s) + np.exp((z - 1) / s) if nu < 1e-3 else (
        -1 + np.cosh((z - 0.5) / s) / np.cosh(0.5 / s))


@pytest.mark.parametrize("nu, N", [(1e-10, 32), (1e-10, 16), (1e-6, 32), (1e-5, 32)])
def test_thin_layer_is_stated_under_resolved(nu, N):
    sol, w = layer(nu, N)
    err = np.abs(sol.grid("u") - exact(sol.coordinate("z"), nu)).max()
    assert sol.max_residual < 1e-10                  # the residual cannot tell
    assert err > SOLVE_TAIL_TOL ** 2                 # and the solution is really off
    assert w
    assert "UNDER-RESOLVED" in sol.explain()


@pytest.mark.parametrize("nu, N", [(1e-4, 128), (1e-2, 64), (1.0, 32)])
def test_resolved_layer_is_not_warned(nu, N):
    sol, w = layer(nu, N)
    assert np.abs(sol.grid("u") - exact(sol.coordinate("z"), nu)).max() < 1e-10
    assert not w
    assert "largest spectral tail" in sol.explain()


def test_nothing_is_changed_by_the_statement():
    sol, _ = layer(1e-10, 32)
    assert sol.plan.domain.basis(sol.plan.domain.coordsys["z"]).size == 32


def test_small_grid_is_not_judged_and_says_so():
    sol, w = layer(1.0, 4)
    assert not w
    assert "not judged" in sol.explain()


def drop(eq, **params):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="drop", **params)
    m.scalar("u")
    m.equations(eq)
    m.at("walls", u=0)
    return m


@pytest.mark.parametrize("eq, params", [
    ("nu*dz(dz(u)) - u = 1", dict(nu=0.0)),
    ("kappa*dz(dz(u)) - u = 1", dict(kappa=0.0)),                     # not a name
    ("a*dz(dz(u)) + b*dz(dz(u)) - u = 1", dict(a=2.0, b=-2.0)),       # cancels
])
def test_order_drop_is_refused(eq, params):
    with pytest.raises(BoundaryConditionError, match="drops"):
        drop(eq, **params).solve(32)


def test_order_drop_from_two_to_one():
    with pytest.raises(BoundaryConditionError, match="drops from 2 to 1"):
        drop("a*dz(dz(u)) + dz(u) = 1", a=0.0).solve(32)


def test_override_at_call_is_read():
    m = drop("nu*dz(dz(u)) - u = 1", nu=1.0)
    with pytest.raises(BoundaryConditionError, match="drops"):
        m.solve(32, nu=0.0)
    assert m.solve(32, nu=1.0).max_residual < 1e-10


def test_order_drop_on_an_evolution_model():
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="ivp", nu=0.0)
    m.scalar("u")
    m.equations("dt(u) - nu*dz(dz(u)) = 1")
    m.at("walls", u=0)
    with pytest.raises(BoundaryConditionError, match="drops"):
        m.compile(16)


def test_annulus_tail_is_measured():
    m = cn.Model(cn.Annulus(1.0, 2.0), name="ann")
    m.scalar("p")
    m.equations("lap(p) = 1")
    m.at("walls", p=0)
    sol = m.solve((4, 32))
    assert "largest spectral tail" in sol.explain()


def test_mapped_solve_says_it_is_not_judged():
    """A mapped domain stores flat coefficient vectors, so the check cannot read them. It says so."""
    m = cn.Model(cn.MappedBox(x=cn.periodic(2 * np.pi),
                              z=cn.between(bottom=lambda x: 0.05 * np.cos(x), top=1.0)), name="mapped")
    m.scalar("u")
    m.equations("lap(u) = 1")
    m.at("bottom", u=0)
    m.at("top", u=0)
    assert "not judged, no mode numbers" in m.solve((16, 16)).explain()


# ------------------------------------------------------------------- independent review
def forced(f, N):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="forced")
    m.scalar("u")
    m.given(f=f)
    m.equations("dz(dz(u)) = f")
    m.at("walls", u=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = m.solve(N)
    return sol, [x for x in w if issubclass(x.category, UnderResolvedWarning)]


def test_a_step_in_the_data_is_stated_though_the_solution_tail_is_small():
    """u'' = sign(z - a), u(0) = u(1) = 0, has the piecewise parabola below as limit of the thin tanh.
    At N = 64 the values are off by more than 1 %, while the coefficient tail of u stays below the
    level: integrating the data smooths it."""
    a, width = 0.37, 1e-3
    sol, w = forced(lambda z: np.tanh((z - a) / width), 64)
    A = (a - 1) ** 2 / 2 + a ** 2 / 2 + (1 - 2 * a) * (a - 1)
    B = A - 2 * a + 1
    z = sol.coordinate("z")
    exact_u = np.where(z < a, -z ** 2 / 2 + A * z, (z - 1) ** 2 / 2 + B * (z - 1))
    assert np.abs(sol.grid("u") - exact_u).max() > 0.01 * np.abs(exact_u).max()
    assert w
    assert "data of equation" in sol.explain()


def test_smooth_data_is_not_warned():
    sol, w = forced(lambda z: np.sin(4 * np.pi * z), 32)
    assert not w
    sol, w = forced(lambda z: np.exp(-((z - 0.5) / 0.02) ** 2), 128)   # data tail 4e-3, u tail 1e-6
    assert not w


def test_round_off_data_beside_real_data_is_not_judged():
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="noise")
    m.scalar("u")
    m.scalar("v")
    rng = np.random.default_rng(1)
    m.given(n=lambda z: 1e-18 * rng.standard_normal(z.shape), f=lambda z: np.sin(np.pi * z))
    m.equations("dz(dz(u)) = n", "dz(dz(v)) = f")
    m.at("walls", u=0, v=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnderResolvedWarning)
        m.solve(32)


def test_an_unknown_too_small_to_judge_is_named():
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="two", nu=1e-6)
    m.scalar("u")
    m.scalar("v")
    m.given(f=lambda z: 1e-8 + 0 * z)
    m.equations("dz(dz(u)) = 1e6", "nu*dz(dz(v)) - v = f")
    m.at("walls", u=0, v=0)
    assert "below 1e-10 of the largest unknown: v" in m.solve(32).explain()


@pytest.mark.parametrize("eq", ["dz(nu*dz(u)) - u = 1", "dz(dz(nu*u)) - u = 1"])
def test_order_drop_under_a_derivative(eq):
    with pytest.raises(BoundaryConditionError, match="drops"):
        drop(eq, nu=0.0).solve(32)


@pytest.mark.parametrize("K", [0.0, lambda z: 0 * z, lambda z: np.zeros_like(z)])
def test_zero_given_lowers_the_order(K):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="zero")
    m.scalar("u")
    m.given(K=K)
    m.equations("K*dz(dz(u)) - u = 1")
    m.at("walls", u=0)
    with pytest.raises(BoundaryConditionError, match="drops"):
        m.solve(32)


@pytest.mark.parametrize("K", [lambda z: 1 + z, lambda z: 0.0 * z + 2.0])
def test_nonzero_given_is_kept(K):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="nz")
    m.scalar("u")
    m.given(K=K)
    m.equations("K*dz(dz(u)) - u = 1")
    m.at("walls", u=0)
    assert m.solve(32).max_residual < 1e-10


def evp_nu(nu):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="evp", nu=nu)
    m.scalar("u")
    m.eigenvalue("s", growth="real")
    m.equations("s*u - nu*dz(dz(u)) = 0")
    m.at("walls", u=0)
    return m


def nearest(result, target):
    e = np.asarray(result.values)
    return e[np.argmin(np.abs(e - target))]


def test_pencil_sweeps_towards_zero():
    """A pencil is built at a non-zero nominal value (a zero one was refused before this check too)
    and later points are not compiled, so a sweep towards nu = 0 keeps working. Oracle: the Dirichlet
    Laplacian on [0, 1], s = -nu pi^2."""
    P = evp_nu(1.0).parametric(24, over="nu")
    for nu in (1.0, 0.5, 1e-3):
        assert abs(nearest(P.solve(nu=nu), -nu * np.pi ** 2) + nu * np.pi ** 2) < 1e-6 * max(nu, 1e-3)
    with pytest.raises(BoundaryConditionError, match="drops"):       # no nominal value at 0
        evp_nu(0.0).parametric(24, over="nu")


def test_a_model_point_at_zero_is_refused_and_the_next_point_runs():
    m = evp_nu(1.0)
    with pytest.raises(BoundaryConditionError, match="drops"):
        m.eigenvalues(24, nu=0.0)
    assert abs(nearest(m.eigenvalues(24, nu=0.5), -0.5 * np.pi ** 2) + 0.5 * np.pi ** 2) < 1e-8


def test_linearized_pencil_from_a_zero_rayleigh_point():
    """A linearized sweep declared at Ra = 1 starts at Ra = 0 without trouble: Ra is not on a highest
    derivative, nothing drops. Oracle: Reid & Harris onset, marginal mode at Ra_c = 1707.762, k = 3.117."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi / 3.117, complex=True), z=cn.walls(-0.5, 0.5)),
                 name="rb", Ra=1.0, Pr=1.0)
    m.velocity("u")
    m.pressure("p")
    m.scalar("T")
    m.equations("dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u@grad(u)", "div(u) = 0",
                "dt(T) - lap(T) = -u@grad(T)")
    m.at("bottom", u=0, T=0.5)
    m.at("top", u=0, T=-0.5)
    P = m.linearize(about={"u": 0, "T": lambda z: -z}).parametric(24, over="Ra", k=3.117)
    assert np.all(np.real(P.solve(Ra=0.0).values)[np.isfinite(P.solve(Ra=0.0).values)] < 1e-6)
    assert abs(nearest(P.solve(Ra=1707.762), 0.0)) < 1e-2

