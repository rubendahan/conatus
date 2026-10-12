""" on curved geometries (native layer, CPU).

* Moving wall data on the disk and in the spherical shell: the default ``bc_time="auto"`` (stage
  values from the scheme applied to the exact g'(t)) gives RK443 its order (disk 2.98, shell 2.85,
  dt = 0.01 -> 0.005 -> 0.0025), "stage" 2.1-1.9, "step" (Dedalus) 1.1. Exact solutions:
  T = exp(-K^2 t) J0(K r) on the unit disk, T = exp(-K^2 t) sin(K r) / r in the shell 1 < r < 2.
* A22: a given function of (phi, r) in the right side of a wall condition used to fail at the
  first step ("cannot evaluate node Interpolate in a polar RHS"); its trace at the wall is now
  evaluated from the function (exact: bit-identical to declaring g(phi, R) by hand), declared in
  explain(); under a derivative (dr(g)) it is refused with what to declare instead.
"""
import numpy as np
import pytest
from scipy.special import j0

import conatus as cn
from conatus.errors import BoundaryConditionError

K = 2.0


def _disk(dt, bc_time, scheme="RK443"):
    m = cn.Model(cn.Disk(1.0))
    m.scalar("T")
    m.equations("heat: dt(T) - lap(T) = 0")
    m.at("wall", T=lambda t: float(j0(K)) * np.exp(-K**2 * t))
    sim = m.simulate(N=(4, 16), scheme=scheme, dt=dt, bc_time=bc_time)
    sim.initial(T=lambda phi, r: j0(K * r) + 0 * phi)
    sim.run(until=0.5)
    r = np.asarray(sim.coordinate("r"))
    return float(np.abs(sim.grid("T") - np.exp(-K**2 * sim.t) * j0(K * r)[None, :]).max()), sim


def _shell(dt, bc_time):
    m = cn.Model(cn.Shell(1.0, 2.0))
    m.scalar("T")
    m.equations("heat: dt(T) - lap(T) = 0")
    m.at("inner", T=lambda t: np.sin(K) * np.exp(-K**2 * t))
    m.at("outer", T=lambda t: np.sin(2 * K) / 2 * np.exp(-K**2 * t))
    sim = m.simulate(N=(4, 4, 16), scheme="RK443", dt=dt, bc_time=bc_time)
    sim.initial(T=lambda phi, theta, r: np.sin(K * r) / r + 0 * phi + 0 * theta)
    sim.run(until=0.5)
    r = np.asarray(sim.coordinate("r")).ravel()
    return float(np.abs(sim.grid("T") - np.exp(-K**2 * sim.t) * np.sin(K * r) / r).max()), sim


@pytest.mark.parametrize("geo", ["disk", "shell"])
def test_curved_moving_wall_orders(geo):
    run = _disk if geo == "disk" else _shell
    orders = {}
    for bc_time in ("auto", "stage"):
        (e1, s), (e2, _) = run(0.01, bc_time), run(0.005, bc_time)
        orders[bc_time] = np.log2(e1 / e2)
        assert s.solver.bc_time == ("rk" if bc_time == "auto" else "stage")
    assert orders["auto"] > 2.7 and orders["stage"] < 2.4, orders
    if geo == "disk":
        e1, e2 = _disk(0.01, "step")[0], _disk(0.005, "step")[0]
        assert np.log2(e1 / e2) < 1.3                          # Dedalus convention: 1.09


def _wall_heat(cond, given):
    m = cn.Model(cn.Disk(1.0))
    m.scalar("T")
    m.given(**given)
    m.equations("heat: dt(T) - lap(T) = 0")
    m.at("wall", cond)
    sim = m.simulate(N=(8, 12), scheme="SBDF2", dt=1e-3)
    sim.run(steps=3)
    return sim


def test_given_function_of_phi_and_r_at_the_wall_is_its_exact_trace():
    a = _wall_heat("T = g", dict(g=lambda phi, r: r**3 * np.cos(phi) + r**2))
    b = _wall_heat("T = gw", dict(gw=lambda phi: np.cos(phi) + 1.0))
    ca, cb = a.solver.coeffs("T"), b.solver.coeffs("T")
    assert np.abs(ca).max() > 0.1 and np.array_equal(ca, cb)
    assert "its trace at r = 1 evaluated from its function (exact)" in a.explain(full=True)
    c = _wall_heat("T = g*cos(t)", dict(g=lambda phi, r: r**3 * np.cos(phi) + r**2))
    assert c.solver.bc_time == "rk"                         # with the time: exact derivative too
    with pytest.raises(BoundaryConditionError, match="appears under an operator") as ei:
        _wall_heat("T = dr(g)", dict(g=lambda phi, r: r**3 * np.cos(phi)))
    assert "m.given(g_wall=lambda phi: ...)" in str(ei.value)
