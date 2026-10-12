"""sim.properties (role of Dedalus' GlobalFlowProperty): reductions of declared grid quantities.

Exact checks on a 2D box periodic in x, walls in z: for T = (1 + cos x)(1 - z^2) the volume average
over [0, 2pi) x [-1, 1] is 2/3 exactly (quadrature exact on these polynomials and trigonometric
terms), max <= 2 (reached at x = 0, z = 0, not a Gauss node); grid_average differs (non-uniform Chebyshev
points) and is reported as such. The cadence follows the run; nothing changes the state.
"""
import numpy as np
import pytest

import conatus as cn
from conatus.errors import InputTypeError, OptionError, UnsupportedError


def _sim():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(-1, 1)), name="heat 2D", kappa=0.1)
    m.scalar("T")
    m.equations("dt(T) - kappa*lap(T) = 0")
    m.at("walls", T=0)
    sim = m.simulate(N=(16, 16), scheme="RK222", dt=1e-3)
    sim.initial(T=lambda x, z: (1 + np.cos(x)) * (1 - z**2))
    return sim


def test_reductions_are_exact_on_the_initial_state():
    sim = _sim()
    p = sim.properties(every=5, T="T", T2=lambda s: s.grid("T") ** 2)
    # (1 + cos x)(1 - z^2): mean over x is 1, over z of 1 - z^2 is 2/3 -> volume average 2/3
    assert p.volume_average("T") == pytest.approx(2 / 3, abs=1e-14)
    assert p.integral("T") == pytest.approx(2 / 3 * 4 * np.pi, abs=1e-13)
    # T^2 = (1 + cos x)^2 (1 - z^2)^2: x-mean 3/2, z-mean (16/15)/2 = 8/15 -> 4/5
    assert p.volume_average("T2") == pytest.approx(0.8, abs=1e-14)
    assert p.max("T") <= 2.0 and p.max("T") > 1.9 and p.min("T") > -1e-15
    assert abs(p.grid_average("T") - 2 / 3) > 1e-3       # not a volume average (declared)
    assert p.evaluated_at == (0, 0.0)


def test_cadence_and_no_effect_on_the_run():
    ref = _sim()
    ref.run(steps=12)
    sim = _sim()
    p = sim.properties(every=5, T="T")
    sim.run(steps=12)
    assert p.evaluated_at[0] == 10                       # last multiple of 5
    assert np.array_equal(sim.solver.X, ref.solver.X)
    assert "every 5 steps" in p.explain() and "FlowProperties" in sim.explain()
    v10 = p.volume_average("T")
    p.evaluate()
    assert p.evaluated_at[0] == 12 and p.volume_average("T") < v10      # decaying


def test_refusals():
    sim = _sim()
    with pytest.raises(InputTypeError, match="not an unknown"):
        sim.properties(every=1, u="u")
    with pytest.raises(InputTypeError, match="no property"):
        sim.properties(every=1)
    with pytest.raises(OptionError, match="integer >= 1"):
        sim.properties(every=0, T="T")
    p = sim.properties(every=1, T="T")
    with pytest.raises(InputTypeError, match="not declared"):
        p.max("S")
    m = cn.Model(cn.Disk(1.0), name="disk heat")
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    m.at("wall", T=0)
    ds = m.simulate(N=(8, 8), scheme="RK222", dt=1e-3)
    ds.initial(T=lambda phi, r: 1 - r**2)
    q = ds.properties(every=1, T="T")
    assert q.max("T") <= 1.0
    with pytest.raises(UnsupportedError, match="quadrature"):
        q.volume_average("T")
