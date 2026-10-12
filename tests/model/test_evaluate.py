"""sim.evaluate(text): grid values of an expression of the state, written as in the equations.

On a 2D box with T = (1 + cos x)(1 - z^2) and u = (z, sin x): derivatives, products, parameters,
vector expressions, given fields, against the closed forms (round-off). The same strings are
accepted as tasks of sim.output and as sim.properties; a typo fails at declaration; curvilinear
geometries are refused.
"""
import numpy as np
import pytest

import conatus as cn
from conatus.errors import UndeclaredVariableError, UnsupportedError


def _sim(tmp_path=None):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(-1, 1)), name="box", nu=0.3)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.given(S=lambda z: 1 + z ** 2)
    m.equations("dt(u) + grad(p) - nu*lap(u) = T*ez - u@grad(u)", "div(u) = 0",
                "dt(T) - nu*lap(T) = -u@grad(T) + S")
    m.at("walls", u=0, T=0)
    sim = m.simulate(N=(16, 16), scheme="RK222", dt=1e-3)
    sim.initial(T=lambda x, z: (1 + np.cos(x)) * (1 - z ** 2),
                u=lambda x, z: np.stack([z * (1 - z ** 2) + 0 * x, 0 * z + 0 * x]))
    x, z = sim.coordinate("x")[:, None], sim.coordinate("z")[None, :]
    return sim, x, z


def test_expressions_against_closed_forms():
    sim, x, z = _sim()
    T = (1 + np.cos(x)) * (1 - z ** 2)
    ux = z * (1 - z ** 2)
    checks = {"dz(T)": (1 + np.cos(x)) * (-2 * z), "T*T": T ** 2, "nu*dx(T)": -0.3 * np.sin(x) * (1 - z ** 2),
              "u@grad(T)": ux * (-np.sin(x)) * (1 - z ** 2), "0.5*u@u": 0.5 * ux ** 2,
              "S*T": (1 + z ** 2) * T}
    for text, exact in checks.items():
        g = sim.evaluate(text)
        assert g.shape == T.shape and np.abs(g - exact).max() < 1e-12, text
    v = sim.evaluate("T*ez")
    assert v.shape == (2,) + T.shape and np.abs(v[1] - T).max() < 1e-13 and np.abs(v[0]).max() == 0


def test_expressions_in_outputs_and_properties(tmp_path):
    sim, x, z = _sim()
    out = sim.output(tmp_path / "e.h5", {"dTdz": "dz(T)", "KE": "0.5*u@u"}, every=2)
    props = sim.properties(every=2, KE="0.5*u@u", T="T")
    sim.run(steps=4)
    d = cn.io.read(tmp_path / "e.h5")
    assert np.array_equal(d.tasks["dTdz"][-1], sim.evaluate("dz(T)"))
    assert "expression 0.5*u@u" in d.sources["KE"] and "expression" in out.explain()
    assert props.max("KE") == pytest.approx(float(np.max(sim.evaluate("0.5*u@u"))), abs=0)
    with pytest.raises(UndeclaredVariableError, match="'Tq' is not declared"):
        sim.output(tmp_path / "f.h5", {"bad": "dz(Tq)"}, every=1)


def test_curvilinear_refused():
    m = cn.Model(cn.Disk(1.0), name="disk")
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    m.at("wall", T=0)
    sim = m.simulate(N=(8, 8), scheme="RK222", dt=1e-3)
    with pytest.raises(UnsupportedError, match="Cartesian"):
        sim.evaluate("T*T")
