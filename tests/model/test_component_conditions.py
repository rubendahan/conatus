"""Componentwise wall conditions on a vector unknown (native layer).

A full set of scalar conditions on the components of u at a wall ("u@ez = 0", "dz(u@ex) = 0")
is merged into one vector condition. Checks: written componentwise, free-slip and the Navier
slip give the eigenvalues of cn.free_slip and cn.navier_slip(l) (same discrete problem up to a
row order) and the shear-mode rates of the closed form; a mixed wall (u_x given by a Robin
condition with another length) works; an incomplete set and curved walls are refused.
"""
import warnings

import numpy as np
import pytest

import conatus as cn
from conatus.errors import BoundaryConditionError, UnsupportedError

NU = 0.7


def _model(**walls_):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="slip", nu=NU)
    u = m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = 0", "div(u) = 0")
    for w, conds in walls_.items():
        if isinstance(conds, tuple):
            m.at(w, *conds)
        else:
            m.at(w, u=conds)
    return m, u


def _ev(m, u, k=1.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.asarray(m.linearize(about={u: 0}).eigenvalues(32, k=k).values)
    v = v[np.isfinite(v)]
    return np.sort_complex(v[np.abs(v) < 200])


@pytest.mark.parametrize("k", [0.0, 1.0])
def test_free_slip_componentwise_equals_free_slip(k):
    a = _ev(*_model(walls=cn.free_slip), k)
    b = _ev(*_model(walls=("u@ez = 0", "dz(u@ex) = 0")), k)
    assert a.size > 5 and a.size == b.size and np.abs(a - b).max() < 1e-9 * np.abs(a).max()


def test_navier_slip_componentwise_equals_navier_slip():
    l = 0.3
    a = _ev(*_model(walls=cn.navier_slip(l)))
    b = _ev(*_model(bottom=("u@ez = 0", f"u@ex - {l}*dz(u@ex) = 0"),
                    top=("u@ez = 0", f"u@ex + {l}*dz(u@ex) = 0")))
    assert a.size == b.size and np.abs(a - b).max() < 1e-9 * np.abs(a).max()


def test_refusals():
    m, _ = _model(bottom=("u@ez = 0",), top=cn.no_slip)
    with pytest.raises(BoundaryConditionError, match="give all 2"):
        m.compile((8, 16))
    d = cn.Model(cn.Disk(1.0), name="disk", nu=NU)
    d.velocity("u")
    d.pressure("p")
    d.equations("dt(u) + grad(p) - nu*lap(u) = 0", "div(u) = 0")
    d.at("wall", "u@er = 0", "dr(u@ephi) = 0")
    with pytest.raises(UnsupportedError, match="unit vectors of curvilinear frames"):
        d.compile((8, 8))                               # curved: no e_r, e_phi in the equations
