"""Data defined on fewer coordinates than their equation on curved geometries,
and the native grid accessors there.

* A forcing f(r) (or f(phi) on the annulus, f(r, z) in a pipe) in a polar volume equation failed at
  the first step with ``matmul: dimension mismatch`` (the defect of the spherical solver, in
  the polar one): the evaluator returns coefficients on the expression's domain. They are now
  embedded exactly in the equation's layout (the same function, constant along the missing
  coordinates); on the disk a function of phi alone is not regular at the centre and is refused.
* ``Simulation.coordinate`` used the Cartesian reference grids: NotImplementedError on every
  curved geometry.
"""
import numpy as np
import pytest

import conatus as cn
from conatus.discretize.compiler import DiscretizationError
from conatus.errors import InputTypeError


def _heat(geo, wall, f, N, steps=3):
    m = cn.Model(geo)
    m.scalar("T")
    m.given(f=f)
    m.equations("heat: dt(T) - lap(T) = f")
    m.at(wall, "T = 0")
    sim = m.simulate(N=N, scheme="SBDF2", dt=1e-3)
    sim.run(steps=steps)
    return sim


# a 1e-300 term keeps every coordinate in the declaration (an exactly constant direction is
# dropped by the native layer, declared); it is far below round-off of the answer
CASES = [
    ("disk r", cn.Disk(1.0), "wall", (8, 12),
     lambda phi, r: r**2 + 1e-300 * np.cos(phi) * r**2, lambda r: r**2),
    ("annulus r", cn.Annulus(1.0, 2.0), "walls", (8, 12),
     lambda phi, r: r**2 + 1e-300 * np.cos(phi), lambda r: r**2),
    ("annulus phi", cn.Annulus(1.0, 2.0), "walls", (8, 12),
     lambda phi, r: np.cos(phi) + 1e-300 * r, lambda phi: np.cos(phi)),
    ("pipe r z", cn.Pipe(1.0, 2.0), "wall", (4, 8, 4),
     lambda phi, r, z: r**2 * np.cos(np.pi * z) + 1e-300 * np.cos(phi) * r**2,
     lambda r, z: r**2 * np.cos(np.pi * z)),
]


@pytest.mark.parametrize("label, geo, wall, N, full, partial", CASES, ids=[c[0] for c in CASES])
def test_forcing_on_fewer_coordinates_is_embedded_exactly(label, geo, wall, N, full, partial):
    a = _heat(geo, wall, full, N).coeffs("T")
    b = _heat(geo, wall, partial, N).coeffs("T")
    assert np.abs(a).max() > 1e-4
    assert np.abs(a - b).max() < 1e-14 * np.abs(a).max()


def test_disk_function_of_phi_alone_is_refused():
    with pytest.raises(DiscretizationError, match="not regular at the centre"):
        _heat(cn.Disk(1.0), "wall", lambda phi: np.cos(phi), (8, 12))


def test_native_coordinates_on_curved_geometries():
    sim = _heat(cn.Disk(1.0), "wall", lambda r: r**2, (8, 12), steps=1)
    phi, r = sim.coordinate("phi"), sim.coordinate("r")
    assert sim.grid("T").shape == (phi.size, r.size)
    assert np.allclose(phi, 2 * np.pi * np.arange(phi.size) / phi.size) and 0 < r.min() < r.max() < 1
    with pytest.raises(InputTypeError, match="not a coordinate"):
        sim.coordinate("z")
