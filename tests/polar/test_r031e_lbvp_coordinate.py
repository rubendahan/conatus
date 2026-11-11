"""LBVPSolution.coordinate on curvilinear plans (hunt of 0.3.0).

It raised NotImplementedError ("reference transforms: DiskBasis not supported yet") on a disk or
an annulus although its docstring promised the grid of Simulation.coordinate. Oracle: the
manufactured solutions u = r^m (1 - r^2) cos(m phi) (lap u = -(4m + 4) r^m cos(m phi), u(1) = 0)
and u = (r - 1)(2 - r) on the annulus 1 < r < 2 (lap u = 3 - 4 + ... closed form below).
"""
import numpy as np
import pytest

import conatus as cn


@pytest.mark.parametrize("mm", [0, 2, 3])
def test_disk_lbvp_coordinates_evaluate_the_exact_solution(mm):
    d = cn.Model(cn.Disk(1.0), name="p")
    d.scalar("u")
    d.given(f=lambda phi, r: -(4 * mm + 4) * r ** mm * np.cos(mm * phi))
    d.equations("lap(u) = f")
    d.at("wall", u=0)
    sol = d.discretize((16, 16)).solve()
    phi, r = sol.coordinate("phi"), sol.coordinate("r")
    P, R = np.meshgrid(phi, r, indexing="ij")
    g = sol.grid("u")
    assert g.shape == P.shape
    assert np.max(np.abs(g - R ** mm * (1 - R ** 2) * np.cos(mm * P))) < 1e-12
    phi[0] = 99.0                                   # a copy: the plan's grid is not changed
    assert sol.coordinate("phi")[0] == 0.0


def test_annulus_lbvp_coordinates():
    """u = (r - 1)(2 - r): lap u = u'' + u'/r = -2 + (3 - 2r)/r = 3/r - 4, u = 0 at r = 1 and 2."""
    d = cn.Model(cn.Annulus(1.0, 2.0), name="a")
    d.scalar("u")
    d.given(f=lambda phi, r: 3.0 / r - 4.0 + 0.0 * phi)
    d.equations("lap(u) = f")
    d.at("inner", u=0)
    d.at("outer", u=0)
    sol = d.discretize((8, 32)).solve()
    phi, r = sol.coordinate("phi"), sol.coordinate("r")
    assert 1.0 < r.min() < r.max() < 2.0 and phi.size == 8
    P, R = np.meshgrid(phi, r, indexing="ij")
    assert np.max(np.abs(sol.grid("u") - (R - 1) * (2 - R))) < 1e-11


def test_unknown_coordinate_is_refused():
    d = cn.Model(cn.Disk(1.0), name="p")
    d.scalar("u")
    d.equations("lap(u) = 1")
    d.at("wall", u=0)
    sol = d.discretize((8, 8)).solve()
    with pytest.raises(KeyError):
        sol.coordinate("z")
