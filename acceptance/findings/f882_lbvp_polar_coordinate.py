"""F882 (hunt of 0.3.0, 2026-11-10, L3): LBVPSolution.coordinate on a disk.
It raised NotImplementedError although documented as Simulation.coordinate. Oracle: the
manufactured solution u = r^2 (1 - r^2) cos(2 phi), lap u = -12 r^2 cos(2 phi), u(1) = 0.
"""
from _acceptance import check, passed, start
start(882, "LBVPSolution.coordinate on a disk")
import numpy as np  # noqa: E402
import conatus as cn  # noqa: E402

d = cn.Model(cn.Disk(1.0), name="p")
d.scalar("u")
d.given(f=lambda phi, r: -12.0 * r ** 2 * np.cos(2 * phi))
d.equations("lap(u) = f")
d.at("wall", u=0)
sol = d.discretize((16, 16)).solve()
P, R = np.meshgrid(sol.coordinate("phi"), sol.coordinate("r"), indexing="ij")
err = float(np.max(np.abs(sol.grid("u") - R ** 2 * (1 - R ** 2) * np.cos(2 * P))))
check(err < 1e-12, f"max error on the coordinates {err:.1e}")
passed(f"grid matches the exact solution to {err:.1e}")
