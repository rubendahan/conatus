"""F368: reduced="no" and an unknown time_parameter on a polar IVP must be refused (review of
the inputs audit).

The polar and spherical systems read bool(reduced): "no" is True. Their solvers also kept a
time_parameter the equations do not use (a forcing in the real time parameter stayed frozen),
which the Cartesian solver already refused.
"""
from _acceptance import check, passed, refused, start

import conatus as cn
from conatus.expert import IVPSolver

start(368, "polar IVP switches and time parameter checked")
m = cn.Model(cn.Disk(1.0), name="disk heat")
m.scalar("T")
m.equations("dt(T) - lap(T) = 0")
m.at("wall", "T = 0")
plan = m.discretize((8, 8))
r, err = refused(IVPSolver, plan, "SBDF2", 1e-3, reduced="no")
check(r and "True or False" in str(err), "reduced='no' not refused as a switch")
r, _ = refused(IVPSolver, plan, "SBDF2", 1e-3, time_parameter="tt")
check(r, "time_parameter='tt' accepted on a polar IVP")
IVPSolver(plan, "SBDF2", 1e-3, reduced=False)
passed("refused; reduced=False accepted")
