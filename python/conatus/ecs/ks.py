"""Kuramoto–Sivashinsky reference model for the ECS workflow.

u_t = −u u_x − u_xx − u_xxxx on a periodic interval of length L, as in Cvitanović, Davidchack &
Siminos (2010, SIAM J. Appl. Dyn. Syst. 9, eq. 2.1). Written with the Conatus IR and solved by
:class:`IVPSolver` on CPU; every numerical choice is an argument (no default resolution or step).
"""
from __future__ import annotations

from .._deprecation import renames
from ..discretize import discretize
from ..ir import IVP, CartesianCoordinates, Domain, Field, RealFourier, diff, dt, lap
from ..timestep import IVPSolver

__all__ = ["ks_solver"]


@renames("conatus.ecs.ks:ks_solver")
def ks_solver(*, L: float, N: int, dealias: float, scheme: str,
              timestep: float | None = None) -> IVPSolver:
    """IVPSolver for KS on (0, L) with RealFourier(N); ``timestep`` is the initial step (ECS
    changes it). ``dt_``: deprecated former name of ``timestep`` (v0.1)."""
    from .._deprecation import given            # dt_= (v0.1 name) is mapped by @renames
    dt_ = given("ks_solver", "timestep", timestep)
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, N, (0, L), dealias=dealias)])
    u = Field("u", D)
    pb = IVP([u], name=f"Kuramoto-Sivashinsky L={L:g}")
    pb.add_equation((dt(u) + lap(u) + lap(lap(u)), -u * diff(u, x)))
    return IVPSolver(discretize(pb), scheme, dt_)
