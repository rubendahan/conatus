"""Library of ready problems: canonical flows and model equations written with the public IR.

**Two layers, one name each**: ``cn.problems.<name>`` (this module) is the
*expert* version, a :class:`Ready` built at a resolution given in the call
(``cn.problems.rayleigh_benard(Nx=, Nz=, ...)``); ``cn.problems.native.<name>`` is the *native*
model of the same problem, a :class:`conatus.Model` or ``Linearization`` without resolution
(``cn.problems.native.rayleigh_benard(Lx=, Ra=, Pr=)`` then ``.simulate(N=...)``). Start with
the native one; the expert one is for reading or changing the discretization line by line.

Each function returns a :class:`Ready` whose ``.problem`` is an ordinary ``IVP`` or ``EVP``
(open the function: it is meant to be read, copied and modified). Every numerical choice
(resolution, dealias, domain size) and every physical parameter is a keyword argument without
default. The docstrings state the equations, units, boundary conditions and the published value
each problem is validated against (tests/problems/)::

    import conatus as cn
    r = cn.problems.channel_evp(N=64, Re=1e4, alpha=1.0)
    plan = cn.api.discretize(r.problem, ncc=r.ncc)
    print(r.explain())

 No EVP function where the linearization is
immediate (``linearize(r.problem, around=..., eigenvalue=...)``: Kuramoto–Sivashinsky,
Swift–Hohenberg) or not expressible (Kolmogorov: see its docstring).

The same problems written in the native layer (equations as on paper, no tau, resolution chosen at
``simulate``) are in :mod:`conatus.problems.native` (``cn.problems.native.rayleigh_benard(...)``,
...), with matrices identical bit for bit to these expert versions.
"""
from ._ready import Ready
from .convection import (rayleigh_benard, rayleigh_benard_evp, rotating_shell_convection,
                         rotating_shell_convection_evp)
from .cylinders import pipe, pipe_evp, taylor_couette, taylor_couette_evp
from .patterns import kolmogorov, kuramoto_sivashinsky, swift_hohenberg
from .shear import channel, channel_evp, plane_couette, plane_couette_evp
from . import native  # noqa: E402  (native-layer versions: cn.problems.native.<name>)

__all__ = ["Ready", "native", "channel", "channel_evp", "plane_couette", "plane_couette_evp",
           "rayleigh_benard", "rayleigh_benard_evp", "taylor_couette", "taylor_couette_evp",
           "pipe", "pipe_evp", "kuramoto_sivashinsky",
           "kolmogorov", "swift_hohenberg", "rotating_shell_convection",
           "rotating_shell_convection_evp", ]
