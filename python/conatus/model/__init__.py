"""Native layer: equations as on paper on named geometries, compiled to the IR.

    import conatus as cn
    box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
    m = cn.Model(box, Ra=2500, Pr=1)
    u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations('''
        dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        div(u) = 0
        dt(T) - lap(T) = -u·grad(T)
    ''')
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)

Tau terms, gauges and the implicit/explicit split are placed by fixed rules, listed by
``explain()``; ambiguous cases are refused with the override. The IR (``conatus.expert``) stays the
expert layer: ``m.compile(N).problem`` is an ordinary IR problem.

Stress-free walls: ``m.at("walls", u=cn.free_slip)``; sine/cosine wall axis:
``cn.walls(a, b, basis="sincos")`` (parities from the wall conditions, no tau). Compound interval
``cn.walls(a, c, b)`` (Chebyshev per piece, continuity at c automatic); whole line
``cn.line(scale=s)`` (Hermite); half line ``cn.half_line(scale=s)`` (Laguerre, one wall).
Channel between two surfaces (wavy wall, bump): ``cn.MappedBox(x=cn.periodic(L),
z=cn.between(bottom=lambda x: a*np.cos(x), top=1.0))`` (terrain-following map).
"""
from .axes import HalfLine, Line, half_line, line
from .compiler import Compiled, TauRecord, compile_model
from .conditions import NavierSlip, WallCondition, free_slip, navier_slip, no_slip
from .geometry import (DEFAULT_DEALIAS, Annulus, Ball, Box, Channel, Cylinder, Disk, Geometry, Periodic,
                       Pipe, Shell, Sphere, TaylorCouette, Walls, periodic, walls)
from .model import Given, Model, Unknown
from .run import Linearization, Simulation
from .terrain import Between, MappedBox, between
from .watch import Stat, View, Watch, profile, series, spectrum, view

__all__ = ["Box", "Channel", "Disk", "Annulus", "Pipe", "Cylinder", "TaylorCouette", "Sphere", "Ball", "Shell",
           "periodic", "walls", "Periodic", "Walls", "Geometry", "DEFAULT_DEALIAS", "Model",
           "Unknown", "Given", "Simulation", "Linearization", "Compiled", "TauRecord",
           "compile_model", "free_slip", "no_slip", "navier_slip", "NavierSlip", "WallCondition", "line", "half_line", "Line",
           "HalfLine", "view", "spectrum", "profile", "series", "View", "Stat", "Watch",
           "MappedBox", "between", "Between"]
