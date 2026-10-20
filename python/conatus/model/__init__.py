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
from .axes import HalfLine, Line, half_line, line  # noqa: F401 -- importable here, not in __all__ (M5)
from .compiler import Compiled, TauRecord, compile_model  # noqa: F401 -- importable here, not in __all__ (M5)
from .conditions import NavierSlip, WallCondition, free_slip, navier_slip, no_slip  # noqa: F401 -- importable here, not in __all__ (M5)
from .geometry import (DEFAULT_DEALIAS, Annulus, Ball, Box, Channel, Cylinder, Disk, Geometry, Periodic,  # noqa: F401 -- importable here, not in __all__ (M5)
                       Pipe, Shell, Sphere, TaylorCouette, Walls, periodic, walls)  # noqa: F401 -- importable here, not in __all__ (M5)
from .model import Given, Model, Unknown  # noqa: F401 -- importable here, not in __all__ (M5)
from .run import Linearization, Simulation
from .terrain import Between, MappedBox, between  # noqa: F401 -- importable here, not in __all__ (M5)
from .watch import Stat, View, Watch, profile, series, spectrum, view  # noqa: F401 -- importable here, not in __all__ (M5)

# Documented names only (M5, 0.2); the others stay importable from this module.
__all__ = ["Box", "Channel", "Disk", "Annulus", "Pipe", "Cylinder", "TaylorCouette", "Sphere",
           "Ball", "Shell", "periodic", "walls", "Periodic", "Geometry", "Model", "Unknown",
           "Simulation", "Linearization", "Compiled", "free_slip", "no_slip", "navier_slip", "line",
           "half_line", "view", "spectrum", "profile", "series", "MappedBox", "between"]
