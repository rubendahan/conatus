"""F841 (hunt 5E, R2): the plan call forms of the guide exist on every plan type.

Oracle: the list of forms the documentation writes (plan.explain, methods, check, rank_report,
solve, resolution_report) and the documented scope of the resolution report (Cartesian plans
only: anything else is refused with UnsupportedError). Control: the Cartesian report of sin(x)
at N = 16 is resolved.
"""
import numpy as np
from _acceptance import check, passed, start

start(841, "every plan type has the documented call forms")
import conatus as cn  # noqa: E402
from conatus.errors import UnsupportedError  # noqa: E402

FORMS = ("explain", "methods", "check", "rank_report", "solve", "resolution_report")
GEOS = {"disk": (cn.Disk(1.0), ("wall",)), "annulus": (cn.Annulus(0.5, 1.0), ("walls",)),
        "sphere": (cn.Sphere(1.0), ()),
        "mapped": (cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0.0, top=1.0)),
                   ("bottom", "top"))}
for kind, (geo, walls) in GEOS.items():
    m = cn.Model(geo, name=kind, nu=1.0)
    m.scalar("T")
    m.equations("dt(T) - nu*lap(T) = 0")
    for w in walls:
        m.at(w, T=0)
    sim = m.simulate((8, 8), scheme="RK222", dt=1e-3)
    for form in FORMS:
        check(callable(getattr(sim.plan, form, None)), f"{kind}: plan.{form} missing")
    try:
        sim.plan.resolution_report(getattr(sim, "solver", None))
        check(False, f"{kind}: resolution report accepted")
    except UnsupportedError:
        pass
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
sim = m.simulate(N=16, scheme="RK443", dt=0.01)
sim.initial(u=lambda x: np.sin(x))
check(sim.plan.resolution_report(sim.solver).ok, "Cartesian control")
passed("4 plan types, 6 forms each")
