"""F252 (audit "assurances", 2026-10-20): critical(over=) says when its minimum is at the edge.

Swift-Hohenberg, r_c(q) = (1 - q^2)^2 with minimum r_c = 0 at q = 1. Over q in (1.2, 1.5) the
bounded minimizer stops at q = 1.2 and 0.1.0 returned r_c = 0.1936 as the critical point,
verified=True, without a word.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(252, "critical(over=) minimum at the end of the interval")
from conatus.expert import (IVP, CartesianCoordinates, ComplexFourier, Domain, Field,  # noqa: E402
                            Parameter, ParametricEVP, Zero, critical, diff, dt, linearize)

cs = CartesianCoordinates("x")
x = cs["x"]
D = Domain(cs, [ComplexFourier(x, 8, (0, 2 * np.pi), dealias=1)])
u, r = Field("u", D), Parameter("r", 0.1)
d2 = lambda f: diff(diff(f, x), x)                        # noqa: E731
ivp = IVP([u])
ivp.add_equation((dt(u) - r * u + (u + d2(u)) + d2(u + d2(u)), -u * u * u))
evp = linearize(ivp, around={u: Zero(())}, eigenvalue=Parameter("s", dtype=complex))
P = ParametricEVP(evp, [r], wavenumbers={"x": Parameter("q", 1.0)})
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cp = critical(P, "r", bracket=(-0.5, 1.0), over={"q": (1.2, 1.5)}, xtol=1e-10)
said = any("edge" in str(m.message) or "end q" in str(m.message) for m in w) and \
    "EDGE" in repr(cp).upper()
check(said, f"{cp!r} returned without saying the minimum is at the end of over")
passed(f"r = {cp.value:.4g} at q = {cp.over['q']:.6g}, at the edge: said")
