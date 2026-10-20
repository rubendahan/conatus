"""F372: lin.resolvent / hinf_norm / h2_norm / optimal_gain arguments are typed refusals before
the operator is built (review of the inputs audit).

resolvent read float(omega): "0.5" was parsed and True was omega = 1; n_modes beyond the state
size was clipped (s[:nmodes]) and True read as 1; hinf_norm parsed text frequencies and kept
NaN; optimal_gain took T=True as a horizon of 1.

The resolvent family (resolvent, hinf_norm, h2_norm) is cut from the public build with its module
conatus.nonmodal.resolvent: those checks run where that module ships and are named as not shipped
otherwise (never a silent pass); optimal_gain is checked in every build.
"""
import importlib.util

import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(372, "native nonmodal arguments checked")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi, complex=True), y=cn.walls(-1, 1),
                    z=cn.periodic(2 * np.pi, complex=True)), name="pois", Re=1000.0)
m.velocity("u")
m.pressure("p")
m.equations("dt(u) - lap(u)/Re + grad(p) = -u@grad(u)", "div(u) = 0")
m.at("walls", u=0)
lin = m.linearize(about={"u": lambda y: np.stack(np.broadcast_arrays(1 - y**2, 0 * y, 0 * y))})
RESOLVENT = importlib.util.find_spec("conatus.nonmodal.resolvent") is not None
if RESOLVENT:
    conv = "exp(-i omega t)"
    for kw in ({"omega": "0.5", "n_modes": 1}, {"omega": True, "n_modes": 1},
               {"omega": 0.5, "n_modes": True}, {"omega": 0.5, "n_modes": 10**6}):
        r, _ = refused(lin.resolvent, 16, k=(1.0, 1.0), convention=conv, **kw)
        check(r, f"resolvent({kw}) accepted")
    r, _ = refused(lin.hinf_norm, 16, k=(0.0, 1.0), omegas=["0", "1"], xtol=1e-3, infinite_tol=1e-10)
    check(r, "hinf_norm(omegas=text) accepted")
    r, _ = refused(lin.h2_norm, 16, k=(0.0, 1.0), infinite_tol="1e-10")
    check(r, "h2_norm(infinite_tol=text) accepted")
    g = lin.resolvent(16, k=(1.0, 1.0), omega=np.float64(0.5), convention=conv, n_modes=np.int64(1))
    check(g.gains.size == 1, "a valid NumPy-typed call did not return one gain")
r, _ = refused(lin.optimal_gain, 16, k=(1.0, 1.0), t0=0.0, T=True, steps=4, scheme="RK443")
check(r, "optimal_gain(T=True) accepted")
passed("refused; NumPy scalars accepted" if RESOLVENT
       else "optimal_gain refused; the resolvent family is not shipped in this build")
