"""F365: model declarations refuse malformed values (audit inputs, 2026-10-20).

given(U=lambda x: x, rank=1.5 / True / -1 / 3) was stored and compiled as a scalar, given(U=nan)
was accepted, equations(3) was a bare AttributeError, parametric(over=()) / ("Re", "Re") built a
pencil and an unknown keyword of discretize/solve/parametric was a bare TypeError.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(365, "model declarations checked")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=1.0)
for kw in (dict(rank=1.5), dict(rank=True), dict(rank=-1), dict(rank=3)):
    r, _ = refused(m.given, U=lambda x: x, **kw)
    check(r, f"given({kw}) accepted")
r, _ = refused(m.given, U=float("nan"))
check(r, "given(U=nan) accepted")
r, _ = refused(m.equations, 3)
check(r, "equations(3) accepted")
os_ = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
os_.scalar("v")
os_.given(U=lambda y: 1 - y**2)
os_.eigenvalue("c", growth="imag")
os_.equations("dy(dy(dy(dy(v))))/(1j*alpha*Re) - U*dy(dy(v)) + dy(dy(U))*v + c*dy(dy(v)) = 0")
os_.at("walls", "v = 0", "dy(v) = 0")
for over in ((), ("Re", "Re"), 3):
    r, _ = refused(os_.parametric, 32, over=over)
    check(r, f"parametric(over={over!r}) accepted")
for call in (os_.parametric, os_.discretize):
    kw = dict(over=("Re",)) if call == os_.parametric else {}
    r, _ = refused(call, 32, foo=1, **kw)
    check(r, f"{call.__name__}(foo=1) accepted")
passed("refused")
