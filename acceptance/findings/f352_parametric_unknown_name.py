"""F352: a pencil value under an unknown name must be refused (audit inputs, 2026-10-20).

P = m.parametric(N, over=("Re",)); P.least_stable(re=5000) ignored `re` and solved at the
model's declared Re = 1e4 (the value filled in for the missing Re), with no warning.
"""
from _acceptance import check, passed, refused, start

import conatus as cn

start(352, "unknown parameter name of a pencil refused")
m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
m.scalar("v")
m.given(U=lambda y: 1 - y**2)
m.eigenvalue("c", growth="imag")
m.define("L2(f) = dy(dy(f)) - alpha**2*f")
m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")
P = m.parametric(32, over=("Re",))
r, exc = refused(P.least_stable, re=5000.0)
check(r and "re" in str(exc), "least_stable(re=5000) accepted (the declared Re used)")
r, _ = refused(P.least_stable, Re=float("nan"))
check(r, "least_stable(Re=nan) accepted")
passed("unknown name and NaN value refused")
