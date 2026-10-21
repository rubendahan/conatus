"""F370: eigenvalues(method=typo) and shift-invert without target are refused before anything
is assembled (review of the inputs audit).

The typo was refused only after the group had been assembled and its rank report computed (and,
on the native layer, after the model had been compiled and discretized).
"""
from _acceptance import check, passed, refused, start

import conatus as cn
from conatus.expert import eigenvalues

start(370, "eigenvalue method checked before the assembly")
m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
m.scalar("v")
m.given(U=lambda y: 1 - y**2)
m.eigenvalue("c", growth="imag")
m.define("L2(f) = dy(dy(f)) - alpha**2*f")
m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")
plan = m.discretize(32)
r, err = refused(eigenvalues, plan, method="Dense")
check(r and "dense" in str(err), "method='Dense' not refused with the close name")
r, _ = refused(eigenvalues, plan, method="shift-invert")
check(r, "shift-invert without target accepted")
check(hasattr(plan, "_cache"), "plan has no _cache: the no-assembly check below would pass on {}")
cache = plan._cache
check(not any(isinstance(k, tuple) and k and k[0] == "full" for k in cache),
      "a group was assembled before the refusal")
for kw in ({"method": "Dense"}, {"method": "shift-invert"}, {"method": "shift-invert",
                                                              "target": "0.2"}):
    r, _ = refused(m.eigenvalues, 32, **kw)
    check(r, f"m.eigenvalues(32, **{kw}) accepted")
passed("refused before any assembly")
