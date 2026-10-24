"""F501 (external review of 0.1.1, F1): an outside pole hid an inside zero from a "certified" count.

T(z) = (z - a) / (z - p), a = 0.123 + (1 - eps) i inside the box [-1, 1]^2 and p = 0.123 +
(1 + eps) i outside: one zero inside, so the count is 1. |T| = 1 on the top edge and the phase
turns by 2 pi within about eps: the sampled boundary walk returned 0 for eps <= 1e-3, and Beyn's
method said the count was "matched by certified roots". Correct behaviour: count 1, or a result
that states it is a sampled test whose scope excludes a zero-pole pair straddling the boundary.
Since 0.2 the walk follows the bend such a pair makes at the samples and counts 1;
Beyn's method may then refuse the root next to the pole (an error, never a wrong list).
"""
from _acceptance import check, passed, start

import conatus as cn
from conatus.errors import ConvergenceError

start(501, "a count that can miss a zero-pole pair does not call itself certified")
b = cn.box(re=(-1, 1), im=(-1, 1))
seen = []
for eps in (1e-2, 1e-3, 1e-5, 1e-8):
    a, p = 0.123 + (1 - eps) * 1j, 0.123 + (1 + eps) * 1j

    def T(z, a=a, p=p):
        return (z - a) / (z - p)

    c = cn.contour.count(T, b, poles=0)
    try:
        r = cn.contour.beyn(T, b, poles=0)
    except ConvergenceError:
        r = None
    if c.count == 1:
        if r is not None:
            check(r.values.size == 1 and abs(r.values[0] - a) < 1e-9,
                  f"eps={eps:g}: Beyn returned {r.values}, exact {a}")
        seen.append(f"eps={eps:g}: 1" + ("" if r is not None else " (Beyn refused)"))
        continue
    check(r is not None, f"eps={eps:g}: count {c.count} (exact 1) and Beyn refused")
    texts = (c.count_method, c.explain(), r.count_method, r.explain(), repr(r))
    check(not any("certif" in t.lower() for t in texts),
          f"eps={eps:g}: count {c.count} (exact 1) still described as certified: {r.count_method}")
    check("not a proof" in c.count_method and "pole" in c.count_method,
          f"eps={eps:g}: count {c.count} (exact 1) without its scope: {c.count_method}")
    seen.append(f"eps={eps:g}: {c.count}, scope stated")
passed("; ".join(seen))
