"""F502 (external review of 0.1.1, F3): Beyn merged close distinct roots into a fictitious double root.

(z - 0.2)(z - 0.2000001) and the pencil diag(0.2, 0.2 + delta): two simple roots. 0.1.1 returned
one value, 0.20000005 or 0.2000001, "multiplicity 2". Correct behaviour: both roots, or one value
explicitly reported as a cluster whose square holds both.
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(502, "close distinct roots are two roots, or a cluster that holds both")
b = cn.box(re=(-1, 1), im=(-1, 1))
a, c = 0.2, 0.2000001
r = cn.contour.beyn(lambda z: (z - a) * (z - c), b, poles=0, derivative=lambda z: 2 * z - a - c)
got = np.sort(r.values.real)
check(r.values.size == 2 and np.allclose(got, [a, c], rtol=0, atol=1e-9),
      f"scalar: values {r.values}, multiplicity {r.multiplicity} (exact {a}, {c})")
notes = []
for delta in (1e-3, 1e-5, 1e-7, 1e-9):
    exact = np.array([0.2, 0.2 + delta])
    r = cn.contour.beyn((-np.diag(exact), np.eye(2)), b)
    if r.values.size == 2:
        check(np.allclose(np.sort(r.values.real), exact, rtol=0, atol=1e-3 * delta),
              f"delta={delta:g}: values {r.values} (exact {exact})")
        notes.append(f"{delta:g}: 2 roots")
    else:
        enc = getattr(r, "enclosure", np.zeros(0))
        check(r.values.size == 1 and int(r.multiplicity[0]) == 2 and enc.size == 1
              and bool(np.all(np.abs(exact - r.values[0]) <= enc[0])),
              f"delta={delta:g}: one value {r.values} without a square holding both {exact}")
        notes.append(f"{delta:g}: cluster of 2 in half-side {enc[0]:.1e}")
passed("scalar: 2 roots; pencil " + ", ".join(notes))
