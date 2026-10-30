"""F756 (audit of 0.2.1, F26 (code side)): the declared growth rule of a phase speed says alpha Im c.

Oracle: the wording of the rule against exp(i alpha (x - c t)) = exp(alpha Im(c) t) ...
"""
from _acceptance import check, passed, start

start(756, "phase speed growth rule")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

import pathlib  # noqa: E402

import conatus.model.model as mm  # noqa: E402

src = pathlib.Path(mm.__file__).read_text()
check("growth rate = alpha Im" in src, "the rule still calls Im c the growth rate")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi, complex=True)), a=1.0)
m.scalar("u")
m.eigenvalue("c", growth="imag")
m.equations("c*u-(1+2j)*u=0")
r = m.eigenvalues(N=4, k=1, spurious="none")
check(abs(r.values[0] - (1 + 2j)) < 1e-12, f"{r.values}")
passed("alpha Im c")
