"""F731 (audit of 0.2.1, N10): vjp_physical promises the adjoint for a^H G b.

Oracle: the multiplier of RK111 on u_t + (1+2j) u = 0 by hand; the identity <w, J v>_G = <J^dag w, v>_G with a NumPy Hermitian G; the real case.
"""
from _acceptance import check, passed, start

start(731, "Hermitian physical adjoint")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

from conatus.adjoint import FlowMap  # noqa: E402

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi, complex=True)))
m.scalar("u")
m.equations("dt(u)+(1+2j)*u=0")
s = m.simulate(N=8, scheme="RK111", dt=.1)
s.initial(u=1.)
fm = FlowMap(s.solver, 3)
x = s.solver.X.copy()
w = np.ones_like(x)
got = fm.vjp_physical(x, w, np.eye(x.size))
mult = (1 + .1 * (1 + 2j)) ** -3             # RK111 with the linear term implicit
check(np.allclose(got, np.conj(mult) * w, rtol=1e-12, atol=0), f"{got[0]} against {np.conj(mult)}")
rng = np.random.default_rng(1)
A = rng.standard_normal((x.size, x.size)) + 1j * rng.standard_normal((x.size, x.size))
G = A.conj().T @ A + x.size * np.eye(x.size)
w = rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
v = rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
dag = fm.vjp_physical(x, w, G)
lhs = np.vdot(w, G @ (mult * v))
rhs = np.vdot(dag, G @ v)
check(abs(lhs - rhs) < 1e-11 * abs(lhs), f"Hermitian identity {lhs} against {rhs}")
b = fm.vjp(x, w)
check(np.allclose(b, mult * w, rtol=1e-12, atol=0), "the raw vjp keeps its bilinear pairing")
passed("Hermitian multiplier and identity")
