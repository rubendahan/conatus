"""F920 (pass 4, H3): Hill ``fl.leading`` is the leading exponent, not the one nearest the targets.

Oracle: v_t = v_xx + (a + e cos 2 pi t) v on (0, pi), v = 0 at the walls. The forcing is uniform
in x, so v = sin(n x) exp((a - n^2) t + (e / 2 pi) sin 2 pi t): exponents a - n^2 exactly.
With targets [0.0] (the AGENTS.md example) and a = 20, 100 the leading exponent 19, 99 is far
from the target; with targets [-9] and a = 1.5 the true 0.5 is unstable while -2.5 reads stable.
Control: n_modes counts distinct exponents (copies mu + 2 pi i n count once).
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(920, "Hill leading exponent is found far from the targets; n_modes counts distinct modes")
import conatus as cn  # noqa: E402


def lin(a, e):
    m = cn.Model(cn.Box(x=cn.walls(0, np.pi)), a=a)
    m.scalar("u")
    m.equations("dt(u)-dx(dx(u))-a*u=u*u")
    m.at("walls", "u=0")
    return m.linearize(about={"u": lambda x, t: e / 2 * np.cos(2 * np.pi * t) + 0 * x},
                       period=1.0, base_residual="accept")


for a, e, tg in ((20.0, 1.0, [0.0]), (100.0, 1.0, [0.0]), (1.5, 1.0, [-9.0]), (20.0, 1.0, [-9.0])):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fl = lin(a, e).floquet(32, method="hill", Nh=8, targets=tg, tol=1e-12, fields={"u": 1.0})
    check(abs(fl.leading.real - (a - 1)) < 1e-6,
          f"a = {a}, targets {tg}: leading {fl.leading:.8g}, oracle {a - 1}")
    check("all exponents" in fl.explain(), f"a = {a}: explain() does not state the scope")

# n_modes = distinct exponents nearest the target (a = 1.5, target 0: 0.5, -2.5, -7.5, -14.5)
fl = lin(1.5, 1.0).floquet(32, method="hill", Nh=8, targets=[0.0], n_modes=4, tol=1e-12,
                           fields={"u": 1.0})
want = np.array([0.5, -2.5, -7.5, -14.5])
got = np.sort(fl.mu.real)[::-1]
check(got.size == 4 and np.allclose(got, want, rtol=0, atol=1e-6), f"n_modes = 4: {got}, oracle {want}")
check(np.allclose(fl.mu.imag, 0, rtol=0, atol=1e-8), f"imaginary parts {fl.mu.imag}")

# above the dense size the scope is stated, never claimed
fl = lin(20.0, 1.0).floquet(32, method="hill", Nh=8, targets=[0.0], tol=1e-12, fields={"u": 1.0},
                            dense_check=10)
check("not excluded" in fl.explain(), "dense check skipped without a statement")
passed("leading exponent 19, 99, 0.5 and 19 found; n_modes counts distinct modes")
