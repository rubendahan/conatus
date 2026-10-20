"""F403 (audit invariance): the rank check does not refuse a well-posed system because one
equation is written times a large constant.

Bratu u'' + lam e^u = 0, u(+-1) = 0, written times 1e8: the Newton Jacobian was refused before
its first factorization ("numerically singular, missing gauge condition") because the rank
report compared singular values of the unequilibrated matrix (O(1) boundary rows next to 1e8
rows). The same holds for the linear solve of an LBVP.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(403, "rank check invariant under equation scaling")
from conatus.expert import NewtonSolver  # noqa: E402


def solve(c):
    m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
    m.scalar("u")
    m.equations(f"bratu: {c!r}*dx(dx(u)) + {c!r}*lam*exp(u) = 0")
    m.at("walls", u=0)
    pb = m.compile(48, dealias=1)
    s = NewtonSolver(pb.problem, tol=1e-12, max_iterations=20, params={"lam": 0.3})
    return s.solve({pb.fields["u"]: lambda x: 0 * x})


ref = solve(1.0)
was_refused, sol = refused(solve, 1e8)
check(not was_refused, "equation x 1e8 refused as rank deficient")
err = float(np.max(np.abs(sol.grid("u") - ref.grid("u"))))
check(err < 1e-12, f"solution differs by {err:.2e}")
passed(f"converged in {len(sol.iterations) - 1} iterations, |du| = {err:.1e}")
