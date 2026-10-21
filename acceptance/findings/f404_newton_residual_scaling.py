"""F404 (audit invariance): Newton's relative residual does not depend on the scale of an
equation.

Bratu u'' + lam e^u = 0 written times 1e-8: the relative residual divided every row by one
global scale, so the O(1) boundary rows' rounding (1e-19) was measured against the 1e-9 bulk
scale and Newton stalled at 6e-11 > tol = 1e-12 (NewtonNotConverged) while the unscaled
equation converges in 4 iterations.
"""
import numpy as np
from _acceptance import check, describe, failed, passed, start

import conatus as cn

start(404, "Newton convergence invariant under equation scaling")
from conatus.expert import NewtonSolver  # noqa: E402


def solve(c):
    m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
    m.scalar("u")
    m.equations(f"bratu: {c!r}*dx(dx(u)) + {c!r}*lam*exp(u) = 0")
    m.at("walls", u=0)
    pb = m.compile(48, dealias=1)
    s = NewtonSolver(pb.problem, tol=1e-12, max_iterations=20, params={"lam": 0.3})
    return s.solve({pb.fields["u"]: lambda x: 0 * x})


def bratu_error(sol, lam=0.3):
    """max |u - u_exact| on the grid, u_exact = 2 ln(cosh(theta) / cosh(theta x)) the lower Bratu
    branch, theta the smaller root of theta = sqrt(lam / 2) cosh(theta) (bisection on [0, 1]).
    An independent anchor (review 2026-10-21, F14): the scaled solves are not only compared with
    the unscaled one."""
    from conatus.reference import transforms as T
    dom = sol.plan.domain
    x = np.asarray(T.grid(dom.basis(dom.coordsys.coords[0]), dealias=False))
    a, lo, hi = np.sqrt(lam / 2), 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid - a * np.cosh(mid) < 0 else (lo, mid)
    th = 0.5 * (lo + hi)
    return float(np.max(np.abs(np.asarray(sol.grid("u")) - 2 * np.log(np.cosh(th) / np.cosh(th * x)))))


ref = solve(1.0)
check(bratu_error(ref) < 1e-10, f"unscaled Bratu solution off the exact one by {bratu_error(ref):.2e}")
for c in (1e-8, 1e-4, 1e4):
    try:
        sol = solve(c)
    except Exception as exc:  # noqa: BLE001 -- any exception is the failure reported
        failed(f"equation x {c:g}: {describe(exc)}")
    check(len(sol.iterations) == len(ref.iterations),
          f"equation x {c:g}: {len(sol.iterations)} iterates, {len(ref.iterations)} unscaled")
    check(float(np.max(np.abs(sol.grid("u") - ref.grid("u")))) < 1e-12, f"equation x {c:g}: other u")
passed(f"{len(ref.iterations) - 1} iterations at every scale")
