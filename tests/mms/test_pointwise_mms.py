"""MMS with pointwise functions (exp, tanh): IR -> sympy translation, NLBVP and LBVP convergence,
and the aliasing caveat reported by the problem, the manufactured solution and plan.methods()."""
import numpy as np
import pytest
import sympy as sy

import conatus.methods  # noqa: F401  (adds Plan.methods)
from conatus.discretize import discretize
from conatus.ir import (LBVP, NLBVP, CartesianCoordinates, Chebyshev, Domain, Field, diff, exp,
                        interp, lift, tanh)
from conatus.reference import transforms as T
from conatus.solvers.nlbvp import NewtonSolver
from conatus.verify.mms import manufacture, observed_order, to_sympy

X = sy.Symbol("x", real=True)
U_EXACT = sy.Rational(3, 10) * sy.sin(sy.pi * X) + X / 5 + sy.Rational(1, 10)


def test_to_sympy_pointwise_functions():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [Chebyshev(x, 8, (-1, 1), dealias=1)])
    u = Field("u", D)
    e = to_sympy(exp(u) + tanh(diff(u, x)), {u: sy.sin(X)})
    assert sy.simplify(e - (sy.exp(sy.sin(X)) + sy.tanh(sy.cos(X)))) == 0


def _nlbvp(N):
    """u'' + exp(u) + tanh(u) = f on [-1, 1], Dirichlet data: nonlinear in u through exp and tanh."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, N, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    D0 = D.without(x)
    u, t1, t2 = Field("u", D), Field("t1", D0), Field("t2", D0)
    f, gl, gr = Field("f", D), Field("gl", D0), Field("gr", D0)
    pb = NLBVP([u, t1, t2], name="exp-tanh MMS")
    pb.add_equation((diff(diff(u, x), x) + exp(u) + tanh(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, -1), gl))
    pb.add_equation((interp(u, x, 1), gr))
    return pb, b, x, u, (f, gl, gr)


# Convergence studies in N: the coarse points are under-resolved on purpose (exp(u) at N = 8: tail
# 2e-2), the measured error includes that truncation and must still converge to 1e-12; the
# accepted coefficient tail is declared, the convergence criteria are unchanged.
STUDY = {"ncc_tail_tol": 0.1}


def _solve_nlbvp(N):
    pb, b, x, u, data = _nlbvp(N)
    man = manufacture(pb, {u: U_EXACT})
    ncc = {fld: man.callable(r, [x]) for fld, r in zip(data, man.rhs)}
    newton = NewtonSolver(pb, tolerance=1e-13, damping=1.0, max_iterations=20, ncc=ncc, **STUDY)
    sol = newton.solve({u: lambda xg: 0 * xg})
    xg = T.grid(b, dealias=False)
    err = float(np.max(np.abs(sol.grid("u") - sy.lambdify(X, U_EXACT)(xg))))
    return pb, man, sol, err


@pytest.mark.slow
def test_nlbvp_mms_exp_tanh_converges_spectrally():
    Ns, errs = [8, 12, 16, 20, 24, 32], []
    for N in Ns:
        _, _, sol, err = _solve_nlbvp(N)
        assert sol.converged
        errs.append(err)
    fit = observed_order(Ns[:4], errs[:4])
    assert fit["rate"] > 0.5, (errs, fit["report"])          # exponential decay in N
    assert errs[-1] < 1e-12, errs


@pytest.mark.slow                   # 2.2 s in the fast suite (qual_durations0)
def test_aliasing_caveat_is_reported():
    pb, man, sol, _ = _solve_nlbvp(16)
    assert man.nonpolynomial == ("exp", "tanh")
    assert "aliasing error" in man.explain() and "convergence in N" in man.explain()
    assert "NOT alias-free" in pb.explain()
    methods = sol.plan.methods()
    text = methods if isinstance(methods, str) else getattr(methods, "text", str(methods))
    assert "dealiasing is not exact" in text


@pytest.mark.slow
def test_lbvp_mms_with_exp_coefficient():
    """Linear problem with a non-polynomial coefficient exp(a(x)) built from a data field."""
    errs = []
    Ns = [8, 12, 16, 24]
    for N in Ns:
        cs = CartesianCoordinates("x")
        x = cs["x"]
        b = Chebyshev(x, N, (-1, 1), dealias=1)
        D = Domain(cs, [b])
        D0 = D.without(x)
        u, a, f = Field("u", D), Field("a", D), Field("f", D)
        t1, t2, gl, gr = Field("t1", D0), Field("t2", D0), Field("gl", D0), Field("gr", D0)
        pb = LBVP([u, t1, t2])
        pb.add_equation((diff(diff(u, x), x) - exp(a) * u + lift(t1, b, -1) + lift(t2, b, -2), f))
        pb.add_equation((interp(u, x, -1), gl))
        pb.add_equation((interp(u, x, 1), gr))
        a_expr = sy.sin(X) / 2
        man = manufacture(pb, {u: U_EXACT, a: a_expr})
        ncc = {a: man.callable(a_expr, [x])}
        data = {fld: man.callable(r, [x]) for fld, r in zip((f, gl, gr), man.rhs)}
        sol = discretize(pb, ncc=ncc, **STUDY).solve(data)
        xg = T.grid(b, dealias=False)
        errs.append(float(np.max(np.abs(sol.grid("u") - sy.lambdify(X, U_EXACT)(xg)))))
    assert errs[-1] < 1e-11 and errs[-1] < 1e-6 * errs[0], errs


def test_unknown_function_is_refused_not_skipped(monkeypatch):
    from conatus.verify import mms
    monkeypatch.delitem(mms._SYMPY_FUNCS, "tanh")
    cs = CartesianCoordinates("x")
    x = cs["x"]
    u = Field("u", Domain(cs, [Chebyshev(x, 8, (-1, 1), dealias=1)]))
    with pytest.raises(NotImplementedError, match="tanh"):
        to_sympy(tanh(u), {u: X})
